"""
관리자 AI 에이전트 LangGraph StateGraph 구성 + SSE 실행 인터페이스.

설계서: docs/관리자_AI에이전트_설계서.md §3.2 (그래프), §8.2 (SSE 10이벤트)

Step 2 범위 (2026-04-23, 확장):
    START → context_loader → intent_classifier ──┐
                                                  │
    ┌────────── smalltalk ─────────────────┤
    ▼                                       │
    smalltalk_responder                     │ stats
    │                                       ▼
    │                                tool_selector ──┐
    │                                       │        │ pending=None
    │                              pending  ▼        │
    │                                tool_executor   │
    │                                       ▼        │
    │                                   narrator     │
    │                                       │        │
    ▼                                       ▼        ▼
    response_formatter ◀──── query/action/report/sql
         ▼
        END

후속 Step 에서 추가될 엣지:
    tool_selector → risk_gate (Tier ≥ 2) → HITL interrupt → tool_executor
    narrator → data_analyzer (추가 tool 필요 여부) → tool_selector 루프 (최대 5회)

SSE 이벤트 (Step 2 에서 발행하는 것):
- session     : 세션 ID 발급
- status      : 노드 진행 상태
- tool_call   : tool_selector 완료 — {tool_name, arguments, tier}
- tool_result : tool_executor 완료 — {tool_name, ok, latency_ms, row_count}
- token       : 최종 응답 텍스트 (narrator or response_formatter)
- done / error

후속 Step 에서 추가: confirmation_required / chart_data / table_data / report_chunk
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from monglepick.agents.admin_assistant.models import (
    AdminAssistantState,
    AdminIntent,
    ToolCall,
)
from monglepick.agents.admin_assistant.nodes import (
    context_loader,
    intent_classifier,
    narrator,
    response_formatter,
    risk_gate,
    smalltalk_responder,
    tool_executor,
    tool_selector,
)
from monglepick.api.admin_backend_client import AdminApiResult

logger = structlog.get_logger()


# ============================================================
# 라우팅 — Step 1 에서는 smalltalk vs 그 외 2분기
# ============================================================

def route_after_intent(state: AdminAssistantState) -> str:
    """
    Intent 분류 이후 분기 (Step 2 확장).

    - admin_role 이 비어있으면 → response_formatter 직행 (차단 메시지).
    - smalltalk → smalltalk_responder.
    - **stats** → tool_selector (실제 Admin Stats API 호출 경로).
    - query/action/report/sql → response_formatter (현재 placeholder; Step 3+ 에서 확장).

    v2 설계에서 sql 은 영구 미지원이라 placeholder. query/action/report 는 Tier 1/2/3
    tool 추가 후 순차적으로 tool_selector 에 붙는다.
    """
    admin_role = state.get("admin_role", "") or ""
    if not admin_role:
        logger.info("route_after_intent_blocked", reason="no_admin_role")
        return "response_formatter"

    intent = state.get("intent")
    kind = intent.kind if isinstance(intent, AdminIntent) else "smalltalk"

    if kind == "smalltalk":
        return "smalltalk_responder"
    # Step 2: stats → tool_selector
    # Step 4(2026-04-23): query 도 tool_selector.
    # Step 5a(2026-04-23): action 도 tool_selector — HITL(risk_gate) 게이트가 있어 안전.
    if kind in ("stats", "query", "action"):
        return "tool_selector"
    # report/sql 은 여전히 placeholder (report=Phase 4 / sql=영구 미지원)
    return "response_formatter"


def route_after_tool_select(state: AdminAssistantState) -> str:
    """
    tool_selector 이후 분기 (Step 5a 수정).

    - pending_tool_call 이 있으면 **risk_gate** 로 (Tier 0/1 은 risk_gate 가 즉시 통과시키고,
      Tier 2/3 는 interrupt 발동). 이 변경으로 쓰기 작업(Tier 2+)은 반드시 HITL 을 지나게 된다.
    - 없으면 (적합한 tool 없음 / 권한 없음 / LLM 에러) response_formatter 에서 placeholder.
    """
    call = state.get("pending_tool_call")
    if isinstance(call, ToolCall):
        logger.info(
            "route_after_tool_select",
            tool_name=call.tool_name,
            tier=call.tier,
        )
        return "risk_gate"
    logger.info("route_after_tool_select_no_tool")
    return "response_formatter"


def route_after_risk_gate(state: AdminAssistantState) -> str:
    """
    risk_gate 이후 분기 (Step 5a).

    - pending_tool_call 이 None 으로 비워졌다면 사용자가 거절했거나 에러 → response_formatter.
    - 여전히 ToolCall 이면 승인(또는 Tier<2 로 통과) → tool_executor.
    """
    call = state.get("pending_tool_call")
    if isinstance(call, ToolCall):
        return "tool_executor"
    logger.info("route_after_risk_gate_rejected_or_missing")
    return "response_formatter"


# ============================================================
# 그래프 빌드
# ============================================================

def build_admin_assistant_graph():
    """
    Admin Assistant StateGraph 구성 + 컴파일.

    Step 5a (2026-04-23): 8노드 + 3개 조건부 분기 + MemorySaver checkpointer.
        risk_gate 가 Tier≥2 에서 LangGraph `interrupt()` 를 호출하려면 checkpointer 가 필요하다.
        MemorySaver 는 프로세스 메모리 기반이므로 재기동 시 흐름이 사라진다 — 운영 레벨은
        RedisSaver/PostgresSaver 로 교체 권장(후속 Step). 다만 Step 5a 는 ephemeral 세션
        모델이라 `run_admin_assistant` → `/resume` 라이프사이클이 같은 프로세스 내에서 완결된다.
    """
    graph = StateGraph(AdminAssistantState)

    graph.add_node("context_loader", context_loader)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("smalltalk_responder", smalltalk_responder)
    graph.add_node("tool_selector", tool_selector)
    graph.add_node("risk_gate", risk_gate)  # Step 5a 신규
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("narrator", narrator)
    graph.add_node("response_formatter", response_formatter)

    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "intent_classifier")
    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "smalltalk_responder": "smalltalk_responder",
            "tool_selector": "tool_selector",
            "response_formatter": "response_formatter",
        },
    )
    graph.add_edge("smalltalk_responder", "response_formatter")

    # tool_selector → risk_gate 또는 response_formatter
    graph.add_conditional_edges(
        "tool_selector",
        route_after_tool_select,
        {
            "risk_gate": "risk_gate",
            "response_formatter": "response_formatter",
        },
    )
    # risk_gate → tool_executor (승인/통과) 또는 response_formatter (거절)
    graph.add_conditional_edges(
        "risk_gate",
        route_after_risk_gate,
        {
            "tool_executor": "tool_executor",
            "response_formatter": "response_formatter",
        },
    )
    graph.add_edge("tool_executor", "narrator")
    graph.add_edge("narrator", "response_formatter")

    graph.add_edge("response_formatter", END)

    # MemorySaver checkpointer — interrupt/resume 을 위해 필수
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("admin_assistant_graph_compiled", node_count=8, checkpointer="memory")
    return compiled


# 모듈 레벨 싱글턴 — 컴파일 1회
admin_assistant_graph = build_admin_assistant_graph()


# ============================================================
# 노드 → 한국어 status 메시지
# ============================================================

_NODE_STATUS_MESSAGES: dict[str, str] = {
    "context_loader": "관리자 정보를 확인하고 있어요...",
    "intent_classifier": "요청 의도를 분석하고 있어요...",
    "smalltalk_responder": "답변을 준비하고 있어요...",
    "tool_selector": "적합한 도구를 고르고 있어요... 🧰",
    "risk_gate": "실행 전 안전 점검 중이에요... 🛡️",
    "tool_executor": "관리자 API를 호출하고 있어요... 🔌",
    "narrator": "결과를 정리해 설명하고 있어요...",
    "response_formatter": "응답을 정리하고 있어요...",
}


# ============================================================
# SSE 유틸
# ============================================================

_KEEPALIVE_INTERVAL_SEC = 15
_SENTINEL = object()


def _format_sse_event(event_type: str, data: dict) -> dict:
    """
    sse_starlette 호환 dict 포맷.

    Chat Agent graph.py 의 _format_sse_event 와 동일 규약.
    EventSourceResponse 가 {"event": ..., "data": ...} dict 를 받으면
    "event: {type}\\ndata: {json}\\n\\n" 로 직렬화한다.
    """
    return {"event": event_type, "data": json.dumps(data, ensure_ascii=False)}


def state_snapshot_tool_call(merged_state: dict) -> ToolCall | None:
    """
    merged state 에서 현재 `pending_tool_call` 을 안전히 꺼낸다.

    SSE 발행 시점에는 `updates` 에만 최신 값이 있고 `final_state` 는 누적본이라,
    tool_executor 완료 이벤트 쪽에서 tool 이름을 참조하려면 이 헬퍼로 꺼낸다.
    """
    call = merged_state.get("pending_tool_call")
    return call if isinstance(call, ToolCall) else None


# ============================================================
# SSE 스트리밍 실행
# ============================================================

async def run_admin_assistant(
    admin_id: str,
    admin_role: str,
    admin_jwt: str,
    session_id: str,
    user_message: str,
    resume_payload: dict | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Admin Assistant 를 SSE 스트리밍 모드로 실행한다.

    Chat Agent run_chat_agent() 와 동일한 asyncio.Queue + keepalive 패턴.
    - 15초 동안 이벤트가 없으면 keepalive status 발행 (SSE 연결 유지)
    - 그래프 완료 시 done / 에러 시 error → done 순 발행
    - Step 5a: HITL interrupt 발생 시 `confirmation_required` 발행 후 done 없이 스트림 종료

    Args:
        admin_id: JWT sub (관리자 user_id)
        admin_role: 정규화된 AdminRoleEnum 문자열 (빈 문자열이면 차단 메시지)
        admin_jwt: Backend forwarding 용 JWT 원문 (tool_executor 에서 사용)
        session_id: 세션 ID (빈 문자열이면 자동 생성). HITL 재개 시에는 반드시 기존 세션 ID 필수.
        user_message: 관리자 발화 (resume 모드에서는 빈 문자열/무시됨)
        resume_payload: 재개용 `Command(resume=...)` 에 들어갈 dict.
            None 이면 신규 대화, dict 면 `/resume` 경로로 기존 thread_id 에서 재개한다.
            기대 shape: {"decision": "approve"|"reject", "comment": str}

    Yields:
        sse_starlette 호환 dict. {"event": ..., "data": json_string}
    """
    graph_start = time.perf_counter()

    # 세션 ID 자동 생성 (resume 모드에서는 호출 측이 기존 session_id 전달해야 함)
    if not session_id:
        session_id = str(uuid.uuid4())

    # LangGraph checkpointer 구분 키 — 세션 단위로 체크포인트 네임스페이스 분리
    graph_config = {"configurable": {"thread_id": session_id}}

    # resume 이면 initial_state 대신 Command(resume=...) 를 astream 에 전달.
    # 신규 대화는 기존처럼 state dict.
    is_resume = resume_payload is not None
    initial_state: AdminAssistantState | None
    if is_resume:
        initial_state = None  # 실행 입력은 Command 로 대체
    else:
        initial_state = {
            "admin_id": admin_id,
            "admin_role": admin_role,
            "admin_jwt": admin_jwt,
            "session_id": session_id,
            "user_message": user_message,
            "history": [],
        }

    logger.info(
        "admin_assistant_stream_start",
        admin_id=admin_id or "(anonymous)",
        admin_role=admin_role or "(blank)",
        session_id=session_id,
        is_resume=is_resume,
        message_preview=("(resume)" if is_resume else user_message[:100]),
    )

    # session 이벤트 발행
    yield _format_sse_event("session", {"session_id": session_id})

    queue: asyncio.Queue = asyncio.Queue()
    # resume 모드면 첫 노드는 risk_gate (interrupt 직후 지점) — 그 외 신규는 context_loader
    current_phase = "risk_gate" if is_resume else "context_loader"
    current_message = _NODE_STATUS_MESSAGES[current_phase]
    final_state: dict = {}

    async def _run_graph_to_queue():
        """LangGraph astream → Queue → SSE 소비 패턴."""
        try:
            # 신규 대화: initial_state dict / 재개: Command(resume=...).
            graph_input = (
                Command(resume=resume_payload) if is_resume else initial_state
            )
            async for event in admin_assistant_graph.astream(
                graph_input,
                config=graph_config,
                stream_mode="updates",
            ):
                await queue.put(event)
            await queue.put(_SENTINEL)
        except Exception as e:
            await queue.put(e)

    graph_task = asyncio.create_task(_run_graph_to_queue())

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=_KEEPALIVE_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                yield _format_sse_event(
                    "status",
                    {"phase": current_phase, "message": current_message, "keepalive": True},
                )
                continue

            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item

            # {"node_name": {updates}}
            for node_name, updates in item.items():
                final_state.update(updates)

                # 노드 완료 status
                completed_msg = _NODE_STATUS_MESSAGES.get(
                    node_name, f"{node_name} 처리 중...",
                )
                yield _format_sse_event(
                    "status", {"phase": node_name, "message": completed_msg},
                )

                # 다음 노드 예측해서 keepalive 메시지 갱신
                next_phase, next_msg = _predict_next_node(
                    node_name, {**initial_state, **final_state},
                )
                if next_msg:
                    current_phase = next_phase
                    current_message = next_msg

                # tool_selector 완료 시 tool_call SSE 이벤트 (투명성 — 사용자에게 "무엇을
                # 하려는지" 노출). pending_tool_call 이 None 이면 발행 스킵.
                if node_name == "tool_selector":
                    call = updates.get("pending_tool_call")
                    if isinstance(call, ToolCall):
                        yield _format_sse_event(
                            "tool_call",
                            {
                                "tool_name": call.tool_name,
                                "arguments": call.arguments,
                                "tier": call.tier,
                            },
                        )

                # tool_executor 완료 시 tool_result SSE 이벤트 — 축약된 메타 정보만.
                # raw 데이터는 프런트에 보내지 않는다 (narrator 가 자연어로 서술).
                if node_name == "tool_executor":
                    ref_id = updates.get("latest_tool_ref_id", "")
                    cache = updates.get("tool_results_cache", {}) or {}
                    call = state_snapshot_tool_call(final_state)
                    result = cache.get(ref_id) if ref_id else None
                    if isinstance(result, AdminApiResult):
                        yield _format_sse_event(
                            "tool_result",
                            {
                                "tool_name": call.tool_name if call else "",
                                "ok": result.ok,
                                "status_code": result.status_code,
                                "latency_ms": result.latency_ms,
                                "row_count": result.row_count,
                                "ref_id": ref_id,
                                "error": result.error if not result.ok else "",
                            },
                        )

                # response_formatter 완료 시 최종 응답 텍스트를 token 으로 발행
                if node_name == "response_formatter":
                    response_text = updates.get("response_text", "")
                    if response_text:
                        yield _format_sse_event(
                            "token", {"delta": response_text},
                        )

        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        intent_kind = (
            final_state.get("intent").kind
            if isinstance(final_state.get("intent"), AdminIntent)
            else "unknown"
        )

        # ── HITL interrupt 감지 ──
        # Step 5a: 그래프가 END 로 가지 않고 risk_gate 에서 interrupt 로 멈춘 경우,
        # checkpointer snapshot 의 next 가 비어있지 않다. 이 때는 confirmation_required
        # 이벤트를 발행한 뒤 done 없이 스트림을 종료해 Client 가 /resume 을 호출하도록 한다.
        snapshot = await admin_assistant_graph.aget_state(graph_config)
        is_interrupted = bool(snapshot.next)

        if is_interrupted:
            # tasks 목록 중 interrupt payload 가 담긴 첫 값 꺼내기
            confirmation_value: dict | None = None
            for task in (snapshot.tasks or []):
                interrupts = getattr(task, "interrupts", None) or []
                for intr in interrupts:
                    val = getattr(intr, "value", None)
                    if isinstance(val, dict):
                        confirmation_value = val
                        break
                if confirmation_value is not None:
                    break

            if confirmation_value is not None:
                logger.info(
                    "admin_assistant_interrupt_emitted",
                    session_id=session_id,
                    tool_name=confirmation_value.get("tool_name", ""),
                    tier=confirmation_value.get("tier", -1),
                    elapsed_ms=round(graph_elapsed_ms, 1),
                )
                yield _format_sse_event("confirmation_required", confirmation_value)
                # done 을 발행하지 않는다 — Client 는 '승인 대기' 상태로 대기.
                return

            # payload 를 꺼내지 못한 예외 상황: 에러 이벤트로 대체
            logger.warning(
                "admin_assistant_interrupt_without_payload",
                session_id=session_id,
                snapshot_next=snapshot.next,
            )
            yield _format_sse_event("error", {
                "message": "승인 요청을 준비하지 못했어요. 잠시 후 다시 시도해주세요.",
            })
            yield _format_sse_event("done", {})
            return

        logger.info(
            "admin_assistant_stream_done",
            session_id=session_id,
            intent=intent_kind,
            elapsed_ms=round(graph_elapsed_ms, 1),
        )
        yield _format_sse_event("done", {})

    except Exception as e:
        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        logger.error(
            "admin_assistant_stream_error",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(graph_elapsed_ms, 1),
        )
        yield _format_sse_event("error", {"message": str(e)})
        yield _format_sse_event("done", {})

    finally:
        if not graph_task.done():
            graph_task.cancel()
            try:
                await graph_task
            except (asyncio.CancelledError, Exception):
                pass


def _predict_next_node(completed_node: str, merged_state: dict) -> tuple[str, str]:
    """
    방금 완료된 노드 이후 실행될 다음 노드의 (phase, message) 예측.

    keepalive 메시지를 정확히 갱신하기 위함. Chat Agent graph.py 의
    _predict_next_node 와 동일 컨셉이지만 Admin 그래프는 훨씬 단순해 inline 처리.

    Returns:
        (phase, message) — 예측 불가면 ("", "")
    """
    try:
        if completed_node == "context_loader":
            return ("intent_classifier", _NODE_STATUS_MESSAGES["intent_classifier"])
        if completed_node == "intent_classifier":
            next_node = route_after_intent(merged_state)
            msg = _NODE_STATUS_MESSAGES.get(next_node, "")
            return (next_node, msg)
        if completed_node == "smalltalk_responder":
            return ("response_formatter", _NODE_STATUS_MESSAGES["response_formatter"])
        if completed_node == "tool_selector":
            next_node = route_after_tool_select(merged_state)
            msg = _NODE_STATUS_MESSAGES.get(next_node, "")
            return (next_node, msg)
        if completed_node == "risk_gate":
            next_node = route_after_risk_gate(merged_state)
            msg = _NODE_STATUS_MESSAGES.get(next_node, "")
            return (next_node, msg)
        if completed_node == "tool_executor":
            return ("narrator", _NODE_STATUS_MESSAGES["narrator"])
        if completed_node == "narrator":
            return ("response_formatter", _NODE_STATUS_MESSAGES["response_formatter"])
    except Exception:
        pass
    return ("", "")


# ============================================================
# 동기 실행 (테스트/디버그 용)
# ============================================================

async def run_admin_assistant_sync(
    admin_id: str,
    admin_role: str,
    admin_jwt: str,
    session_id: str,
    user_message: str,
) -> AdminAssistantState:
    """
    동기 모드 실행 — 최종 state 반환 (SSE 이벤트 수집 없이 디버그용).
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    initial_state: AdminAssistantState = {
        "admin_id": admin_id,
        "admin_role": admin_role,
        "admin_jwt": admin_jwt,
        "session_id": session_id,
        "user_message": user_message,
        "history": [],
    }
    # Step 5a: checkpointer 활성화로 thread_id 필수. 세션 ID 를 그대로 재사용.
    return await admin_assistant_graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
    )
