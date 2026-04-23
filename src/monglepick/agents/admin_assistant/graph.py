"""
관리자 AI 에이전트 LangGraph StateGraph 구성 + SSE 실행 인터페이스.

설계서: docs/관리자_AI에이전트_v3_재설계.md §2 (ReAct 그래프), §3 (SSE 이벤트)

Phase D v3 범위 (2026-04-23):
    START → context_loader → intent_classifier ──┐
                                                  │
    ┌────────── smalltalk ─────────────────┤
    ▼                                       │ stats/query/action
    smalltalk_responder                     ▼
    │                                tool_selector ◀─────────────┐
    │                                       │                     │ continue
    │                         pending=None  │ pending_tool_call   │
    │                                ▼      ▼                     │
    │                  smart_fallback  tool_executor              │
    │                  _responder           ▼                     │
    │                       │           observation ──────────────┘
    │                       │               │
    │                       │               ├─ *_draft  → draft_emitter
    │                       │               ├─ goto_*   → navigator
    │                       │               └─ finish/max_hops → narrator
    │                       │               │
    │                       │               ▼
    │                       └──────→ narrator → response_formatter → END
    │                                                ▲
    └────────────────────────────────────────────────┘

변경 이력 (v2 → v3):
- risk_gate 노드 제거 (실제 쓰기 tool 없음, HITL 불필요)
- observation / draft_emitter / navigator 신규 노드 추가
- ReAct 루프: tool_executor → observation → (tool_selector | draft_emitter | navigator | narrator)
- SSE 이벤트 2종 신규: form_prefill, navigation
- HITL interrupt 감지 블록 보존 (v3 에서 발동 안 함 — risk_gate 제거로 snapshot.next 항상 빔)

SSE 이벤트 (v3 발행 목록):
- session, status, tool_call (매 hop), tool_result (매 hop), token, done, error
- form_prefill (draft_emitter 완료 시)
- navigation (navigator 완료 시)
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
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
    ensure_intent,
    ensure_tool_call,
)
from monglepick.agents.admin_assistant.nodes import (
    MAX_HOPS,
    context_loader,
    draft_emitter,
    intent_classifier,
    narrator,
    navigator,
    observation,
    response_formatter,
    smalltalk_responder,
    smart_fallback_responder,
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

    # MemorySaver 복원 시 dict 로 변환된 경우도 AdminIntent 로 되살린다.
    intent = ensure_intent(state.get("intent"))
    kind = intent.kind if intent is not None else "smalltalk"

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
    tool_selector 이후 분기 (v3 Phase D).

    v3 변경점:
    - risk_gate 제거 — tool_executor 로 직행.
    - finish_task 가 선택된 경우 tool_executor 를 건너뛰고 narrator 로 직행.
      (finish_task 는 가상 tool 로 실제 Backend 호출이 없으므로 executor 불필요)
    - pending_tool_call 이 None 이면 smart_fallback_responder.

    흐름:
      pending=finish_task → narrator
      pending=실제 tool → tool_executor → observation → ...
      pending=None → smart_fallback_responder
    """
    call = ensure_tool_call(state.get("pending_tool_call"))
    if call is None:
        logger.info("route_after_tool_select_no_tool_to_fallback")
        return "smart_fallback_responder"

    if call.tool_name == "finish_task":
        # 가상 tool — executor 건너뛰고 narrator 로 직행
        logger.info(
            "route_after_tool_select_finish_task",
            reason=call.arguments.get("reason", ""),
        )
        return "narrator"

    logger.info(
        "route_after_tool_select",
        tool_name=call.tool_name,
        tier=call.tier,
    )
    return "tool_executor"


def route_after_observation(state: AdminAssistantState) -> str:
    """
    observation 이후 분기 (v3 Phase D 신규).

    ReAct 루프의 핵심 분기점. tool_executor 결과를 observation 이 기록한 뒤,
    다음 hop 을 계속할지 종결 경로로 나갈지 결정한다.

    우선순위:
    1. iteration_count >= MAX_HOPS → narrator (강제 종결, 토큰 비용/무한 루프 방어)
    2. tool_call_history 가 비어있음 → narrator (방어 코드, 정상 흐름에서는 발생 안 함)
    3. 마지막 tool 이 finish_task → narrator (LLM 이 "충분하다" 고 판단)
    4. 마지막 tool 이 *_draft → draft_emitter (form_prefill SSE 발행 후 narrator)
    5. 마지막 tool 이 goto_* → navigator (navigation SSE 발행 후 narrator)
    6. 그 외 read tool → tool_selector (다음 hop 계속)
    """
    hop_count: int = state.get("iteration_count") or 0

    # 1) MAX_HOPS 도달 → 강제 종결
    if hop_count >= MAX_HOPS:
        logger.info(
            "route_after_observation_max_hops",
            hop_count=hop_count,
            max_hops=MAX_HOPS,
        )
        return "narrator"

    # tool_call_history 에서 마지막 항목 꺼내기
    history = state.get("tool_call_history") or []
    if not history:
        logger.debug("route_after_observation_empty_history")
        return "narrator"

    last_raw = history[-1]
    # MemorySaver 직렬화로 dict 화된 경우도 처리
    last_call = ensure_tool_call(last_raw)
    last_name: str = last_call.tool_name if last_call else (
        last_raw.get("tool_name", "") if isinstance(last_raw, dict) else ""
    )

    # 2) finish_task → narrator
    if last_name == "finish_task":
        logger.info("route_after_observation_finish_task")
        return "narrator"

    # 3) *_draft → draft_emitter
    if last_name.endswith("_draft"):
        logger.info("route_after_observation_draft", tool_name=last_name)
        return "draft_emitter"

    # 4) goto_* → navigator
    if last_name.startswith("goto_"):
        logger.info("route_after_observation_navigate", tool_name=last_name)
        return "navigator"

    # 5) 그 외 read tool → tool_selector (다음 hop)
    logger.info(
        "route_after_observation_continue",
        tool_name=last_name,
        hop_count=hop_count,
    )
    return "tool_selector"


# ============================================================
# 그래프 빌드
# ============================================================

def build_admin_assistant_graph():
    """
    Admin Assistant StateGraph 구성 + 컴파일 (v3 Phase D).

    v3 변경점:
    - risk_gate 노드 제거. tool_selector → tool_executor 직행.
    - observation / draft_emitter / navigator 신규 노드 추가.
    - ReAct 루프: tool_executor → observation → route_after_observation →
        (tool_selector | draft_emitter | navigator | narrator)
    - draft_emitter / navigator → narrator → response_formatter → END
    - MemorySaver checkpointer 유지 (v3 에서 interrupt 발동 안 하지만 세션 유지 용도 보존).

    노드 수: 11개 (context_loader, intent_classifier, smalltalk_responder, tool_selector,
             tool_executor, observation, draft_emitter, navigator, narrator,
             smart_fallback_responder, response_formatter)
    """
    graph = StateGraph(AdminAssistantState)

    # ── 기존 노드 ──
    graph.add_node("context_loader", context_loader)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("smalltalk_responder", smalltalk_responder)
    graph.add_node("tool_selector", tool_selector)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("narrator", narrator)
    graph.add_node("smart_fallback_responder", smart_fallback_responder)
    graph.add_node("response_formatter", response_formatter)

    # ── v3 Phase D 신규 노드 ──
    graph.add_node("observation", observation)        # tool_executor 결과 누적
    graph.add_node("draft_emitter", draft_emitter)   # *_draft tool 결과 → form_prefill
    graph.add_node("navigator", navigator)            # goto_* tool 결과 → navigation

    # ── 고정 엣지 ──
    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "intent_classifier")

    # intent_classifier → (smalltalk_responder | tool_selector | response_formatter)
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

    # tool_selector → (tool_executor | narrator | smart_fallback_responder)
    # finish_task 선택 시 narrator 직행, 일반 tool 은 tool_executor, 매칭 실패는 fallback
    graph.add_conditional_edges(
        "tool_selector",
        route_after_tool_select,
        {
            "tool_executor": "tool_executor",
            "narrator": "narrator",
            "smart_fallback_responder": "smart_fallback_responder",
        },
    )
    graph.add_edge("smart_fallback_responder", "response_formatter")

    # tool_executor → observation (항상)
    graph.add_edge("tool_executor", "observation")

    # observation → (tool_selector | draft_emitter | navigator | narrator)
    graph.add_conditional_edges(
        "observation",
        route_after_observation,
        {
            "tool_selector": "tool_selector",
            "draft_emitter": "draft_emitter",
            "navigator": "navigator",
            "narrator": "narrator",
        },
    )

    # draft_emitter / navigator → narrator (form_prefill/navigation 세팅 후 자연어 안내)
    graph.add_edge("draft_emitter", "narrator")
    graph.add_edge("navigator", "narrator")

    # narrator → response_formatter → END
    graph.add_edge("narrator", "response_formatter")
    graph.add_edge("response_formatter", END)

    # MemorySaver checkpointer — v3 에서 interrupt 발동 안 하지만 세션 컨텍스트 보존용 유지.
    # 운영 레벨에서는 RedisSaver 로 교체 권장 (Phase E Step 7).
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "admin_assistant_graph_compiled",
        node_count=11,
        checkpointer="memory",
        version="v3_phase_d",
    )
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
    "tool_selector": "적합한 도구를 고르고 있어요...",
    "tool_executor": "관리자 API를 호출하고 있어요...",
    # v3 Phase D 신규 노드
    "observation": "결과를 검토하고 있어요...",
    "draft_emitter": "폼 내용을 정리하고 있어요...",
    "navigator": "관리 화면 링크를 준비하고 있어요...",
    "narrator": "결과를 정리해 설명하고 있어요...",
    "smart_fallback_responder": "답변 방향을 고민하고 있어요...",
    "response_formatter": "응답을 정리하고 있어요...",
    # v3 에서 제거된 노드 — 하위 호환 메시지 보존 (SSE status 가 이 키를 참조하는 경우 대비)
    "risk_gate": "실행 전 안전 점검 중이에요...",  # v3 미사용
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
    MemorySaver 직렬화로 dict 화된 경우도 ensure_tool_call 로 복원한다.
    """
    return ensure_tool_call(merged_state.get("pending_tool_call"))


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

            # {"node_name": {updates}} — 단, LangGraph 1.0 은 내부 특수 노드
            # (`__start__`, `__interrupt__`, `__end__`) 이벤트에서 value 로 None 또는
            # non-dict(tuple of Interrupt) 을 실어 보내기도 한다. 이런 이벤트는 final_state
            # 에 merge 할 대상이 아니라 스킵 + SSE 이벤트도 발행하지 않는다.
            for node_name, updates in item.items():
                if updates is None or not isinstance(updates, dict):
                    logger.debug(
                        "admin_assistant_stream_skip_special_event",
                        node_name=node_name,
                        updates_type=type(updates).__name__,
                    )
                    continue
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
                # 하려는지" 노출). pending_tool_call 이 None 이거나 finish_task 이면 발행 스킵.
                if node_name == "tool_selector":
                    call = ensure_tool_call(updates.get("pending_tool_call"))
                    if call is not None and call.tool_name != "finish_task":
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

                # v3 Phase D: draft_emitter 완료 시 form_prefill SSE 이벤트 발행.
                # Client 가 이 이벤트를 받으면 FormPrefillCard 를 렌더하고
                # "[action_label]" 버튼으로 navigate(target_path, {state: {draft: ...}}) 제공.
                if node_name == "draft_emitter":
                    prefill = updates.get("form_prefill")
                    if prefill and isinstance(prefill, dict):
                        yield _format_sse_event("form_prefill", prefill)

                # v3 Phase D: navigator 완료 시 navigation SSE 이벤트 발행.
                # Client 가 이 이벤트를 받으면 NavigationCard 를 렌더하고
                # 단건이면 "이동" 버튼, 다건이면 candidates 리스트 + 각각의 "이동" 버튼 제공.
                if node_name == "navigator":
                    nav = updates.get("navigation")
                    if nav and isinstance(nav, dict):
                        yield _format_sse_event("navigation", nav)

                # response_formatter 완료 시 최종 응답 텍스트를 token 으로 발행
                if node_name == "response_formatter":
                    response_text = updates.get("response_text", "")
                    if response_text:
                        yield _format_sse_event(
                            "token", {"delta": response_text},
                        )

        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        _final_intent = ensure_intent(final_state.get("intent"))
        intent_kind = _final_intent.kind if _final_intent is not None else "unknown"

        # ── HITL interrupt 감지 ──
        # v3 에서 발동 안 함 — risk_gate 노드 제거로 interrupt() 호출 지점이 사라졌다.
        # snapshot.next 는 항상 빈 tuple 이므로 is_interrupted=False 로 처리된다.
        # 블록 자체는 하위 호환 및 향후 Phase E+ HITL 재도입 대비로 보존.
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
        # Step 6b 후속 디버깅(2026-04-23): 실전에서 `'NoneType' object is not iterable`
        # 같은 에러가 원인 지점 불명으로 올라왔다. 스택 트레이스를 로그에 남겨 다음
        # 재현에서 정확한 프레임을 특정한다.
        logger.error(
            "admin_assistant_stream_error",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(graph_elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
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
    방금 완료된 노드 이후 실행될 다음 노드의 (phase, message) 예측 (v3 Phase D).

    keepalive status 메시지를 정확히 갱신하기 위함. 라우팅 함수를 직접 호출해 예측.
    예외 발생 시 ("", "") 반환 — keepalive 메시지가 갱신 안 될 뿐 흐름에 영향 없음.

    v3 변경점:
    - risk_gate 예측 제거
    - tool_executor → observation 예측 추가
    - observation → route_after_observation 호출로 예측
    - draft_emitter / navigator → narrator 예측 추가

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
        if completed_node == "tool_executor":
            # v3: tool_executor 는 항상 observation 으로 이동
            return ("observation", _NODE_STATUS_MESSAGES["observation"])
        if completed_node == "observation":
            next_node = route_after_observation(merged_state)
            msg = _NODE_STATUS_MESSAGES.get(next_node, "")
            return (next_node, msg)
        if completed_node == "draft_emitter":
            return ("narrator", _NODE_STATUS_MESSAGES["narrator"])
        if completed_node == "navigator":
            return ("narrator", _NODE_STATUS_MESSAGES["narrator"])
        if completed_node == "narrator":
            return ("response_formatter", _NODE_STATUS_MESSAGES["response_formatter"])
        if completed_node == "smart_fallback_responder":
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
