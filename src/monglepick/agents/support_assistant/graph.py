"""
support_assistant LangGraph StateGraph + SSE 스트리머 (v3 — 3노드).

그래프:
    START → context_loader → support_agent → response_formatter → END

SSE 이벤트 (7종 — 기존과 동일하게 Client 영향 없음):
    session     : 세션 ID 발급
    status      : 노드 진행 상태
    matched_faq : support_agent 완료 시 근거 FAQ 요약 (kind ∈ faq/partial 일 때만)
    token       : 최종 본문 (현재 MVP 는 1회 전체 전송)
    needs_human : '상담원 연결' 배너 노출 여부
    done / error: 종료/에러

설계 포인트:
- HITL 없음 (체크포인터 미사용)
- 에러 전파 금지 — 모든 노드가 graceful fallback 으로 응답 문자열을 보장
- v2 의 asyncio.Queue + keepalive 패턴은 유지 (15초 간격 keepalive status)
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
import uuid
from collections.abc import AsyncGenerator

import structlog
from langgraph.graph import END, START, StateGraph

from monglepick.agents.support_assistant.models import (
    MatchedFaq,
    SupportAssistantState,
    ensure_reply,
)
from monglepick.agents.support_assistant.nodes import (
    context_loader,
    response_formatter,
    support_agent,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# 그래프 빌드 (3노드 · 조건부 분기 없음)
# =============================================================================


def build_support_assistant_graph():
    """
    support_assistant StateGraph 구성 + 컴파일.

    v3 는 LLM 이 모든 분기(faq/partial/complaint/out_of_scope/smalltalk) 를
    한 번에 결정하므로 LangGraph 조건부 엣지가 필요 없다. START → 3노드 → END.
    """
    graph = StateGraph(SupportAssistantState)

    graph.add_node("context_loader", context_loader)
    graph.add_node("support_agent", support_agent)
    graph.add_node("response_formatter", response_formatter)

    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "support_agent")
    graph.add_edge("support_agent", "response_formatter")
    graph.add_edge("response_formatter", END)

    compiled = graph.compile()
    logger.info("support_assistant_graph_compiled", node_count=3)
    return compiled


# 모듈 레벨 싱글턴 — 컴파일 1회.
support_assistant_graph = build_support_assistant_graph()


# =============================================================================
# SSE 유틸
# =============================================================================

_KEEPALIVE_INTERVAL_SEC = 15
_SENTINEL = object()

_NODE_STATUS_MESSAGES: dict[str, str] = {
    "context_loader": "FAQ 목록을 확인하고 있어요...",
    "support_agent": "질문 내용을 정리하고 있어요...",
    "response_formatter": "답변을 마무리하고 있어요...",
}


def _format_sse_event(event_type: str, data: dict) -> dict:
    """sse_starlette EventSourceResponse 호환 dict."""
    return {"event": event_type, "data": json.dumps(data, ensure_ascii=False)}


def _serialize_matched_faqs(value) -> list[dict]:
    """matched_faqs 를 SSE JSON 직렬화 가능 형태로 변환."""
    if not value:
        return []
    out: list[dict] = []
    for item in value:
        if isinstance(item, MatchedFaq):
            out.append(
                {
                    "faq_id": item.faq_id,
                    "category": item.category,
                    "question": item.question,
                }
            )
        elif isinstance(item, dict):
            out.append(
                {
                    "faq_id": item.get("faq_id"),
                    "category": item.get("category"),
                    "question": item.get("question"),
                }
            )
    return out


# =============================================================================
# SSE 스트리밍 실행
# =============================================================================


async def run_support_assistant(
    user_id: str,
    session_id: str,
    user_message: str,
) -> AsyncGenerator[dict, None]:
    """
    support_assistant 를 SSE 스트리밍 모드로 실행한다.

    Args:
        user_id: JWT 에서 추출한 사용자 ID (비로그인이면 빈 문자열).
        session_id: 세션 ID (빈 문자열이면 자동 생성).
        user_message: 현재 턴 사용자 발화.

    Yields:
        sse_starlette 호환 dict. {"event": ..., "data": json_string}
    """
    graph_start = time.perf_counter()

    if not session_id:
        session_id = str(uuid.uuid4())

    initial_state: SupportAssistantState = {
        "user_id": user_id or "",
        "session_id": session_id,
        "user_message": user_message,
        "history": [],
    }

    logger.info(
        "support_assistant_stream_start",
        session_id=session_id,
        user_id=user_id or "(guest)",
        message_preview=user_message[:100],
    )

    yield _format_sse_event("session", {"session_id": session_id})

    queue: asyncio.Queue = asyncio.Queue()
    current_phase = "context_loader"
    current_message = _NODE_STATUS_MESSAGES[current_phase]
    final_state: dict = {}

    async def _run_graph_to_queue():
        """LangGraph astream 의 이벤트를 큐에 흘려넣는다."""
        try:
            async for event in support_assistant_graph.astream(
                initial_state,
                stream_mode="updates",
            ):
                await queue.put(event)
            await queue.put(_SENTINEL)
        except Exception as exc:  # noqa: BLE001
            await queue.put(exc)

    graph_task = asyncio.create_task(_run_graph_to_queue())

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=_KEEPALIVE_INTERVAL_SEC
                )
            except asyncio.TimeoutError:
                yield _format_sse_event(
                    "status",
                    {
                        "phase": current_phase,
                        "message": current_message,
                        "keepalive": True,
                    },
                )
                continue

            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item

            # item = {"node_name": {updates}} — LangGraph 특수 이벤트(None/tuple) 방어
            for node_name, updates in item.items():
                if updates is None or not isinstance(updates, dict):
                    logger.debug(
                        "support_stream_skip_special_event",
                        node_name=node_name,
                        updates_type=type(updates).__name__,
                    )
                    continue
                final_state.update(updates)

                # 노드 완료 status 이벤트
                completed_msg = _NODE_STATUS_MESSAGES.get(
                    node_name, f"{node_name} 처리 중..."
                )
                yield _format_sse_event(
                    "status", {"phase": node_name, "message": completed_msg}
                )

                # 다음 phase 예측 — keepalive 메시지 업데이트
                next_phase, next_msg = _predict_next_node(node_name)
                if next_msg:
                    current_phase = next_phase
                    current_message = next_msg

                # support_agent 완료 → matched_faq 이벤트 (근거가 있을 때만)
                if node_name == "support_agent":
                    faqs = _serialize_matched_faqs(updates.get("matched_faqs"))
                    if faqs:
                        yield _format_sse_event("matched_faq", {"items": faqs})

                # response_formatter 완료 → token + needs_human
                if node_name == "response_formatter":
                    response_text = updates.get("response_text", "")
                    needs_human = bool(
                        final_state.get("needs_human_agent", False)
                    )
                    if response_text:
                        yield _format_sse_event(
                            "token", {"delta": response_text}
                        )
                    yield _format_sse_event(
                        "needs_human", {"value": needs_human}
                    )

        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        reply = ensure_reply(final_state.get("reply"))
        kind = reply.kind if reply is not None else "unknown"
        logger.info(
            "support_assistant_stream_done",
            session_id=session_id,
            kind=kind,
            matched_count=len(final_state.get("matched_faqs") or []),
            needs_human=bool(final_state.get("needs_human_agent", False)),
            elapsed_ms=round(graph_elapsed_ms, 1),
        )
        yield _format_sse_event("done", {})

    except Exception as exc:  # noqa: BLE001
        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        logger.error(
            "support_assistant_stream_error",
            error=str(exc),
            error_type=type(exc).__name__,
            elapsed_ms=round(graph_elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        yield _format_sse_event("error", {"message": str(exc)})
        yield _format_sse_event("done", {})

    finally:
        if not graph_task.done():
            graph_task.cancel()
            try:
                await graph_task
            except (asyncio.CancelledError, Exception):
                pass


def _predict_next_node(completed_node: str) -> tuple[str, str]:
    """
    방금 완료된 노드 다음의 (phase, message) 를 돌려 keepalive 메시지 정확도를 유지.

    3노드 구조라 간단히 다음 하나만 매핑한다.
    """
    if completed_node == "context_loader":
        return ("support_agent", _NODE_STATUS_MESSAGES["support_agent"])
    if completed_node == "support_agent":
        return ("response_formatter", _NODE_STATUS_MESSAGES["response_formatter"])
    return ("", "")


# =============================================================================
# 동기 실행 (디버그/테스트용)
# =============================================================================


async def run_support_assistant_sync(
    user_id: str,
    session_id: str,
    user_message: str,
) -> SupportAssistantState:
    """SSE 없이 최종 state 를 반환한다. 테스트 코드에서 사용."""
    if not session_id:
        session_id = str(uuid.uuid4())
    initial_state: SupportAssistantState = {
        "user_id": user_id or "",
        "session_id": session_id,
        "user_message": user_message,
        "history": [],
    }
    return await support_assistant_graph.ainvoke(initial_state)
