"""
support_assistant LangGraph 노드 (v3 — 3노드).

그래프:
    START → context_loader → support_agent → response_formatter → END

v2 의 7노드(context_loader / intent_classifier / smalltalk / faq_retriever /
answer_generator / fallback / response_formatter) 를 LLM 1회 호출로 축약.
분기 로직은 전부 `support_agent` 노드 안에서 Solar Pro structured output
(SupportReply.kind) 로 결정된다.

모든 노드:
- async def
- 반환값은 state 업데이트용 dict (LangGraph 규약)
- 에러 전파 금지 — 실패 시 graceful fallback 응답으로 떨어진다
"""

from __future__ import annotations

import structlog

from monglepick.agents.support_assistant.faq_client import fetch_faqs
from monglepick.agents.support_assistant.models import (
    FaqDoc,
    MatchedFaq,
    SupportAssistantState,
    SupportReply,
    ensure_reply,
)
from monglepick.chains.support_reply_chain import generate_support_reply

logger = structlog.get_logger(__name__)


# =============================================================================
# 1) context_loader — 매 요청마다 Backend 에서 FAQ 전체를 가져온다
# =============================================================================


async def context_loader(state: SupportAssistantState) -> dict:
    """
    진입 노드. RDB FAQ 전체를 조회해 state.faqs 에 싣고 기본 필드를 초기화한다.

    캐시 없음 — 관리자가 FAQ 를 추가/수정/삭제한 즉시 다음 요청부터 반영된다.
    Backend 장애 시 faqs=[] 상태로 계속 진행 (support_agent 가 complaint 폴백).
    """
    user_message = (state.get("user_message") or "").strip()
    logger.info(
        "support_context_loader_start",
        session_id=state.get("session_id", ""),
        user_id=state.get("user_id", "") or "(guest)",
        message_preview=user_message[:120],
    )

    try:
        faqs = await fetch_faqs()
    except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
        logger.warning(
            "support_context_loader_fetch_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        faqs = []

    return {
        "faqs": faqs,
        # 이전 턴 잔재가 혼입되지 않도록 초기화
        "reply": None,
        "matched_faqs": [],
        "response_text": "",
        "needs_human_agent": False,
        "error": None,
    }


# =============================================================================
# 2) support_agent — Solar Pro 1회 호출로 통합 응답 생성
# =============================================================================


def _select_matched_faqs(
    faqs: list[FaqDoc], matched_ids: list[int]
) -> list[MatchedFaq]:
    """
    LLM 이 돌려준 matched_faq_ids 를 실제 FAQ 메타와 매핑해 SSE/UI 용 축약 리스트로 변환.

    - ID 순서를 LLM 이 제시한 대로 유지 (중요 FAQ 가 먼저 노출되도록)
    - 존재하지 않는 ID 는 조용히 스킵 (환각 방지)
    """
    by_id: dict[int, FaqDoc] = {f.faq_id: f for f in faqs}
    out: list[MatchedFaq] = []
    for fid in matched_ids:
        faq = by_id.get(int(fid))
        if faq is None:
            continue
        out.append(
            MatchedFaq(
                faq_id=faq.faq_id,
                category=faq.category,
                question=faq.question,
            )
        )
    return out


async def support_agent(state: SupportAssistantState) -> dict:
    """
    Solar Pro structured output 으로 SupportReply 를 받아 state 에 반영한다.

    실패/범위 밖/환각은 `generate_support_reply` 내부에서 graceful 처리되어
    항상 SupportReply 인스턴스가 돌아온다. 여기서는 matched_faq_ids 를 실제
    FAQ 와 교차 매핑해 허위 ID 를 걸러낸다.
    """
    user_message = (state.get("user_message") or "").strip()
    faqs: list[FaqDoc] = state.get("faqs") or []

    reply: SupportReply = await generate_support_reply(
        user_message=user_message,
        faqs=faqs,
    )

    matched = _select_matched_faqs(faqs, reply.matched_faq_ids)

    # matched_faq_ids 에는 LLM 이 채웠지만 실제 존재하지 않는 ID 였다면 드롭 후
    # kind 도 보정 (faq/partial 이었는데 매칭이 0건이면 복구 어려움 → complaint 로 격하).
    if reply.kind in ("faq", "partial") and not matched:
        logger.info(
            "support_agent_empty_matches_for_faq_kind",
            original_kind=reply.kind,
            original_ids=reply.matched_faq_ids,
        )
        reply = reply.model_copy(
            update={
                "kind": "complaint",
                "matched_faq_ids": [],
                "needs_human": True,
            }
        )

    logger.info(
        "support_agent_done",
        kind=reply.kind,
        matched_count=len(matched),
        needs_human=reply.needs_human,
    )

    return {
        "reply": reply,
        "matched_faqs": matched,
        "response_text": reply.answer,
        "needs_human_agent": bool(reply.needs_human),
    }


# =============================================================================
# 3) response_formatter — 최종 검증 + 빈 응답 방어
# =============================================================================


async def response_formatter(state: SupportAssistantState) -> dict:
    """
    최종 본문/배너 플래그를 한 번 더 가드한다.

    support_agent 가 이미 기본값을 채우지만, 극단적인 케이스(state 직렬화
    중 reply 손실 등) 에 대비해 방어적으로 보정한다.
    """
    text = (state.get("response_text") or "").strip()
    needs_human = bool(state.get("needs_human_agent", False))

    if not text:
        text = (
            "지금은 답변을 드리기가 어려워요. '문의하기' 탭에서 1:1 티켓으로 "
            "남겨주시면 담당자가 확인해 드릴게요."
        )
        needs_human = True

    # 체크포인트 복원 방어 — reply 가 dict 로 보존된 경우도 정상 복원 가능.
    reply = ensure_reply(state.get("reply"))
    if reply is not None:
        kind = reply.kind
    else:
        kind = "unknown"

    logger.info(
        "support_response_formatter_done",
        kind=kind,
        needs_human=needs_human,
        text_length=len(text),
    )

    return {"response_text": text, "needs_human_agent": needs_human}
