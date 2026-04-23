"""
고객센터 AI 챗봇 API 엔드포인트.

엔드포인트:
    POST /api/v1/support/chat        — SSE 스트리밍 (Client SupportChatbotWidget 이 호출)
    POST /api/v1/support/chat/sync   — 동기 JSON (디버그/테스트용)

인증 전략:
    - 로그인 유저: Authorization Bearer JWT → user_id 추출 (세션 추적 용도)
    - 비로그인 게스트: 그대로 허용 (익명 ok) — 고객센터는 게스트 FAQ 문의를 막지 않는다.
    - JWT_SECRET 미설정 개발 환경: body 의 user_id 를 그대로 사용.

설계 의도:
    관리자 어시스턴트(admin_assistant.py) 와 달리 위험 작업이 없으므로
    JWT 강제 X, HITL 없음, 복잡한 권한 검사 없음. 게스트 이용은 허용하되
    세션 저장은 user_id 가 있을 때만 후속 단계에서 고려한다.
"""

from __future__ import annotations

import time

import jwt
import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from monglepick.agents.support_assistant.graph import (
    run_support_assistant,
    run_support_assistant_sync,
)
from monglepick.agents.support_assistant.models import (
    MatchedFaq,
    ensure_reply,
)
from monglepick.config import settings

logger = structlog.get_logger(__name__)


# =============================================================================
# Router
# =============================================================================

support_router = APIRouter(tags=["support"])


# =============================================================================
# 인증 — 게스트 허용. JWT 는 있으면 user_id 추출, 없어도 통과
# =============================================================================


def _resolve_user_id(raw_request: Request) -> str:
    """
    JWT 에서 user_id 를 꺼내되 없으면 빈 문자열 반환.

    chat.py 의 `_extract_user_id_from_jwt` 보다 단순화된 버전:
    - 고객센터는 게스트 허용이 전제라 JWT 실패 시 즉시 빈 문자열 반환.
    - 실패 사유 로깅은 최소 DEBUG 레벨 (운영 로그 노이즈 방지).
    """
    if not settings.JWT_SECRET:
        return ""

    auth_header = raw_request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return ""

    token = auth_header[7:]
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256", "HS384", "HS512"],
        )
    except jwt.ExpiredSignatureError:
        logger.debug("support_chat_jwt_expired")
        return ""
    except jwt.InvalidTokenError as exc:
        logger.debug(
            "support_chat_jwt_invalid",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return ""

    if payload.get("type") == "refresh":
        return ""
    return str(payload.get("sub", "") or "")


# =============================================================================
# Request / Response Pydantic
# =============================================================================


class SupportChatRequest(BaseModel):
    """고객센터 챗봇 요청 payload."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"session_id": "", "message": "전화번호 알려줘"},
                {"session_id": "abc-123", "message": "환불은 어떻게 하나요?"},
            ]
        }
    }

    session_id: str = Field(
        default="",
        description="세션 ID (빈 문자열이면 신규 세션 자동 생성).",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=1500,
        description="사용자 자연어 발화 (1~1500자).",
    )


class SupportMatchedFaqResponse(BaseModel):
    """동기 응답용 MatchedFaq 축약본 (SSE matched_faq 이벤트 포맷과 동일)."""

    faq_id: int
    category: str
    question: str


class SupportChatSyncResponse(BaseModel):
    """동기 디버그 응답. 실제 프런트는 SSE 사용."""

    session_id: str
    response: str
    needs_human_agent: bool = False
    # v3: SupportReply.kind 5종 (faq/partial/complaint/out_of_scope/smalltalk)
    kind: str = ""
    matched_faqs: list[SupportMatchedFaqResponse] = Field(default_factory=list)


# =============================================================================
# SSE 엔드포인트
# =============================================================================


@support_router.post(
    "/support/chat",
    summary="고객센터 AI 챗봇 SSE",
    response_description="SSE 이벤트 스트림 (text/event-stream)",
    responses={
        200: {
            "description": (
                "SSE 스트리밍. 이벤트 타입: session, status, matched_faq, token, "
                "needs_human, done, error."
            ),
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: session\ndata: {"session_id": "abc-..."}\n\n'
                        'event: status\ndata: {"phase": "intent_classifier", "message": "질문 의도를 살펴보는 중이에요..."}\n\n'
                        'event: matched_faq\ndata: {"items": [{"faq_id": 5, "category": "GENERAL", "question": "고객센터 전화번호와 연락처가 어떻게 되나요?", "score": 0.73}]}\n\n'
                        'event: token\ndata: {"delta": "몽글 고객센터는..."}\n\n'
                        'event: needs_human\ndata: {"value": false}\n\n'
                        "event: done\ndata: {}\n\n"
                    )
                }
            },
        }
    },
)
async def support_chat_sse(
    request: SupportChatRequest,
    raw_request: Request,
):
    """
    고객센터 챗봇 SSE 엔드포인트.

    - 로그인 유저/게스트 모두 접근 허용 (JWT 있으면 user_id 추출).
    - LangGraph support_assistant 를 실행하며 각 노드 이벤트를 SSE 로 중계한다.
    """
    user_id = _resolve_user_id(raw_request)
    request_start = time.perf_counter()

    logger.info(
        "support_chat_sse_request",
        user_id=user_id or "(guest)",
        session_id=request.session_id or "(new)",
        message_preview=request.message[:80],
    )

    async def event_generator():
        async for sse_event in run_support_assistant(
            user_id=user_id,
            session_id=request.session_id,
            user_message=request.message,
        ):
            yield sse_event

        elapsed_ms = (time.perf_counter() - request_start) * 1000
        logger.info(
            "support_chat_sse_completed",
            user_id=user_id or "(guest)",
            session_id=request.session_id or "(new)",
            elapsed_ms=round(elapsed_ms, 1),
        )

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


# =============================================================================
# 동기 엔드포인트 (디버그/테스트)
# =============================================================================


@support_router.post(
    "/support/chat/sync",
    summary="고객센터 AI 챗봇 동기 실행 (디버그용)",
    response_model=SupportChatSyncResponse,
)
async def support_chat_sync(
    request: SupportChatRequest,
    raw_request: Request,
):
    """
    그래프를 SSE 없이 1회 실행하고 최종 state 를 JSON 으로 반환한다.

    Client 에서는 SSE 엔드포인트를 쓰고, 이 엔드포인트는 개발자 테스트·E2E 단위 검증
    용도로만 사용한다.
    """
    user_id = _resolve_user_id(raw_request)
    final_state = await run_support_assistant_sync(
        user_id=user_id,
        session_id=request.session_id,
        user_message=request.message,
    )

    reply = ensure_reply(final_state.get("reply"))
    reply_kind = reply.kind if reply is not None else ""

    matched = final_state.get("matched_faqs") or []
    matched_payload: list[SupportMatchedFaqResponse] = []
    for item in matched:
        if isinstance(item, MatchedFaq):
            matched_payload.append(
                SupportMatchedFaqResponse(
                    faq_id=item.faq_id,
                    category=item.category,
                    question=item.question,
                )
            )

    return SupportChatSyncResponse(
        session_id=final_state.get("session_id", request.session_id),
        response=final_state.get("response_text", ""),
        needs_human_agent=bool(final_state.get("needs_human_agent", False)),
        kind=reply_kind,
        matched_faqs=matched_payload,
    )
