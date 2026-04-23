"""
관리자 AI 에이전트 API 엔드포인트.

설계서: docs/관리자_AI에이전트_설계서.md §8.1 (API), §5 (인증), §8.2 (SSE)

Step 1 범위 (2026-04-23):
- POST /api/v1/admin/assistant/chat       — SSE 스트리밍 (구현)
- POST /api/v1/admin/assistant/chat/sync  — 동기 JSON (디버그용)

후속 Step 에서 추가될 EP:
- POST /api/v1/admin/assistant/resume     — HITL 승인/거절
- GET  /api/v1/admin/assistant/sessions   — 세션 목록
- GET  /api/v1/admin/assistant/sessions/{id}
- DEL  /api/v1/admin/assistant/sessions/{id}

인증 전략 (§5.1):
- Client → fresh JWT (refresh 직후) 를 Bearer 로 보냄
- Agent 는 JWT 검증 → admin_id / admin_role 추출
- admin_role 이 유효하지 않으면 엔드포인트 응답은 정상(200) 이지만 그래프 내부에서
  "관리자 권한이 필요합니다" 안내 텍스트만 내려가도록 graceful 처리 (context_loader)
- Backend 호출 시 JWT forwarding 은 후속 Step 에서 tool 구현 시 도입

보안:
- JWT 미검증 또는 role 이 관리자 계열이 아니면 HTTP 403 Forbidden 으로 차단
  (Chat 과 달리 Admin 은 익명 허용 X)
- JWT_SECRET 미설정 시에는 개발 환경 호환으로 body 의 admin_id 를 그대로 사용
  (운영에서는 반드시 JWT_SECRET 설정 — main.py lifespan 에서 경고)
"""

from __future__ import annotations

import time

import jwt
import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from monglepick.agents.admin_assistant.graph import (
    run_admin_assistant,
    run_admin_assistant_sync,
)
from monglepick.agents.admin_assistant.models import (
    AdminIntent,
    ConfirmationDecision,
    normalize_admin_role,
)
from monglepick.config import settings

logger = structlog.get_logger()


# ============================================================
# Router
# ============================================================

admin_assistant_router = APIRouter(tags=["admin_assistant"])


# ============================================================
# JWT 검증 — admin 권한 요구
# ============================================================

class AdminAuthContext(BaseModel):
    """JWT 에서 추출된 인증 맥락 (memory-scoped, 요청 수명 동안만 존재)."""

    admin_id: str
    admin_role: str            # 정규화된 AdminRoleEnum 값 (빈 문자열이면 권한 없음)
    raw_jwt: str               # Backend forwarding 용 (tool 구현 시 사용)


def _extract_admin_context(raw_request: Request) -> AdminAuthContext:
    """
    Authorization Bearer JWT 에서 admin_id / admin_role 추출.

    Chat Agent `_extract_user_id_from_jwt` 와 패턴 동일하지만 다음 차이:
    - role 클레임 추가 추출 ("role" 또는 "admin_role" 필드 둘 다 허용)
    - 관리자가 아니면 HTTPException 403 으로 즉시 차단 (chat 은 익명 허용)
    - JWT_SECRET 미설정 개발 환경에서는 body 의 admin_id 를 그대로 사용

    Raises:
        HTTPException 401: JWT 없음/만료/서명 오류
        HTTPException 403: JWT 는 유효하나 admin_role 이 관리자 계열이 아님

    Returns:
        AdminAuthContext(admin_id, admin_role, raw_jwt)
    """
    # 개발 환경 호환: JWT_SECRET 미설정이면 검증 생략 → "ADMIN" 역할 가정
    if not settings.JWT_SECRET:
        logger.warning(
            "admin_assistant_jwt_secret_missing_dev_mode",
            message="JWT_SECRET 미설정 — 인증 없이 ADMIN 가정으로 진행 (운영에서는 금지)",
        )
        return AdminAuthContext(admin_id="dev_admin", admin_role="SUPER_ADMIN", raw_jwt="")

    auth_header = raw_request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization 헤더가 필요합니다.")
    token = auth_header[7:]

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256", "HS384", "HS512"],
        )
    except jwt.ExpiredSignatureError:
        logger.warning("admin_assistant_jwt_expired")
        raise HTTPException(status_code=401, detail="JWT 가 만료되었습니다. 다시 로그인해주세요.")
    except jwt.InvalidTokenError as e:
        logger.warning(
            "admin_assistant_jwt_invalid",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(status_code=401, detail="JWT 검증에 실패했습니다.")

    if payload.get("type") == "refresh":
        raise HTTPException(status_code=401, detail="Refresh 토큰은 사용할 수 없습니다.")

    admin_id = payload.get("sub", "") or ""
    raw_role = payload.get("role") or payload.get("admin_role") or ""
    admin_role = normalize_admin_role(str(raw_role) if raw_role else "")

    if not admin_id:
        raise HTTPException(status_code=401, detail="JWT subject 가 비어있습니다.")
    if not admin_role:
        logger.info(
            "admin_assistant_non_admin_blocked",
            admin_id=admin_id,
            raw_role=raw_role,
        )
        raise HTTPException(
            status_code=403,
            detail="관리자 권한이 필요한 기능입니다.",
        )

    return AdminAuthContext(admin_id=admin_id, admin_role=admin_role, raw_jwt=token)


# ============================================================
# Request / Response Pydantic
# ============================================================

class AdminAssistantRequest(BaseModel):
    """관리자 어시스턴트 요청."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"session_id": "", "message": "안녕, 너 뭐 할 수 있어?"},
                {"session_id": "abc-123", "message": "지난 7일 DAU 추이 보여줘"},
            ]
        }
    }

    session_id: str = Field(
        default="",
        description="세션 ID (빈 문자열이면 신규 세션 자동 생성)",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="관리자 자연어 발화 (1~3000자)",
    )


class AdminAssistantResumeRequest(BaseModel):
    """
    HITL 승인/거절 재개 요청 (Step 5a).

    Client(Admin UI) 가 `confirmation_required` SSE 이벤트를 수신해 모달을 띄우고,
    사용자가 승인/거절을 누르면 이 payload 로 `POST /api/v1/admin/assistant/resume` 를
    호출한다. Agent 는 해당 session_id 의 thread_id checkpointer 에서 risk_gate 지점을
    재개해 승인이면 tool_executor 로, 거절이면 response_formatter 로 흐름을 완결한다.

    session_id 는 필수 — 빈 문자열이면 재개할 thread 를 특정할 수 없다.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"session_id": "abc-123", "decision": "approve", "comment": ""},
                {"session_id": "abc-123", "decision": "reject", "comment": "중복 등록"},
            ]
        }
    }

    session_id: str = Field(..., min_length=1, description="재개할 세션 ID (필수).")
    decision: str = Field(
        ..., description="'approve' 또는 'reject'.",
    )
    comment: str = Field(
        default="", max_length=500,
        description="사용자가 모달에 남긴 메모 (선택). 감사 로그에 기록될 수 있다.",
    )


class AdminAssistantSyncResponse(BaseModel):
    """동기 디버그 응답."""

    session_id: str
    response: str
    intent: str = ""
    intent_confidence: float = 0.0
    intent_reason: str = ""


# ============================================================
# SSE 엔드포인트
# ============================================================

@admin_assistant_router.post(
    "/admin/assistant/chat",
    summary="관리자 AI 어시스턴트 SSE 채팅",
    response_description="SSE 이벤트 스트림 (text/event-stream)",
    responses={
        200: {
            "description": (
                "SSE 스트리밍. 이벤트 타입: session, status, token, done, error. "
                "후속 Step 에서 tool_call / tool_result / confirmation_required / chart_data / table_data 추가."
            ),
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: session\ndata: {"session_id": "abc-..."}\n\n'
                        'event: status\ndata: {"phase": "intent_classifier", "message": "요청 의도를 분석하고 있어요..."}\n\n'
                        'event: token\ndata: {"delta": "안녕하세요! 관리자 어시스턴트예요..."}\n\n'
                        "event: done\ndata: {}\n\n"
                    )
                }
            },
        },
        401: {"description": "JWT 미검증 또는 만료"},
        403: {"description": "관리자 권한 없음 (role 이 관리자 계열이 아님)"},
    },
)
async def admin_assistant_chat_sse(
    request: AdminAssistantRequest,
    raw_request: Request,
):
    """
    관리자 AI 어시스턴트 SSE 엔드포인트.

    JWT 검증 후 Admin Assistant LangGraph 를 실행하고 SSE 이벤트를 스트리밍한다.
    세션 저장소(마이그레이션 필요) 는 후속 Step 에서 추가 — Step 1 은 ephemeral.
    """
    auth = _extract_admin_context(raw_request)
    request_start = time.perf_counter()

    logger.info(
        "admin_assistant_sse_request",
        admin_id=auth.admin_id,
        admin_role=auth.admin_role,
        session_id=request.session_id,
        message_preview=request.message[:80],
    )

    async def event_generator():
        """run_admin_assistant() 의 이벤트를 그대로 relay."""
        async for sse_event in run_admin_assistant(
            admin_id=auth.admin_id,
            admin_role=auth.admin_role,
            admin_jwt=auth.raw_jwt,
            session_id=request.session_id,
            user_message=request.message,
        ):
            yield sse_event

        elapsed_ms = (time.perf_counter() - request_start) * 1000
        logger.info(
            "admin_assistant_sse_completed",
            admin_id=auth.admin_id,
            session_id=request.session_id,
            elapsed_ms=round(elapsed_ms, 1),
        )

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


# ============================================================
# Step 5a — HITL 승인 재개 엔드포인트
# ============================================================

@admin_assistant_router.post(
    "/admin/assistant/resume",
    summary="HITL 승인/거절 재개 (SSE)",
    response_description="재개된 그래프의 남은 이벤트 SSE 스트림",
    responses={
        200: {
            "description": "재개 SSE. confirmation_required 는 다시 나오지 않고 "
                           "승인이면 tool_call → tool_result → token → done, "
                           "거절이면 token(거절 안내) → done 흐름.",
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: session\ndata: {"session_id": "abc-123"}\n\n'
                        'event: tool_result\ndata: {"tool_name": "faq_create", "ok": true}\n\n'
                        'event: token\ndata: {"delta": "FAQ 1건을 등록했어요. [출처: faq_create]"}\n\n'
                        "event: done\ndata: {}\n\n"
                    )
                }
            },
        },
        400: {"description": "decision 값이 'approve'/'reject' 외 이거나 session_id 누락"},
        401: {"description": "JWT 미검증/만료"},
        403: {"description": "관리자 권한 없음"},
    },
)
async def admin_assistant_resume(
    request: AdminAssistantResumeRequest,
    raw_request: Request,
):
    """
    HITL 승인/거절 후 LangGraph interrupt 지점을 재개한다.

    v3 주의: risk_gate 노드가 제거되어 이 엔드포인트는 v3 에서 실제로 interrupt 를
    발동하지 않는다. 호출해도 snapshot.next 가 비어있어 그래프가 즉시 done 을 발행한다.
    엔드포인트 자체는 하위 호환 및 향후 HITL 재도입 대비로 보존 (403 으로 차단하지 않음).

    - 관리자 JWT 재검증 (세션 탈취 방어)
    - ConfirmationDecision Pydantic 으로 decision 값 검증(approve/reject)
    - run_admin_assistant 에 resume_payload 를 실어 SSE 스트리밍 실행 (v3 에서 즉시 done)
    """
    logger.warning(
        "admin_assistant_resume_v3_noop",
        message="v3: HITL 재개 비활성화 — risk_gate 제거로 interrupt 발동 안 함. no-op 으로 흘러감.",
        session_id=request.session_id,
    )
    auth = _extract_admin_context(raw_request)

    try:
        decision_obj = ConfirmationDecision(
            decision=request.decision, comment=request.comment,
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="decision 은 'approve' 또는 'reject' 여야 합니다.",
        )

    logger.info(
        "admin_assistant_resume_request",
        admin_id=auth.admin_id,
        session_id=request.session_id,
        decision=decision_obj.decision,
    )

    async def event_generator():
        async for sse_event in run_admin_assistant(
            admin_id=auth.admin_id,
            admin_role=auth.admin_role,
            admin_jwt=auth.raw_jwt,
            session_id=request.session_id,
            user_message="",  # resume 모드는 user_message 미사용
            resume_payload=decision_obj.model_dump(),
        ):
            yield sse_event

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


# ============================================================
# 동기 엔드포인트 (디버그/테스트용)
# ============================================================

@admin_assistant_router.post(
    "/admin/assistant/chat/sync",
    response_model=AdminAssistantSyncResponse,
    summary="관리자 AI 어시스턴트 동기 JSON (디버그용)",
    responses={
        200: {"description": "동기 채팅 응답."},
        401: {"description": "JWT 미검증 또는 만료"},
        403: {"description": "관리자 권한 없음"},
    },
)
async def admin_assistant_chat_sync(
    request: AdminAssistantRequest,
    raw_request: Request,
):
    """
    관리자 어시스턴트 동기 엔드포인트 (Swagger 테스트 / 단위 테스트용).

    SSE 이벤트 수집 없이 최종 state 를 받아 ResponseBody 로 반환한다.
    """
    auth = _extract_admin_context(raw_request)
    request_start = time.perf_counter()

    state = await run_admin_assistant_sync(
        admin_id=auth.admin_id,
        admin_role=auth.admin_role,
        admin_jwt=auth.raw_jwt,
        session_id=request.session_id,
        user_message=request.message,
    )

    intent = state.get("intent")
    intent_kind = intent.kind if isinstance(intent, AdminIntent) else ""
    intent_confidence = intent.confidence if isinstance(intent, AdminIntent) else 0.0
    intent_reason = intent.reason if isinstance(intent, AdminIntent) else ""

    elapsed_ms = (time.perf_counter() - request_start) * 1000
    logger.info(
        "admin_assistant_sync_completed",
        admin_id=auth.admin_id,
        session_id=state.get("session_id", ""),
        intent=intent_kind,
        elapsed_ms=round(elapsed_ms, 1),
    )

    return AdminAssistantSyncResponse(
        session_id=state.get("session_id", request.session_id),
        response=state.get("response_text", ""),
        intent=intent_kind,
        intent_confidence=intent_confidence,
        intent_reason=intent_reason,
    )
