"""
관리자 AI 에이전트 — Tier 0 Chat Suggestions Read-only Tool (1개).

설계서: docs/관리자_AI에이전트_v3_재설계.md §4.1

Backend `/api/v1/admin/chat-suggestions` prefix:
- chat_suggestions_list — GET /api/v1/admin/chat-suggestions?surface=&active=&page&size

surface 3채널:
- user_chat       — 메인 채팅 인터페이스 추천 칩
- admin_assistant — 관리자 AI 어시스턴트 빈상태 칩
- faq_chatbot     — 고객센터 FAQ 챗봇 칩

Role matrix (§4.2):
- chat_suggestions 조회 → SUPER_ADMIN, ADMIN, AI_OPS_ADMIN
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from monglepick.api.admin_backend_client import (
    AdminApiResult,
    get_admin_json,
    unwrap_api_response,
)
from monglepick.tools.admin_tools import (
    ToolContext,
    ToolSpec,
    register_tool,
)


# ============================================================
# Role matrix
# ============================================================

_CHAT_SUGGESTIONS_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN", "AI_OPS_ADMIN"}


# ============================================================
# Args Schema
# ============================================================

class _ChatSuggestionsListArgs(BaseModel):
    """채팅 추천 칩 목록 조회 args."""

    surface: Optional[Literal["user_chat", "admin_assistant", "faq_chatbot"]] = Field(
        default=None,
        description=(
            "노출 채널 필터. "
            "'user_chat'=메인 채팅 칩, "
            "'admin_assistant'=관리자 AI 어시스턴트 빈상태 칩, "
            "'faq_chatbot'=고객센터 FAQ 챗봇 칩. "
            "생략하면 전체 채널 목록을 반환한다."
        ),
    )
    active: Optional[bool] = Field(
        default=None,
        description=(
            "활성화 여부 필터. True=활성 칩만, False=비활성 칩만. "
            "생략하면 활성/비활성 전체를 반환한다."
        ),
    )
    page: int = Field(default=0, ge=0, description="페이지 번호 (0-indexed).")
    size: int = Field(default=20, ge=1, le=100, description="페이지 크기 (최대 100).")


# ============================================================
# Handler
# ============================================================

async def _handle_chat_suggestions_list(
    ctx: ToolContext,
    surface: Optional[str] = None,
    active: Optional[bool] = None,
    page: int = 0,
    size: int = 20,
) -> AdminApiResult:
    """`GET /api/v1/admin/chat-suggestions?surface=...&active=...&page=...&size=...` 호출.

    surface / active 가 None 이면 해당 파라미터를 쿼리에 포함하지 않아 전체를 조회한다.
    active 는 Backend 가 문자열 'true'/'false' 로 수신하므로 소문자 문자열로 변환해 전달한다.
    """
    params: dict = {"page": page, "size": size}
    if surface:
        params["surface"] = surface
    if active is not None:
        params["active"] = str(active).lower()
    raw = await get_admin_json(
        "/api/v1/admin/chat-suggestions",
        admin_jwt=ctx.admin_jwt,
        params=params,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="chat_suggestions_list",
    tier=0,
    required_roles=_CHAT_SUGGESTIONS_ROLES,
    description=(
        "채팅 추천 칩 목록 페이징 조회. surface(user_chat/admin_assistant/faq_chatbot) 및 "
        "active(활성 여부) 필터 가능. "
        "'현재 활성 추천 칩 몇 개야?', '관리자 어시스턴트 칩 확인' 질문에 사용한다."
    ),
    example_questions=[
        "메인 채팅(user_chat) 추천 칩 목록",
        "현재 활성화된 추천 칩 전체 보여줘",
        "고객센터 챗봇 칩 얼마나 등록돼 있어?",
    ],
    args_schema=_ChatSuggestionsListArgs,
    handler=_handle_chat_suggestions_list,
))
