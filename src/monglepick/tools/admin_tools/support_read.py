"""
관리자 AI 에이전트 — Tier 1 Support Read-only Tool (3개).

설계서: docs/관리자_AI에이전트_설계서.md §4 Tier 1

Backend `AdminSupportController`:
- faqs_list          — GET /api/v1/admin/faq?category&page&size               (Page)
- help_articles_list — GET /api/v1/admin/help-articles?category&page&size     (Page)
- tickets_list       — GET /api/v1/admin/tickets?status&page&size             (Page)

Role matrix (§4.2):
  support (faq/help/ticket read) → SUPER_ADMIN, ADMIN, SUPPORT_ADMIN
"""

from __future__ import annotations

from typing import Literal

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

_SUPPORT_READ_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN", "SUPPORT_ADMIN"}


# ============================================================
# Args Schemas
# ============================================================

class _FaqsListArgs(BaseModel):
    category: str = Field(
        default="",
        description="카테고리 필터 (예: '계정', '결제', 'AI'). 빈 문자열이면 전체.",
    )
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


class _HelpArticlesListArgs(BaseModel):
    category: str = Field(
        default="",
        description="도움말 카테고리 필터. 빈 문자열이면 전체.",
    )
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


class _TicketsListArgs(BaseModel):
    status: Literal["", "OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"] = Field(
        default="", description="티켓 상태 필터. 빈 문자열이면 전체.",
    )
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


# ============================================================
# Handlers
# ============================================================

async def _handle_faqs_list(
    ctx: ToolContext, category: str = "", page: int = 0, size: int = 20,
) -> AdminApiResult:
    params: dict = {"page": page, "size": size}
    if category:
        params["category"] = category
    raw = await get_admin_json(
        "/api/v1/admin/faq",
        admin_jwt=ctx.admin_jwt, params=params, invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_help_articles_list(
    ctx: ToolContext, category: str = "", page: int = 0, size: int = 20,
) -> AdminApiResult:
    params: dict = {"page": page, "size": size}
    if category:
        params["category"] = category
    raw = await get_admin_json(
        "/api/v1/admin/help-articles",
        admin_jwt=ctx.admin_jwt, params=params, invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_tickets_list(
    ctx: ToolContext, status: str = "", page: int = 0, size: int = 20,
) -> AdminApiResult:
    params: dict = {"page": page, "size": size}
    if status:
        params["status"] = status
    raw = await get_admin_json(
        "/api/v1/admin/tickets",
        admin_jwt=ctx.admin_jwt, params=params, invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="faqs_list",
    tier=1,
    required_roles=_SUPPORT_READ_ROLES,
    description=(
        "FAQ 목록 페이징 조회. category 필터 가능 (예: '계정', '결제', 'AI'). "
        "'FAQ 몇 개 등록돼 있어?', '결제 카테고리 FAQ 확인' 에 사용."
    ),
    example_questions=[
        "현재 FAQ 전체 몇 개야?",
        "결제 카테고리 FAQ 보여줘",
        "AI 관련 FAQ 목록",
    ],
    args_schema=_FaqsListArgs,
    handler=_handle_faqs_list,
))


register_tool(ToolSpec(
    name="help_articles_list",
    tier=1,
    required_roles=_SUPPORT_READ_ROLES,
    description=(
        "도움말 아티클 목록 페이징 조회. category 필터 가능. "
        "'도움말 몇 개 있어?', '카테고리별 도움말 확인' 에 사용."
    ),
    example_questions=[
        "도움말 전체 목록 보여줘",
        "사용법 카테고리 도움말",
    ],
    args_schema=_HelpArticlesListArgs,
    handler=_handle_help_articles_list,
))


register_tool(ToolSpec(
    name="tickets_list",
    tier=1,
    required_roles=_SUPPORT_READ_ROLES,
    description=(
        "상담 티켓 목록 페이징 조회. status(OPEN/IN_PROGRESS/RESOLVED/CLOSED) 필터 가능. "
        "'미처리 티켓 수', '진행 중인 문의', '오래 대기 중인 티켓' 에 사용."
    ),
    example_questions=[
        "대기 중인(OPEN) 티켓 목록",
        "미처리 티켓 몇 개야?",
        "진행 중인 상담 확인",
    ],
    args_schema=_TicketsListArgs,
    handler=_handle_tickets_list,
))


# ─ v3 확장 ─────────────────────────────────────────────────────────────────
#
# 추가 대상:
#   - notices_list        GET /api/v1/admin/notices?page&size               (확인됨)
#   - notice_detail       GET /api/v1/admin/notices/{id}                    (확인됨)
#   - ticket_detail       GET /api/v1/admin/tickets/{id}                    (확인됨)
#
# 제외 (Backend 단건 GET EP 없음):
#   - faq_detail          GET /api/v1/admin/faq/{id}      — PUT/DELETE 만 존재
#   - help_article_detail GET /api/v1/admin/help-articles/{id} — PUT/DELETE 만 존재
#
# 권한: SUPER_ADMIN, ADMIN, SUPPORT_ADMIN, MODERATOR (설계서 §4.2)
# ────────────────────────────────────────────────────────────────────────────

# notices/ticket 단건 조회 포함 역할 — MODERATOR 추가
_SUPPORT_EXTENDED_ROLES: set[str] = {
    "SUPER_ADMIN", "ADMIN", "SUPPORT_ADMIN", "MODERATOR",
}


# ── Args Schemas (v3 추가) ──────────────────────────────────────────────────

class _NoticesListArgs(BaseModel):
    """GET /api/v1/admin/notices 쿼리 파라미터 스키마."""

    page: int = Field(default=0, ge=0, description="페이지 번호 (0부터 시작).")
    size: int = Field(default=20, ge=1, le=100, description="페이지 크기 (1~100).")


class _NoticeDetailArgs(BaseModel):
    """notices/{id} 단건 조회용 — id 하나만 받음 (Long)."""

    id: int = Field(
        ...,
        gt=0,
        description="조회할 공지사항 ID (Long). 발화에 반드시 명시되어야 한다.",
    )


class _TicketDetailArgs(BaseModel):
    """tickets/{id} 단건 조회용 — id 하나만 받음 (Long)."""

    id: int = Field(
        ...,
        gt=0,
        description="조회할 티켓 ID (Long). 발화에 반드시 명시되어야 한다.",
    )


# ── Handlers (v3 추가) ────────────────────────────────────────────────────

async def _handle_notices_list(
    ctx: ToolContext,
    page: int = 0,
    size: int = 20,
) -> AdminApiResult:
    """GET /api/v1/admin/notices — 공지사항 목록 페이징 조회."""
    raw = await get_admin_json(
        "/api/v1/admin/notices",
        admin_jwt=ctx.admin_jwt,
        params={"page": page, "size": size},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_notice_detail(
    ctx: ToolContext,
    id: int,
) -> AdminApiResult:
    """GET /api/v1/admin/notices/{id} — 공지사항 단건 상세 조회."""
    raw = await get_admin_json(
        f"/api/v1/admin/notices/{id}",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_ticket_detail(
    ctx: ToolContext,
    id: int,
) -> AdminApiResult:
    """GET /api/v1/admin/tickets/{id} — 상담 티켓 단건 상세 조회 (답변 내역 포함)."""
    raw = await get_admin_json(
        f"/api/v1/admin/tickets/{id}",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ── Registration (v3 추가) ─────────────────────────────────────────────────

register_tool(ToolSpec(
    name="notices_list",
    tier=1,
    required_roles=_SUPPORT_EXTENDED_ROLES,
    description=(
        "공지사항 목록을 페이징 조회한다. 인자는 page/size 만 받는다. "
        "'현재 공지사항 몇 개야?', '최근 공지 목록 보여줘', "
        "'등록된 공지 전부 보고 싶어' 같은 질문에 사용한다."
    ),
    example_questions=[
        "현재 등록된 공지사항 목록 보여줘",
        "최근 공지 10개 확인해줘",
        "공지사항 총 몇 개야?",
    ],
    args_schema=_NoticesListArgs,
    handler=_handle_notices_list,
))


register_tool(ToolSpec(
    name="notice_detail",
    tier=1,
    required_roles=_SUPPORT_EXTENDED_ROLES,
    description=(
        "특정 공지사항의 상세 내용을 조회한다. id(Long) 필수. "
        "제목·본문·공개 여부·고정 여부 등 전체 필드를 확인할 때 사용한다. "
        "'공지 id=3 내용 보여줘', '이 공지 핀 고정돼 있어?' 에 사용."
    ),
    example_questions=[
        "공지사항 id=5 상세 내용 보여줘",
        "notice id=2 본문 내용 알려줘",
    ],
    args_schema=_NoticeDetailArgs,
    handler=_handle_notice_detail,
))


register_tool(ToolSpec(
    name="ticket_detail",
    tier=1,
    required_roles=_SUPPORT_EXTENDED_ROLES,
    description=(
        "특정 상담 티켓의 상세 내용을 조회한다. id(Long) 필수. "
        "문의 내용·답변 이력·상태·담당자 등 전체 필드를 확인할 때 사용한다. "
        "'티켓 id=10 내용 뭐야?', '이 문의 답변 달렸어?' 에 사용."
    ),
    example_questions=[
        "티켓 id=10 상세 내용 알려줘",
        "ticket id=3 답변 현황 확인",
    ],
    args_schema=_TicketDetailArgs,
    handler=_handle_ticket_detail,
))
