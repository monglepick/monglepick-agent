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
