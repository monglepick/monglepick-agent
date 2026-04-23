"""
관리자 AI 에이전트 — Tier 2 Settings Write Tool (1개).

설계서: docs/관리자_AI에이전트_설계서.md §4 Tier 2

Backend `AdminSettingsController`:
- banner_create — POST /api/v1/admin/banners

요청 body (BannerCreateRequest):
  { title: string(≤200), imageUrl: string(≤500), linkUrl?: string(≤500),
    position?: string(≤50), sortOrder?: int,
    startDate?: ISO-8601 string, endDate?: ISO-8601 string }

Role matrix (§4.2):
  settings(admins/terms/banners) 쓰기 → **SUPER_ADMIN 만**.
  (다른 ADMIN 계열은 배너/약관 변경 불가 — 시스템 정책 영역)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from monglepick.api.admin_backend_client import (
    AdminApiResult,
    post_admin_json,
    unwrap_api_response,
)
from monglepick.tools.admin_tools import (
    ToolContext,
    ToolSpec,
    register_tool,
)


# ============================================================
# Role matrix — SUPER_ADMIN 독점 (시스템 정책 영역)
# ============================================================

_SETTINGS_WRITE_ROLES: set[str] = {"SUPER_ADMIN"}


# ============================================================
# Args Schemas
# ============================================================

class _BannerCreateArgs(BaseModel):
    """배너 등록 — Backend BannerCreateRequest 와 필드 매칭."""

    title: str = Field(
        ..., min_length=1, max_length=200, description="배너 제목 (최대 200자).",
    )
    imageUrl: str = Field(
        ..., min_length=1, max_length=500, description="배너 이미지 URL (최대 500자).",
    )
    linkUrl: str = Field(
        default="", max_length=500,
        description="클릭 시 이동할 URL. 빈 문자열이면 null 로 전송.",
    )
    position: str = Field(
        default="", max_length=50,
        description="노출 위치 코드 (예: 'home_top', 'chat_top'). 빈 문자열이면 생략.",
    )
    sortOrder: int | None = Field(
        default=None, description="정렬 순서 (낮을수록 상단).",
    )
    startDate: str = Field(
        default="", description="게시 시작 ISO-8601 (예: 2026-04-23T09:00:00). 빈 문자열이면 즉시.",
    )
    endDate: str = Field(
        default="", description="게시 종료 ISO-8601. 빈 문자열이면 무기한.",
    )


# ============================================================
# Handlers
# ============================================================

async def _handle_banner_create(
    ctx: ToolContext,
    title: str,
    imageUrl: str,
    linkUrl: str = "",
    position: str = "",
    sortOrder: int | None = None,
    startDate: str = "",
    endDate: str = "",
) -> AdminApiResult:
    body: dict = {"title": title, "imageUrl": imageUrl}
    if linkUrl:
        body["linkUrl"] = linkUrl
    if position:
        body["position"] = position
    if sortOrder is not None:
        body["sortOrder"] = sortOrder
    if startDate:
        body["startDate"] = startDate
    if endDate:
        body["endDate"] = endDate

    raw = await post_admin_json(
        "/api/v1/admin/banners",
        admin_jwt=ctx.admin_jwt,
        json_body=body,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="banner_create",
    tier=2,
    required_roles=_SETTINGS_WRITE_ROLES,
    description=(
        "앱 배너 한 건 등록. title(≤200자)/imageUrl(≤500자) 필수, linkUrl/position/"
        "sortOrder/startDate/endDate 선택. 쓰기 작업이므로 Tier 2 승인 필요. "
        "'프로모 배너 등록', '겨울 이벤트 배너 만들어줘' 같은 발화에 사용."
    ),
    example_questions=[
        "겨울 프로모 배너 등록 — 제목 '겨울 이벤트', 이미지 URL xxx",
        "홈 상단에 새 배너 올려줘",
    ],
    args_schema=_BannerCreateArgs,
    handler=_handle_banner_create,
))
