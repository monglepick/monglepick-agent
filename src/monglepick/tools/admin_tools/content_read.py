"""
관리자 AI 에이전트 — Tier 1 Content Read-only Tool (4개).

설계서: docs/관리자_AI에이전트_설계서.md §4 Tier 1

Backend `AdminContentController` (`/api/v1/admin/...`):
- reports_list      — GET /admin/reports?status&page&size                    (Page)
- toxicity_list     — GET /admin/toxicity?minScore&page&size                 (Page)
- posts_list        — GET /admin/posts?keyword&category&status&page&size     (Page)
- reviews_list      — GET /admin/reviews?movieId&minRating&categoryCode&page&size  (Page)

Role matrix (§4.2):
  posts/reviews/reports(read) → SUPER_ADMIN, ADMIN, MODERATOR
  (모더레이션 영역 — FINANCE/STATS/DATA 등 다른 영역 관리자는 접근 불가)
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
# Role matrix — 모더레이션 3인방만
# ============================================================

_CONTENT_READ_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN", "MODERATOR"}


# ============================================================
# Args Schemas
# ============================================================

class _ReportsListArgs(BaseModel):
    status: Literal["", "pending", "reviewed", "resolved", "dismissed"] = Field(
        default="",
        description="신고 처리 상태. 빈 문자열이면 전체.",
    )
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


class _ToxicityListArgs(BaseModel):
    minScore: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="최소 독성 점수 (0.0~1.0). None 또는 생략 시 전체. "
                    "0.6 이상을 주면 HIGH/CRITICAL 로그만 반환.",
    )
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


class _PostsListArgs(BaseModel):
    keyword: str = Field(
        default="", description="제목·본문 LIKE 검색어. 빈 문자열이면 전체.",
    )
    category: Literal["", "FREE", "DISCUSSION", "RECOMMENDATION", "NEWS"] = Field(
        default="", description="카테고리 필터. 빈 문자열이면 전체.",
    )
    status: Literal["", "DRAFT", "PUBLISHED"] = Field(
        default="", description="게시 상태. 빈 문자열이면 전체.",
    )
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


class _ReviewsListArgs(BaseModel):
    movieId: str = Field(default="", description="영화 ID 필터. 빈 문자열이면 전체.")
    minRating: float | None = Field(
        default=None, ge=1.0, le=5.0,
        description="최소 평점 (1.0~5.0). None 또는 생략 시 전체.",
    )
    categoryCode: Literal[
        "",
        "THEATER_RECEIPT", "COURSE", "WORLDCUP",
        "WISHLIST", "AI_RECOMMEND", "PLAYLIST",
    ] = Field(
        default="",
        description="작성 카테고리. 'COURSE' 면 도장깨기 인증 리뷰만. 빈 문자열이면 전체.",
    )
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


# ============================================================
# Handlers
# ============================================================

async def _handle_reports_list(
    ctx: ToolContext, status: str = "", page: int = 0, size: int = 20,
) -> AdminApiResult:
    params: dict = {"page": page, "size": size}
    if status:
        params["status"] = status
    raw = await get_admin_json(
        "/api/v1/admin/reports",
        admin_jwt=ctx.admin_jwt, params=params, invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_toxicity_list(
    ctx: ToolContext, minScore: float | None = None, page: int = 0, size: int = 20,
) -> AdminApiResult:
    params: dict = {"page": page, "size": size}
    if minScore is not None:
        params["minScore"] = minScore
    raw = await get_admin_json(
        "/api/v1/admin/toxicity",
        admin_jwt=ctx.admin_jwt, params=params, invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_posts_list(
    ctx: ToolContext,
    keyword: str = "", category: str = "", status: str = "",
    page: int = 0, size: int = 20,
) -> AdminApiResult:
    params: dict = {"page": page, "size": size}
    if keyword:
        params["keyword"] = keyword
    if category:
        params["category"] = category
    if status:
        params["status"] = status
    raw = await get_admin_json(
        "/api/v1/admin/posts",
        admin_jwt=ctx.admin_jwt, params=params, invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_reviews_list(
    ctx: ToolContext,
    movieId: str = "", minRating: float | None = None, categoryCode: str = "",
    page: int = 0, size: int = 20,
) -> AdminApiResult:
    params: dict = {"page": page, "size": size}
    if movieId:
        params["movieId"] = movieId
    if minRating is not None:
        params["minRating"] = minRating
    if categoryCode:
        params["categoryCode"] = categoryCode
    raw = await get_admin_json(
        "/api/v1/admin/reviews",
        admin_jwt=ctx.admin_jwt, params=params, invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="reports_list",
    tier=1,
    required_roles=_CONTENT_READ_ROLES,
    description=(
        "유저 신고(report) 목록 페이징 조회. 처리 상태(pending/reviewed/resolved/dismissed) "
        "필터 가능. '접수된 신고 건수', '대기 중인 신고 목록' 같은 질문에 사용."
    ),
    example_questions=[
        "대기 중인 신고 목록 보여줘",
        "처리 완료된 신고 건수 요약",
        "최근 신고 20건",
    ],
    args_schema=_ReportsListArgs,
    handler=_handle_reports_list,
))


register_tool(ToolSpec(
    name="toxicity_list",
    tier=1,
    required_roles=_CONTENT_READ_ROLES,
    description=(
        "혐오표현(toxicity) 탐지 로그 페이징 조회. minScore 로 HIGH/CRITICAL 만 필터 가능 "
        "(예: minScore=0.6). '독성 점수 높은 댓글', '혐오표현 최근 건수' 에 사용."
    ),
    example_questions=[
        "독성 점수 0.8 이상 로그 보여줘",
        "최근 혐오표현 탐지 건수 알려줘",
    ],
    args_schema=_ToxicityListArgs,
    handler=_handle_toxicity_list,
))


register_tool(ToolSpec(
    name="posts_list",
    tier=1,
    required_roles=_CONTENT_READ_ROLES,
    description=(
        "커뮤니티 게시글 페이징 조회. keyword(제목·본문 LIKE), category(FREE/DISCUSSION/"
        "RECOMMENDATION/NEWS), status(DRAFT/PUBLISHED) 필터 가능. "
        "'최근 논란 게시글', '공지 카테고리만' 등에 사용."
    ),
    example_questions=[
        "'환불' 키워드 게시글 최근 20개",
        "뉴스 카테고리 게시글 목록",
        "DRAFT 상태 게시글 확인",
    ],
    args_schema=_PostsListArgs,
    handler=_handle_posts_list,
))


register_tool(ToolSpec(
    name="reviews_list",
    tier=1,
    required_roles=_CONTENT_READ_ROLES,
    description=(
        "리뷰 목록 페이징 조회. movieId, minRating(1.0~5.0), categoryCode 필터. "
        "categoryCode='COURSE' 는 도장깨기 인증 리뷰만, 'AI_RECOMMEND' 는 AI 추천 리뷰만."
    ),
    example_questions=[
        "도장깨기 인증 리뷰 최근 20개",
        "평점 5점 리뷰 보여줘",
        "영화 ID 12345 리뷰 목록",
    ],
    args_schema=_ReviewsListArgs,
    handler=_handle_reviews_list,
))
