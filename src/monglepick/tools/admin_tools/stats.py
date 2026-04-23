"""
관리자 AI 에이전트 — Tier 0 Stats Read-only Tool (5개).

설계서: docs/관리자_AI에이전트_설계서.md §4 Tier 0 (통계 조회 GET)

Backend `AdminStatsController` (30+ EP 중 가장 빈도 높은 5개만 Step 2 범위에 등록):
- stats_overview           — GET /api/v1/admin/stats/overview?period={7d|30d|90d}
- stats_trends             — GET /api/v1/admin/stats/trends?period={7d|30d}
- stats_revenue            — GET /api/v1/admin/stats/revenue?period={7d|30d|90d}
- stats_ai_service_overview — GET /api/v1/admin/stats/ai-service/overview
- stats_community_overview  — GET /api/v1/admin/stats/community/overview

Backend 응답은 전부 `ApiResponse<T>` 래퍼 (`{success, data, error}`) 이므로 여기서 data 만
언래핑해서 AdminApiResult.data 에 재주입한다. Client 쪽의 axios interceptor 가 하는 일을
Agent 측에서 동일하게 처리하는 셈.

Role matrix (§4.2):
- stats/* Tier 0 은 **모든 AdminRole 계열** (8종) 에 허용.
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
# 공통 — Backend ApiResponse 래퍼 언래핑
# ============================================================
# Step 4 에서 `admin_backend_client.unwrap_api_response` 공개 유틸로 승격.
# users_read/content_read/payment_read/support_read 도 동일 함수를 공유한다.
# 기존 호출부 호환을 위해 이 모듈에서도 `_unwrap_api_response` 이름으로 재노출.
_unwrap_api_response = unwrap_api_response


# ============================================================
# Args Schemas (LLM bind 용)
# ============================================================

class _PeriodArgs(BaseModel):
    """`period` 쿼리 하나만 받는 통계 EP 공통 args."""

    period: Literal["7d", "30d", "90d"] = Field(
        default="7d",
        description=(
            "조회 기간. '7d' = 최근 7일, '30d' = 최근 30일, '90d' = 최근 90일. "
            "사용자가 기간을 명시하지 않으면 '7d' 를 사용한다."
        ),
    )


class _TrendsArgs(BaseModel):
    """Trends EP 는 7d / 30d 만 지원."""

    period: Literal["7d", "30d"] = Field(
        default="7d",
        description="일별 추이 조회 기간 ('7d' 또는 '30d').",
    )


class _NoArgs(BaseModel):
    """파라미터 없는 EP 용 빈 스키마 (LLM 이 arguments={} 로 호출)."""

    pass


# ============================================================
# Stats Role matrix — 전체 AdminRole 계열 허용 (§4.2)
# ============================================================

_STATS_ALLOWED_ROLES: set[str] = {
    "SUPER_ADMIN", "ADMIN", "MODERATOR",
    "FINANCE_ADMIN", "SUPPORT_ADMIN", "DATA_ADMIN",
    "AI_OPS_ADMIN", "STATS_ADMIN",
}


# ============================================================
# Handlers
# ============================================================

async def _handle_stats_overview(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """`GET /api/v1/admin/stats/overview?period=...` 호출 후 래퍼 언래핑."""
    raw = await get_admin_json(
        "/api/v1/admin/stats/overview",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return _unwrap_api_response(raw)


async def _handle_stats_trends(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """`GET /api/v1/admin/stats/trends?period=...`"""
    raw = await get_admin_json(
        "/api/v1/admin/stats/trends",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return _unwrap_api_response(raw)


async def _handle_stats_revenue(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """`GET /api/v1/admin/stats/revenue?period=...`"""
    raw = await get_admin_json(
        "/api/v1/admin/stats/revenue",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return _unwrap_api_response(raw)


async def _handle_stats_ai_service_overview(
    ctx: ToolContext,
) -> AdminApiResult:
    """`GET /api/v1/admin/stats/ai-service/overview` (파라미터 없음)."""
    raw = await get_admin_json(
        "/api/v1/admin/stats/ai-service/overview",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return _unwrap_api_response(raw)


async def _handle_stats_community_overview(
    ctx: ToolContext,
) -> AdminApiResult:
    """`GET /api/v1/admin/stats/community/overview`."""
    raw = await get_admin_json(
        "/api/v1/admin/stats/community/overview",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return _unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="stats_overview",
    tier=0,
    required_roles=_STATS_ALLOWED_ROLES,
    description=(
        "서비스 전체 KPI 개요 조회. DAU/MAU/신규 가입자 수/리뷰 수/평균 평점/게시글 수를 "
        "최근 7일·30일·90일 중 하나의 기간으로 집계해 돌려준다. "
        "'오늘 DAU', '이번 주 신규 가입', '평균 평점' 같은 요약 질문에 사용한다."
    ),
    example_questions=[
        "지난 7일 DAU 얼마나 돼?",
        "이번 달 신규 가입자 수 보여줘",
        "서비스 현황 요약해줘",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_overview,
))


register_tool(ToolSpec(
    name="stats_trends",
    tier=0,
    required_roles=_STATS_ALLOWED_ROLES,
    description=(
        "일별 추이 차트 데이터. 기간 내 날짜별 DAU/신규 가입/리뷰 수/게시글 수의 "
        "시계열을 돌려준다. 추세·변동·피크 일자 분석에 사용한다."
    ),
    example_questions=[
        "지난 30일 DAU 추이 보여줘",
        "일별 신규 가입 변화 차트로 보고 싶어",
        "최근 일주일 리뷰 수 추세",
    ],
    args_schema=_TrendsArgs,
    handler=_handle_stats_trends,
))


register_tool(ToolSpec(
    name="stats_revenue",
    tier=0,
    required_roles=_STATS_ALLOWED_ROLES,
    description=(
        "결제 매출 통계. 기간별 총 매출/MRR/일별 매출 추이. "
        "매출 관련 질문 (이번 달 매출, 구독 MRR, 매출 추이) 에 사용한다."
    ),
    example_questions=[
        "이번 달 매출 얼마야?",
        "지난 90일 매출 추이",
        "현재 MRR 수준 알려줘",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_revenue,
))


register_tool(ToolSpec(
    name="stats_ai_service_overview",
    tier=0,
    required_roles=_STATS_ALLOWED_ROLES,
    description=(
        "AI 서비스 사용 현황. 전체 AI 세션 수, 평균 턴 수, 사용자당 평균 요청 수 등. "
        "AI 추천 이용 현황/쿼터 소비 관련 질문에 사용한다. 파라미터 없음."
    ),
    example_questions=[
        "AI 추천 얼마나 쓰이고 있어?",
        "챗봇 세션 현황 보여줘",
        "AI 서비스 총 이용 현황",
    ],
    args_schema=_NoArgs,
    handler=_handle_stats_ai_service_overview,
))


register_tool(ToolSpec(
    name="stats_community_overview",
    tier=0,
    required_roles=_STATS_ALLOWED_ROLES,
    description=(
        "커뮤니티 활동 개요. 게시글 수/댓글 수/신고 건수/독성 탐지 건수 등 커뮤니티 KPI. "
        "커뮤니티 건강도 관련 질문에 사용한다. 파라미터 없음."
    ),
    example_questions=[
        "커뮤니티 게시글 얼마나 올라와?",
        "최근 신고 건수 얼마나 돼?",
        "커뮤니티 건강도 요약해줘",
    ],
    args_schema=_NoArgs,
    handler=_handle_stats_community_overview,
))
