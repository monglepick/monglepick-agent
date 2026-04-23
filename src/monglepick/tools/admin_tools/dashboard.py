"""
관리자 AI 에이전트 — 대시보드 Read Tool (3개).

설계서: docs/관리자_AI에이전트_v3_재설계.md §4.1 Read Tool

Backend `AdminDashboardController` (`/api/v1/admin/dashboard`):
- dashboard_kpi     — GET /api/v1/admin/dashboard/kpi           (파라미터 없음)
- dashboard_trends  — GET /api/v1/admin/dashboard/trends?days=  (1~90일, 기본 7)
- dashboard_recent  — GET /api/v1/admin/dashboard/recent?limit= (1~100건, 기본 20)

모든 tool 은 Tier 0 (자동 실행, 승인 없음).
Role matrix (§5):
- dashboard (* read) → SUPER_ADMIN, ADMIN, DATA_ADMIN, STATS_ADMIN
  (MODERATOR/FINANCE/SUPPORT/AI_OPS 는 대시보드 전체 KPI 보다는 각 도메인 조회 사용)
"""

from __future__ import annotations

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
# Role matrix — 전체 현황 조회 권한 보유 역할
# ============================================================

_DASHBOARD_ROLES: set[str] = {
    "SUPER_ADMIN", "ADMIN", "DATA_ADMIN", "STATS_ADMIN",
}


# ============================================================
# Args Schemas (LLM bind 용)
# ============================================================

class _NoArgs(BaseModel):
    """파라미터 없는 EP 용 빈 스키마."""

    pass


class _DashboardTrendsArgs(BaseModel):
    """`GET /api/v1/admin/dashboard/trends` 용 인자."""

    days: int = Field(
        default=7,
        ge=1,
        le=90,
        description=(
            "조회 일수. 최소 1일, 최대 90일. "
            "사용자가 기간을 명시하지 않으면 기본값 7을 사용한다."
        ),
    )


class _DashboardRecentArgs(BaseModel):
    """`GET /api/v1/admin/dashboard/recent` 용 인자."""

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="반환할 최근 활동 건수. 최소 1, 최대 100. 기본값 20.",
    )


# ============================================================
# Handlers
# ============================================================

async def _handle_dashboard_kpi(ctx: ToolContext) -> AdminApiResult:
    """`GET /api/v1/admin/dashboard/kpi` 호출 후 래퍼 언래핑.

    전체 회원 수, 오늘/어제 신규 가입, 활성 구독,
    오늘/어제 결제 금액, 미처리 신고, 오늘 AI 채팅 요청 수를 반환한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/dashboard/kpi",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_dashboard_trends(ctx: ToolContext, days: int = 7) -> AdminApiResult:
    """`GET /api/v1/admin/dashboard/trends?days=N` 호출 후 래퍼 언래핑.

    최근 N일의 일별 신규 가입 수, 결제 금액, AI 채팅 요청 수 추이를 반환한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/dashboard/trends",
        admin_jwt=ctx.admin_jwt,
        params={"days": days},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_dashboard_recent(ctx: ToolContext, limit: int = 20) -> AdminApiResult:
    """`GET /api/v1/admin/dashboard/recent?limit=N` 호출 후 래퍼 언래핑.

    결제, 신고 등 여러 도메인의 최근 N건 활동 피드를 통합·최신순으로 반환한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/dashboard/recent",
        admin_jwt=ctx.admin_jwt,
        params={"limit": limit},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="dashboard_kpi",
    tier=0,
    required_roles=_DASHBOARD_ROLES,
    description=(
        "관리자 대시보드 KPI 카드 조회. 전체 회원 수, 오늘/어제 신규 가입, 활성 구독 수, "
        "오늘/어제 결제 금액, 미처리 신고 건수, 오늘 AI 채팅 요청 수 핵심 지표를 반환한다. "
        "전일 대비 증감 계산에 필요한 오늘/어제 쌍 데이터 포함. 파라미터 없음."
    ),
    example_questions=[
        "대시보드 KPI 카드 보여줘",
        "오늘 신규 가입자 얼마야?",
        "현재 활성 구독 수 알려줘",
    ],
    args_schema=_NoArgs,
    handler=_handle_dashboard_kpi,
))


register_tool(ToolSpec(
    name="dashboard_trends",
    tier=0,
    required_roles=_DASHBOARD_ROLES,
    description=(
        "관리자 대시보드 추이 차트 조회. 최근 N일(기본 7일, 최대 90일)의 일별 신규 가입 수, "
        "결제 금액, AI 채팅 요청 수 시계열 데이터를 반환한다. "
        "추세 분석·피크 일자 파악·이상 탐지에 사용한다."
    ),
    example_questions=[
        "최근 7일 신규 가입 추이",
        "지난 30일 결제 금액 변화 보여줘",
        "AI 채팅 요청 수 추세 30일",
    ],
    args_schema=_DashboardTrendsArgs,
    handler=_handle_dashboard_trends,
))


register_tool(ToolSpec(
    name="dashboard_recent",
    tier=0,
    required_roles=_DASHBOARD_ROLES,
    description=(
        "관리자 대시보드 최근 활동 피드 조회. 결제·신고 등 여러 도메인의 최근 활동을 "
        "통합하여 최신순으로 반환한다. 기본 20건, 최대 100건. "
        "'최근에 무슨 일 있었어?', '최신 활동 피드' 같은 질문에 사용한다."
    ),
    example_questions=[
        "최근 활동 피드 20건",
        "대시보드 최신 이벤트 보여줘",
        "방금 어떤 일들이 있었어?",
    ],
    args_schema=_DashboardRecentArgs,
    handler=_handle_dashboard_recent,
))
