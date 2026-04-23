"""
관리자 AI 에이전트 — Tier 1 Payment Read-only Tool (3개).

설계서: docs/관리자_AI에이전트_설계서.md §4 Tier 1

Backend `AdminPaymentController`:
- orders_list         — GET /api/v1/admin/payment/orders?status&orderType&userId&page&size   (Page)
- order_detail        — GET /api/v1/admin/payment/orders/{orderId}
- subscriptions_list  — GET /api/v1/admin/subscription?status&planCode&userId&page&size       (Page)

Role matrix (§4.2):
  payment/subscription(read) → SUPER_ADMIN, ADMIN, FINANCE_ADMIN
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
# Role matrix — 재무 영역
# ============================================================

_PAYMENT_READ_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN", "FINANCE_ADMIN"}


# ============================================================
# Args Schemas
# ============================================================

class _OrdersListArgs(BaseModel):
    status: Literal[
        "", "PENDING", "COMPLETED", "FAILED", "REFUNDED", "COMPENSATION_FAILED",
    ] = Field(default="", description="주문 상태 필터. 빈 문자열이면 전체.")
    orderType: Literal["", "SUBSCRIPTION", "POINT_PACK"] = Field(
        default="", description="주문 유형 필터. 빈 문자열이면 전체.",
    )
    userId: str = Field(default="", description="특정 사용자 ID 필터. 빈 문자열이면 전체.")
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


class _OrderDetailArgs(BaseModel):
    orderId: str = Field(
        ..., min_length=1,
        description="조회할 주문 UUID. 발화에 명시되어야 한다.",
    )


class _SubscriptionsListArgs(BaseModel):
    status: Literal["", "ACTIVE", "CANCELLED", "EXPIRED"] = Field(
        default="", description="구독 상태 필터. 빈 문자열이면 전체.",
    )
    planCode: str = Field(
        default="",
        description=(
            "구독 플랜 코드 필터 (예: monthly_basic, monthly_premium, yearly_basic, "
            "yearly_premium). 빈 문자열이면 전체."
        ),
    )
    userId: str = Field(default="", description="특정 사용자 ID 필터. 빈 문자열이면 전체.")
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


# ============================================================
# Handlers
# ============================================================

async def _handle_orders_list(
    ctx: ToolContext,
    status: str = "", orderType: str = "", userId: str = "",
    page: int = 0, size: int = 20,
) -> AdminApiResult:
    params: dict = {"page": page, "size": size}
    if status:
        params["status"] = status
    if orderType:
        params["orderType"] = orderType
    if userId:
        params["userId"] = userId
    raw = await get_admin_json(
        "/api/v1/admin/payment/orders",
        admin_jwt=ctx.admin_jwt, params=params, invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_order_detail(ctx: ToolContext, orderId: str) -> AdminApiResult:
    raw = await get_admin_json(
        f"/api/v1/admin/payment/orders/{orderId}",
        admin_jwt=ctx.admin_jwt, invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_subscriptions_list(
    ctx: ToolContext,
    status: str = "", planCode: str = "", userId: str = "",
    page: int = 0, size: int = 20,
) -> AdminApiResult:
    params: dict = {"page": page, "size": size}
    if status:
        params["status"] = status
    if planCode:
        params["planCode"] = planCode
    if userId:
        params["userId"] = userId
    raw = await get_admin_json(
        "/api/v1/admin/subscription",
        admin_jwt=ctx.admin_jwt, params=params, invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="orders_list",
    tier=1,
    required_roles=_PAYMENT_READ_ROLES,
    description=(
        "결제 주문 목록 페이징 조회. status(PENDING/COMPLETED/FAILED/REFUNDED/COMPENSATION_FAILED), "
        "orderType(SUBSCRIPTION/POINT_PACK), userId 필터 조합. '최근 환불된 주문', "
        "'실패한 결제 목록', '특정 유저 결제 이력' 에 사용."
    ),
    example_questions=[
        "환불된 주문 최근 20건",
        "COMPENSATION_FAILED 상태 주문 찾아줘",
        "user_id=abc 구독 결제 이력",
    ],
    args_schema=_OrdersListArgs,
    handler=_handle_orders_list,
))


register_tool(ToolSpec(
    name="order_detail",
    tier=1,
    required_roles=_PAYMENT_READ_ROLES,
    description=(
        "결제 주문 단건 상세 조회. 환불 정보, PG 거래 ID, 영수증 URL 등 전체 필드. "
        "발화에 orderId(UUID) 가 명시되어야 한다."
    ),
    example_questions=[
        "order_id=xxxx 상세 정보 알려줘",
        "이 주문 환불 내역 확인",
    ],
    args_schema=_OrderDetailArgs,
    handler=_handle_order_detail,
))


register_tool(ToolSpec(
    name="subscriptions_list",
    tier=1,
    required_roles=_PAYMENT_READ_ROLES,
    description=(
        "구독 목록 페이징 조회. status(ACTIVE/CANCELLED/EXPIRED), planCode, userId 필터. "
        "'활성 구독 수', '취소된 프리미엄 구독', '이번 달 만료 예정' 같은 질문에 사용."
    ),
    example_questions=[
        "활성 구독 목록 보여줘",
        "monthly_premium 취소한 유저 목록",
        "user_id=abc 구독 상태 확인",
    ],
    args_schema=_SubscriptionsListArgs,
    handler=_handle_subscriptions_list,
))
