"""
관리자 AI 에이전트 — Tier 3 Points Write Tool (Step 6b).

설계서: docs/관리자_AI에이전트_설계서.md §4 Tier 3

Backend `AdminUserController`:
- points_manual_adjust — POST /api/v1/admin/users/{userId}/points/adjust
    Body: ManualPointAdjustRequest { amount(필수, ±1억 이내, 0 금지), reason(필수, ≤300자) }
    Response: ManualPointResponse { userId, deltaApplied, balanceBefore, balanceAfter, ... }

Role matrix (§4.2):
  payment/포인트 쓰기 → SUPER_ADMIN / ADMIN / FINANCE_ADMIN 만.

HITL 규칙:
- confirm_keyword = "포인트 조정".
- Backend 응답이 이미 balanceBefore/balanceAfter 를 내려주므로, 별도 before 스냅샷은
  user_points 상세를 따로 GET 하지 않고 **Backend 응답 자체**를 after_data 로 보존하고,
  실행 직전 user_detail 의 `pointBalance`(또는 동등 필드) 를 간단 before_data 로 보존한다.
  Backend 가 "트랜잭션 내 실제 before/after" 를 보장하므로 이중 snapshot 불필요하나,
  에이전트 턴 관점의 시각적 before 를 남기는 용도.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from monglepick.api.admin_backend_client import (
    AdminApiResult,
    get_admin_json,
    unwrap_api_response,
    write_admin_json,
)
from monglepick.tools.admin_tools import (
    ToolContext,
    ToolSpec,
    register_tool,
)


# ============================================================
# Role matrix
# ============================================================

_POINTS_WRITE_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN", "FINANCE_ADMIN"}


# ============================================================
# Args Schemas
# ============================================================

class _PointsManualAdjustArgs(BaseModel):
    """POST /api/v1/admin/users/{userId}/points/adjust 용 인자."""

    userId: str = Field(..., min_length=1, description="대상 사용자 ID.")
    amount: int = Field(
        ...,
        ge=-100_000_000,
        le=100_000_000,
        description=(
            "포인트 변동량. 양수=지급(bonus), 음수=회수(revoke). **0은 허용되지 않는다** "
            "(Backend 400 응답). ±1억 범위."
        ),
    )
    reason: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="조정 사유 (필수, ≤300자). PointsHistory.description 에 기록된다.",
    )


# ============================================================
# Helpers — 사용자 상세에서 포인트 잔액만 추출
# ============================================================

async def _snapshot_user_point_summary(
    ctx: ToolContext, user_id: str,
) -> dict | None:
    """
    `/admin/users/{userId}` 응답에서 포인트 잔액/등급 등 핵심 필드만 압축.
    전체 프로필은 너무 크고 개인정보 노출 우려가 있어 최소 필드만 보존한다.
    """
    try:
        raw = await get_admin_json(
            f"/api/v1/admin/users/{user_id}",
            admin_jwt=ctx.admin_jwt,
            invocation_id=ctx.invocation_id,
        )
        unwrapped = unwrap_api_response(raw)
        if not unwrapped.ok or not isinstance(unwrapped.data, dict):
            return None
        profile = unwrapped.data
        # Backend UserDetailResponse 의 정확한 필드명에 의존하지 않도록
        # 여러 후보 키를 시도 (pointBalance / totalPoints / points 등).
        point_balance = (
            profile.get("pointBalance")
            or profile.get("totalPoints")
            or profile.get("points")
        )
        return {
            "userId": user_id,
            "pointBalance": point_balance,
            "gradeCode": profile.get("gradeCode"),
            "nickname": profile.get("nickname"),
        }
    except Exception:  # noqa: BLE001
        return None


# ============================================================
# Handlers
# ============================================================

async def _handle_points_manual_adjust(
    ctx: ToolContext,
    userId: str,
    amount: int,
    reason: str,
) -> AdminApiResult:
    """
    포인트 수동 조정 + before/after 감사 스냅샷.

    Backend 가 ManualPointResponse 에 balanceBefore/balanceAfter 를 이미 내려준다. 이
    값을 after_data 에 포함해 감사 로그의 JSON 본문으로 저장한다. before_data 는 user_detail
    에서 잔액만 따로 추출해 "에이전트가 본 시점" 의 사진을 한 장 남긴다(Backend 트랜잭션
    내부 before 와는 미세한 시간차 있을 수 있음).
    """
    # 1. before 스냅샷 — 간단한 잔액/등급 요약
    before = await _snapshot_user_point_summary(ctx, userId)

    # 2. POST 실행
    body = {"amount": amount, "reason": reason}
    raw = await write_admin_json(
        "POST",
        f"/api/v1/admin/users/{userId}/points/adjust",
        admin_jwt=ctx.admin_jwt,
        json_body=body,
        invocation_id=ctx.invocation_id,
    )
    result = unwrap_api_response(raw)

    # 3. after — Backend 응답 자체가 전/후 잔액을 포함하므로 그걸 그대로 저장
    if result.ok and isinstance(result.data, dict):
        # 예: {"userId":"abc","deltaApplied":500,"balanceBefore":100,"balanceAfter":600,
        #      "pointType":"bonus","reason":"사과","historyId":42}
        after = result.data
    else:
        # 실패 시에도 감사 기록은 남긴다 — after 는 null.
        after = None

    result.before_data = before
    result.after_data = after
    return result


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="points_manual_adjust",
    tier=3,
    required_roles=_POINTS_WRITE_ROLES,
    description=(
        "**위험 작업** — 특정 사용자의 포인트를 수동 조정한다. "
        "amount 양수=지급, 음수=회수(0 금지, ±1억 이내). reason 필수(≤300자). "
        "Backend 가 balanceBefore/balanceAfter 를 응답에 포함하므로 감사 기록에 즉시 반영. "
        "CS 사과 보상, 운영 사고 복구, 프로모션 수동 지급 등에 사용. userId 필수."
    ),
    example_questions=[
        "user_id=abc 에게 500P 사과 지급 — 추천 오류 CS",
        "user_id=xxx 에서 -1000P 회수 — 어뷰징 제재",
    ],
    args_schema=_PointsManualAdjustArgs,
    handler=_handle_points_manual_adjust,
    # Step 6b: "포인트 조정" 을 정확히 타이핑해야 승인 가능.
    confirm_keyword="포인트 조정",
))
