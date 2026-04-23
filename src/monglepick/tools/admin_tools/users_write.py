"""
관리자 AI 에이전트 — Tier 3 Users Write Tool (Step 6b).

설계서: docs/관리자_AI에이전트_설계서.md §4 Tier 3 (위험 쓰기 — 유저/금전/권한/영구손실)

Backend `AdminUserController`:
- user_suspend — PUT /api/v1/admin/users/{userId}/suspend
    Body: SuspendRequest { reason?, durationDays? (null=영구) }

Role matrix (§4.2):
  users 쓰기 → SUPER_ADMIN / ADMIN 만.
  (MODERATOR 는 콘텐츠만, SUPPORT_ADMIN 은 티켓·FAQ 만 — 계정 제재는 최상위 권한 필요)

HITL 규칙:
- ToolSpec.confirm_keyword = "정지" → Admin UI 가 모달에서 정확 타이핑 강제.
- tool_executor 가 실행 전 사용자 상세 GET → before_data, 실행 후 재 GET → after_data.
  `AdminApiResult.before_data`/`after_data` 에 담아 감사 로그에 전달.
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
# Role matrix — Tier 3 쓰기는 최상위 권한만
# ============================================================

_USERS_WRITE_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN"}


# ============================================================
# Args Schemas
# ============================================================

class _UserSuspendArgs(BaseModel):
    """PUT /api/v1/admin/users/{userId}/suspend 용 인자."""

    userId: str = Field(..., min_length=1, description="정지할 사용자 ID (VARCHAR(50) PK).")
    reason: str = Field(
        default="",
        max_length=500,
        description="정지 사유 (선택, 최대 500자). 감사 로그에 그대로 기록된다.",
    )
    durationDays: int | None = Field(
        default=None,
        ge=1,
        description=(
            "임시 정지 일수. null 또는 생략 시 **영구 정지**. 양수면 해당 일수 후 자동 복구 대상."
        ),
    )


# ============================================================
# Helpers — 사용자 상세 스냅샷 (before/after 공통)
# ============================================================

async def _snapshot_user(ctx: ToolContext, user_id: str) -> dict | None:
    """
    단건 사용자 상세를 dict 로 반환 (감사 스냅샷용). 실패 시 None — 감사 기록은
    스냅샷 없이도 진행 가능하도록 graceful.
    """
    try:
        raw = await get_admin_json(
            f"/api/v1/admin/users/{user_id}",
            admin_jwt=ctx.admin_jwt,
            invocation_id=ctx.invocation_id,
        )
        unwrapped = unwrap_api_response(raw)
        if unwrapped.ok and isinstance(unwrapped.data, dict):
            return unwrapped.data
    except Exception:  # noqa: BLE001 — 스냅샷 실패는 graceful
        pass
    return None


# ============================================================
# Handlers
# ============================================================

async def _handle_user_suspend(
    ctx: ToolContext,
    userId: str,
    reason: str = "",
    durationDays: int | None = None,
) -> AdminApiResult:
    """
    사용자 정지 실행 + before/after 스냅샷 자동 주입.

    1. 실행 직전 `GET /admin/users/{userId}` → before_data.
    2. `PUT /admin/users/{userId}/suspend` 실행.
    3. 실행 성공 시 재 GET → after_data.
    4. 최종 AdminApiResult 의 data 는 Backend 응답(정지 성공 메시지 래퍼) 을 언래핑한 값.
    """
    # ── 1. before 스냅샷 ──
    before = await _snapshot_user(ctx, userId)

    # ── 2. PUT 실행 ──
    body: dict = {}
    if reason:
        body["reason"] = reason
    if durationDays is not None:
        body["durationDays"] = durationDays
    raw = await write_admin_json(
        "PUT",
        f"/api/v1/admin/users/{userId}/suspend",
        admin_jwt=ctx.admin_jwt,
        json_body=body,
        invocation_id=ctx.invocation_id,
    )
    result = unwrap_api_response(raw)

    # ── 3. after 스냅샷 (실행 성공일 때만) ──
    after = None
    if result.ok:
        after = await _snapshot_user(ctx, userId)

    # before/after 를 AdminApiResult 에 담아 tool_executor → audit client 로 전달.
    # 실행 실패 시에도 before 는 감사 가치가 있으므로 보존한다.
    result.before_data = before
    result.after_data = after
    return result


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="user_suspend",
    tier=3,
    required_roles=_USERS_WRITE_ROLES,
    description=(
        "**위험 작업** — 특정 사용자 계정을 정지한다. reason(≤500자)/durationDays(null=영구, "
        "양수=임시) 파라미터. PUT /api/v1/admin/users/{userId}/suspend 호출. "
        "실행 전 사용자 상세를 스냅샷으로 감사 로그에 기록한다. 복구는 user_unsuspend "
        "(Step 6c 예정) 로만 가능. 발화에 userId 명시 필수."
    ),
    example_questions=[
        "user_id=abc 7일 정지시켜줘 — 비속어 반복",
        "xxxxxxxx-xxxx 영구 정지 처리",
    ],
    args_schema=_UserSuspendArgs,
    handler=_handle_user_suspend,
    # Step 6b: 사용자가 모달에 "정지" 를 정확 타이핑해야 승인 가능.
    confirm_keyword="정지",
))
