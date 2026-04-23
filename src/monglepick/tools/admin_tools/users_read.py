"""
관리자 AI 에이전트 — Tier 1 Users Read-only Tool (5개).

설계서: docs/관리자_AI에이전트_설계서.md §4 Tier 1 (리소스 조회 GET)

Backend `AdminUserController` (`/api/v1/admin/users`):
- users_list         — GET /api/v1/admin/users?keyword&status&role&page&size   (Page 응답)
- user_detail        — GET /api/v1/admin/users/{userId}
- user_activity      — GET /api/v1/admin/users/{userId}/activity?page&size
- user_points_history — GET /api/v1/admin/users/{userId}/points?page&size
- user_payments      — GET /api/v1/admin/users/{userId}/payments?page&size

Role matrix (§4.2):
  users (read) → SUPER_ADMIN, ADMIN, MODERATOR, FINANCE_ADMIN, SUPPORT_ADMIN, DATA_ADMIN, AI_OPS_ADMIN
  (STATS_ADMIN 은 통계만 허용 — 개인 정보 보호)

응답은 전부 ApiResponse<T> 래퍼 — unwrap_api_response 로 data 만 추출.
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
# Role matrix — STATS_ADMIN 제외 (개인정보 접근은 통계 전용 관리자에게 허용 안 함)
# ============================================================

_USERS_READ_ROLES: set[str] = {
    "SUPER_ADMIN", "ADMIN", "MODERATOR",
    "FINANCE_ADMIN", "SUPPORT_ADMIN", "DATA_ADMIN", "AI_OPS_ADMIN",
}


# ============================================================
# Args Schemas
# ============================================================

class _UsersListArgs(BaseModel):
    """`GET /api/v1/admin/users` 용 인자."""

    keyword: str = Field(
        default="",
        description=(
            "닉네임 또는 이메일 검색 키워드 (부분 일치). 빈 문자열이면 필터 미적용."
        ),
    )
    status: Literal["", "ACTIVE", "SUSPENDED", "LOCKED"] = Field(
        default="",
        description="계정 상태 필터. 빈 문자열이면 전체.",
    )
    role: Literal["", "USER", "ADMIN"] = Field(
        default="",
        description="역할 필터. 빈 문자열이면 전체.",
    )
    page: int = Field(default=0, ge=0, description="페이지 번호 (0부터 시작).")
    size: int = Field(default=20, ge=1, le=100, description="페이지 크기 (1~100).")


class _UserIdArg(BaseModel):
    """user 단건 조회용 — userId 하나만 받음."""

    userId: str = Field(
        ...,
        min_length=1,
        description="조회할 사용자 ID (VARCHAR(50) PK). 발화에 userId 가 명시되어야 한다.",
    )


class _UserPagedArgs(BaseModel):
    """user 서브리소스 조회용 — userId + page/size."""

    userId: str = Field(..., min_length=1, description="대상 사용자 ID.")
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=100)


# ============================================================
# Handlers
# ============================================================

async def _handle_users_list(
    ctx: ToolContext,
    keyword: str = "",
    status: str = "",
    role: str = "",
    page: int = 0,
    size: int = 20,
) -> AdminApiResult:
    params: dict = {"page": page, "size": size}
    # 빈 문자열 필터는 Backend 가 null 로 취급해 "전체" 가 되므로, 전달하지 않는다.
    if keyword:
        params["keyword"] = keyword
    if status:
        params["status"] = status
    if role:
        params["role"] = role

    raw = await get_admin_json(
        "/api/v1/admin/users",
        admin_jwt=ctx.admin_jwt,
        params=params,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_user_detail(ctx: ToolContext, userId: str) -> AdminApiResult:
    raw = await get_admin_json(
        f"/api/v1/admin/users/{userId}",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_user_activity(
    ctx: ToolContext, userId: str, page: int = 0, size: int = 20,
) -> AdminApiResult:
    raw = await get_admin_json(
        f"/api/v1/admin/users/{userId}/activity",
        admin_jwt=ctx.admin_jwt,
        params={"page": page, "size": size},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_user_points_history(
    ctx: ToolContext, userId: str, page: int = 0, size: int = 20,
) -> AdminApiResult:
    raw = await get_admin_json(
        f"/api/v1/admin/users/{userId}/points",
        admin_jwt=ctx.admin_jwt,
        params={"page": page, "size": size},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_user_payments(
    ctx: ToolContext, userId: str, page: int = 0, size: int = 20,
) -> AdminApiResult:
    raw = await get_admin_json(
        f"/api/v1/admin/users/{userId}/payments",
        admin_jwt=ctx.admin_jwt,
        params={"page": page, "size": size},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="users_list",
    tier=1,
    required_roles=_USERS_READ_ROLES,
    description=(
        "사용자 목록 페이징 조회. 닉네임/이메일 키워드, 계정 상태(ACTIVE/SUSPENDED/LOCKED), "
        "역할(USER/ADMIN) 필터 조합 가능. '최근 가입 N명', '정지된 사용자 목록', "
        "'이메일이 xxx 들어간 사용자' 같은 질문에 사용한다. 탈퇴 회원은 자동 제외."
    ),
    example_questions=[
        "정지된 사용자 목록 보여줘",
        "최근 가입한 관리자 계정 찾아줘",
        "이메일에 gmail 포함된 유저 20명",
    ],
    args_schema=_UsersListArgs,
    handler=_handle_users_list,
))


register_tool(ToolSpec(
    name="user_detail",
    tier=1,
    required_roles=_USERS_READ_ROLES,
    description=(
        "특정 사용자의 상세 프로필 조회. 닉네임/이메일/포인트 잔액·등급/게시글·리뷰·댓글 "
        "작성 건수 등. 발화에 반드시 userId 가 있어야 한다."
    ),
    example_questions=[
        "user_id=abc123 상세 정보 보여줘",
        "xxxxxxxx-xxxx 이 유저 포인트 얼마 남았어?",
    ],
    args_schema=_UserIdArg,
    handler=_handle_user_detail,
))


register_tool(ToolSpec(
    name="user_activity",
    tier=1,
    required_roles=_USERS_READ_ROLES,
    description=(
        "특정 사용자의 활동 이력(게시글/리뷰/댓글 작성 등)을 페이징 조회한다. userId 필수. "
        "시간순 활동 흐름을 파악할 때 사용한다."
    ),
    example_questions=[
        "user_id=abc 최근 활동 내역 보여줘",
        "이 유저 최근에 뭐 했어?",
    ],
    args_schema=_UserPagedArgs,
    handler=_handle_user_activity,
))


register_tool(ToolSpec(
    name="user_points_history",
    tier=1,
    required_roles=_USERS_READ_ROLES,
    description=(
        "특정 사용자의 포인트 변동 이력(적립/사용/환불)을 페이징 조회한다. userId 필수. "
        "'user_id=xxx 의 포인트 이력', '최근 차감 내역' 같은 질문에 사용한다."
    ),
    example_questions=[
        "user_id=abc 포인트 변동 내역",
        "이 유저 최근 1000P 써진 이유 알려줘",
    ],
    args_schema=_UserPagedArgs,
    handler=_handle_user_points_history,
))


register_tool(ToolSpec(
    name="user_payments",
    tier=1,
    required_roles=_USERS_READ_ROLES,
    description=(
        "특정 사용자의 결제 주문 이력을 페이징 조회한다. userId 필수. "
        "'이 유저 결제 내역', '환불된 주문이 뭐가 있어?' 에 사용."
    ),
    example_questions=[
        "user_id=abc 결제 주문 목록",
        "이 유저가 얼마 결제했어?",
    ],
    args_schema=_UserPagedArgs,
    handler=_handle_user_payments,
))
