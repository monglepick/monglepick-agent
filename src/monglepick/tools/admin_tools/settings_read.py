"""
관리자 AI 에이전트 — Tier 0 Settings Read-only Tool (4개).

설계서: docs/관리자_AI에이전트_v3_재설계.md §4.1

Backend `/api/v1/admin` prefix:
- terms_list      — GET /api/v1/admin/terms?type=(옵션)
- banners_list    — GET /api/v1/admin/banners?position=&active=(옵션)
- audit_logs_list — GET /api/v1/admin/audit-logs?actor&action&from&to&page&size
- admins_list     — GET /api/v1/admin/admins

Role matrix (§4.2):
- terms/banners     → SUPER_ADMIN, ADMIN
- audit_logs/admins → SUPER_ADMIN, ADMIN (민감 정보 — 상위 권한만)
"""

from __future__ import annotations

from typing import Optional

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

# 약관·배너는 SUPER_ADMIN, ADMIN
_SETTINGS_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN"}

# 감사 로그·관리자 목록은 동일 범위 유지 (민감 정보)
_AUDIT_ADMIN_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN"}


# ============================================================
# Args Schemas
# ============================================================

class _TermsListArgs(BaseModel):
    """약관 목록 조회 args."""

    type: Optional[str] = Field(
        default=None,
        description=(
            "약관 종류 필터 (예: 'SERVICE', 'PRIVACY', 'MARKETING'). "
            "생략하면 전체 약관 목록을 반환한다."
        ),
    )


class _BannersListArgs(BaseModel):
    """배너 목록 조회 args."""

    position: Optional[str] = Field(
        default=None,
        description=(
            "배너 노출 위치 필터 (예: 'HOME_TOP', 'CHAT_BOTTOM'). "
            "생략하면 모든 위치의 배너를 반환한다."
        ),
    )
    active: Optional[bool] = Field(
        default=None,
        description=(
            "활성화 여부 필터. True=활성, False=비활성. "
            "생략하면 활성/비활성 전체를 반환한다."
        ),
    )


class _AuditLogsListArgs(BaseModel):
    """감사 로그 목록 조회 args."""

    actor: Optional[str] = Field(
        default=None,
        description="행위자 필터. 관리자 이메일 또는 ID. 생략하면 전체 행위자.",
    )
    action: Optional[str] = Field(
        default=None,
        description="액션 유형 필터 (예: 'USER_SUSPEND', 'POINTS_ADJUST'). 생략하면 전체.",
    )
    from_: Optional[str] = Field(
        default=None,
        alias="from",
        description="조회 시작일 (ISO 8601, 예: '2026-04-01'). 생략하면 제한 없음.",
    )
    to: Optional[str] = Field(
        default=None,
        description="조회 종료일 (ISO 8601, 예: '2026-04-23'). 생략하면 제한 없음.",
    )
    page: int = Field(default=0, ge=0, description="페이지 번호 (0-indexed).")
    size: int = Field(default=20, ge=1, le=100, description="페이지 크기 (최대 100).")

    class Config:
        populate_by_name = True


class _NoArgs(BaseModel):
    """파라미터 없는 EP 용 빈 스키마."""

    pass


# ============================================================
# Handlers
# ============================================================

async def _handle_terms_list(
    ctx: ToolContext,
    type: Optional[str] = None,
) -> AdminApiResult:
    """`GET /api/v1/admin/terms?type=...` 호출.

    type 이 None 이면 쿼리 파라미터를 생략해 전체 약관 목록을 반환받는다.
    """
    params: dict = {}
    if type:
        params["type"] = type
    raw = await get_admin_json(
        "/api/v1/admin/terms",
        admin_jwt=ctx.admin_jwt,
        params=params,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_banners_list(
    ctx: ToolContext,
    position: Optional[str] = None,
    active: Optional[bool] = None,
) -> AdminApiResult:
    """`GET /api/v1/admin/banners?position=...&active=...` 호출.

    position / active 가 None 이면 해당 파라미터를 생략해 전체 배너를 조회한다.
    """
    params: dict = {}
    if position:
        params["position"] = position
    if active is not None:
        # Backend 에서 'true'/'false' 문자열로 수신하므로 소문자 변환
        params["active"] = str(active).lower()
    raw = await get_admin_json(
        "/api/v1/admin/banners",
        admin_jwt=ctx.admin_jwt,
        params=params,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_audit_logs_list(
    ctx: ToolContext,
    actor: Optional[str] = None,
    action: Optional[str] = None,
    from_: Optional[str] = None,
    to: Optional[str] = None,
    page: int = 0,
    size: int = 20,
) -> AdminApiResult:
    """`GET /api/v1/admin/audit-logs?actor=...&action=...&from=...&to=...&page=...&size=...` 호출.

    optional 파라미터는 None 이면 쿼리에 포함하지 않는다.
    `from_` 은 Python 예약어 from 과의 충돌을 피하기 위해 trailing underscore 를 사용하며,
    실제 쿼리 키는 'from' 으로 전송한다.
    """
    params: dict = {"page": page, "size": size}
    if actor:
        params["actor"] = actor
    if action:
        params["action"] = action
    if from_:
        params["from"] = from_
    if to:
        params["to"] = to
    raw = await get_admin_json(
        "/api/v1/admin/audit-logs",
        admin_jwt=ctx.admin_jwt,
        params=params,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_admins_list(
    ctx: ToolContext,
) -> AdminApiResult:
    """`GET /api/v1/admin/admins` 호출 (파라미터 없음).

    등록된 관리자 계정 전체 목록을 반환한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/admins",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="terms_list",
    tier=0,
    required_roles=_SETTINGS_ROLES,
    description=(
        "등록된 약관 목록 조회. type(SERVICE/PRIVACY/MARKETING 등)으로 필터링 가능. "
        "'현재 서비스 약관 내용', '개인정보처리방침 버전 확인' 질문에 사용한다."
    ),
    example_questions=[
        "현재 서비스 이용약관 보여줘",
        "개인정보처리방침 최신 버전 확인",
        "마케팅 동의 약관 있어?",
    ],
    args_schema=_TermsListArgs,
    handler=_handle_terms_list,
))


register_tool(ToolSpec(
    name="banners_list",
    tier=0,
    required_roles=_SETTINGS_ROLES,
    description=(
        "배너 목록 조회. position(노출 위치)·active(활성 여부) 필터 가능. "
        "'지금 활성 배너 몇 개야?', '홈 상단 배너 확인' 질문에 사용한다."
    ),
    example_questions=[
        "현재 활성화된 배너 목록",
        "홈 상단(HOME_TOP) 배너 확인",
        "비활성 배너 현황",
    ],
    args_schema=_BannersListArgs,
    handler=_handle_banners_list,
))


register_tool(ToolSpec(
    name="audit_logs_list",
    tier=0,
    required_roles=_AUDIT_ADMIN_ROLES,
    description=(
        "관리자 행위 감사 로그 목록 조회. actor(행위자)·action(액션 유형)·날짜 범위로 필터링 가능. "
        "'어제 포인트 조정 기록', '특정 관리자 활동 내역' 질문에 사용한다."
    ),
    example_questions=[
        "어제 포인트 조정(POINTS_ADJUST) 감사 로그",
        "관리자 admin@example.com 최근 활동 이력",
        "이번 주 계정 정지 처리 기록",
    ],
    args_schema=_AuditLogsListArgs,
    handler=_handle_audit_logs_list,
))


register_tool(ToolSpec(
    name="admins_list",
    tier=0,
    required_roles=_AUDIT_ADMIN_ROLES,
    description=(
        "등록된 관리자 계정 전체 목록 조회. 이름·이메일·역할(AdminRole)을 확인한다. "
        "'관리자 몇 명이야?', '현재 MODERATOR 역할 누가 있어?' 질문에 사용한다. 파라미터 없음."
    ),
    example_questions=[
        "현재 관리자 계정 전체 목록",
        "MODERATOR 역할 관리자 누구야?",
        "관리자 몇 명 등록돼 있어?",
    ],
    args_schema=_NoArgs,
    handler=_handle_admins_list,
))
