"""
관리자 AI 에이전트 — Tier 0 System Read-only Tool (2개).

설계서: docs/관리자_AI에이전트_v3_재설계.md §4.1

Backend `AdminSystemController` (/api/v1/admin/system prefix):
- system_services_status — GET /api/v1/admin/system/services
- system_config          — GET /api/v1/admin/system/config

참고: /api/v1/admin/system/logs 는 SSE 스트림 형식이라 tool 로 노출하기 부적합 — 제외.

Role matrix (§4.2):
- system 조회 → SUPER_ADMIN, ADMIN, DATA_ADMIN
"""

from __future__ import annotations

from pydantic import BaseModel

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

_SYSTEM_READ_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN", "DATA_ADMIN"}


# ============================================================
# Args Schemas
# ============================================================

class _NoArgs(BaseModel):
    """파라미터 없는 EP 용 빈 스키마 (LLM 이 arguments={} 로 호출)."""

    pass


# ============================================================
# Handlers
# ============================================================

async def _handle_system_services_status(
    ctx: ToolContext,
) -> AdminApiResult:
    """`GET /api/v1/admin/system/services` 호출.

    각 백엔드 서비스(DB, Redis, Elasticsearch, Qdrant, Neo4j, vLLM 등)의
    헬스 상태를 집계해서 돌려주는 EP. 파라미터 없음.
    """
    raw = await get_admin_json(
        "/api/v1/admin/system/services",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_system_config(
    ctx: ToolContext,
) -> AdminApiResult:
    """`GET /api/v1/admin/system/config` 호출.

    현재 적용된 서버 설정값(피처 플래그, 쿼터 기본값, AI 모델 설정 등)을 반환.
    파라미터 없음.
    """
    raw = await get_admin_json(
        "/api/v1/admin/system/config",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="system_services_status",
    tier=0,
    required_roles=_SYSTEM_READ_ROLES,
    description=(
        "운영 중인 백엔드 서비스 헬스 상태 조회. MySQL/Redis/Elasticsearch/Qdrant/Neo4j/vLLM 등 "
        "각 인프라 컴포넌트의 UP/DOWN 상태를 한 번에 확인한다. "
        "'서비스 상태 이상 없어?', '지금 DB 살아 있어?' 질문에 사용한다. 파라미터 없음."
    ),
    example_questions=[
        "지금 모든 서비스 상태 괜찮아?",
        "DB·Redis 헬스 확인해줘",
        "인프라 컴포넌트 상태 요약",
    ],
    args_schema=_NoArgs,
    handler=_handle_system_services_status,
))


register_tool(ToolSpec(
    name="system_config",
    tier=0,
    required_roles=_SYSTEM_READ_ROLES,
    description=(
        "현재 서버 설정값 조회. 피처 플래그·AI 쿼터 기본값·모델 파라미터 등 운영 설정을 확인한다. "
        "'현재 AI 쿼터 기본값', '피처 플래그 어떻게 설정돼 있어?' 질문에 사용한다. 파라미터 없음."
    ),
    example_questions=[
        "현재 AI 쿼터 기본값 얼마야?",
        "피처 플래그 설정 확인",
        "서버 설정값 보여줘",
    ],
    args_schema=_NoArgs,
    handler=_handle_system_config,
))
