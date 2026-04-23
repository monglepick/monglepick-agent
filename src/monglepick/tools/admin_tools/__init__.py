"""
관리자 AI 에이전트 Tool 레지스트리.

설계서: docs/관리자_AI에이전트_설계서.md §4 (Tool Tier), §4.2 (Role × Tool 매트릭스)

구조:
- **ToolContext**: 런타임 컨텍스트. admin_jwt/admin_role/admin_id/session_id/invocation_id 를
  담고, handler 는 이걸로 Backend 에 실제 HTTP 호출한다. LLM 에는 노출되지 않는다.
- **ToolSpec**: 레지스트리 단위. name/tier/required_roles/description/example_questions/
  args_schema/handler. args_schema 는 Pydantic BaseModel — LLM 에 bind 되는 유일한 "보이는"
  스키마다.
- **ADMIN_TOOL_REGISTRY**: name → ToolSpec dict. 각 서브 모듈(stats.py 등)이 register() 로 등록.
- **list_tools_for_role()**: admin_role 기준 matrix 필터. tool_selector 가 이 결과를 Solar
  bind_tools 로 LLM 에 주입한다.

Step 2 현재 등록된 tool:
- stats.py — Tier 0 Read-only GET 5개 (overview/trends/revenue/ai-service/community)

후속 Step 에서 추가될 서브 모듈:
- users_read.py, content_read.py, payment_read.py (Tier 1 리소스 조회)
- content_write.py, support_write.py (Tier 2 경량 쓰기)
- users_write.py, payment_write.py (Tier 3 위험 쓰기)
- aggregate.py (pandas 메타 tool)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from monglepick.api.admin_backend_client import AdminApiResult


# ============================================================
# 공통 타입
# ============================================================

@dataclass
class ToolContext:
    """
    Tool 실행 시점의 런타임 컨텍스트 (LLM 에 비노출).

    admin_jwt 는 Backend 호출 시 `Authorization: Bearer` 헤더로 forwarding 된다 (설계서 §5.1).
    invocation_id 는 Backend 감사 로그와 Agent 턴을 양방향 추적하는 식별자.
    """

    admin_jwt: str
    admin_role: str
    admin_id: str
    session_id: str
    invocation_id: str


# Handler 시그니처: async def(ctx=ToolContext, **validated_args) -> AdminApiResult
ToolHandler = Callable[..., Awaitable[AdminApiResult]]


@dataclass
class ToolSpec:
    """
    Tool 레지스트리 단위.

    - name: LLM 에 노출되는 함수명 (snake_case, 영문)
    - tier: 0~4. 0/1 자동 실행, 2/3 HITL 승인, 4 는 SQL 샌드박스(v2 미지원)
    - required_roles: 이 tool 을 실행할 수 있는 AdminRoleEnum 집합
    - description: LLM 이 tool 선택 시 참고하는 한글 설명 (50~200자)
    - example_questions: "이 tool 이 답할 수 있는 질문" 예시. 후속 Tool RAG 임베딩 소스.
    - args_schema: LLM bind 용 Pydantic 모델. 이름은 `<ToolName>Args` 관례.
    - handler: 실제 실행 함수. ctx + validated args 로 AdminApiResult 반환.
    - confirm_keyword: (Step 6b) Tier 3 위험 쓰기 tool 에 강제할 확인 키워드.
      risk_gate 가 ConfirmationPayload.required_keyword 로 실어 보내고, Admin UI 가
      이 문자열을 사용자에게 정확히 타이핑하라고 요구해 오조작을 2중 방어한다.
      Tier 0/1/2 는 None (메모만 받는 1차 확인으로 충분).
    """

    name: str
    tier: int
    required_roles: set[str]
    description: str
    example_questions: list[str]
    args_schema: type[BaseModel]
    handler: ToolHandler
    confirm_keyword: str | None = None


# ============================================================
# 레지스트리
# ============================================================

ADMIN_TOOL_REGISTRY: dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> None:
    """ToolSpec 을 전역 레지스트리에 등록한다. 중복 이름은 에러."""
    if spec.name in ADMIN_TOOL_REGISTRY:
        raise ValueError(f"Duplicate tool name: {spec.name}")
    ADMIN_TOOL_REGISTRY[spec.name] = spec


def list_tools_for_role(admin_role: str) -> list[ToolSpec]:
    """
    admin_role 에 허용된 tool 목록을 반환한다 (§4.2 Role × Tool 매트릭스).

    - admin_role 이 빈 문자열이면 빈 리스트 (권한 없음).
    - SUPER_ADMIN 은 모든 tool 에 접근 가능 (ToolSpec.required_roles 에 포함 여부와 무관).
    - 그 외는 spec.required_roles 교집합.
    """
    if not admin_role:
        return []
    if admin_role == "SUPER_ADMIN":
        return list(ADMIN_TOOL_REGISTRY.values())
    return [
        s for s in ADMIN_TOOL_REGISTRY.values()
        if admin_role in s.required_roles
    ]


# ============================================================
# 서브 모듈 자동 등록 (import side-effect)
# ============================================================
# 레지스트리에 register_tool() 호출을 트리거하려면 서브 모듈이 한 번 import 되어야 한다.
# __init__.py 하단에서 import 하는 패턴으로 애플리케이션 기동 시 자동 등록한다.
# 순환 import 를 피하기 위해 파일 하단에 위치.
#
# ── Phase D (2026-04-23): v3 재설계 — Read/Draft/Navigate 3종 분류.
#    v2 Write tool 4개(support_write / settings_write / users_write / points_write)
#    는 레지스트리 등록 제외 (파일 보존 — revert 가 필요하면 아래 4줄 재활성화).
#    Draft/Navigate 가 해당 역할을 대체한다.
#
# Step 2: stats(5) — 서비스 KPI 통계
# Step 4: users_read(5) / content_read(4) / payment_read(3) / support_read(3) → Tier 1 15개
# Phase A: stats_extended(10) / ai_ops_read(5) / system_read(3) / settings_read(4) /
#          chat_suggestions_read(1) / dashboard(3) → Read 26개 추가
# Phase B: drafts(10) → Draft tool 10개
# Phase C: navigation(12) → Navigate tool 12개
# Phase D: v2 Write 4개 제거 → 총 56+10+12 = 76개 (가상 finish_task 제외)
from monglepick.tools.admin_tools import stats as _stats  # noqa: E402, F401
from monglepick.tools.admin_tools import users_read as _users_read  # noqa: E402, F401
from monglepick.tools.admin_tools import content_read as _content_read  # noqa: E402, F401
from monglepick.tools.admin_tools import payment_read as _payment_read  # noqa: E402, F401
from monglepick.tools.admin_tools import support_read as _support_read  # noqa: E402, F401
# v2 Write tool 4개 — Phase D 에서 제거 (파일은 revert 용으로 보존)
# from monglepick.tools.admin_tools import support_write as _support_write  # noqa: E402, F401
# from monglepick.tools.admin_tools import settings_write as _settings_write  # noqa: E402, F401
# from monglepick.tools.admin_tools import users_write as _users_write  # noqa: E402, F401
# from monglepick.tools.admin_tools import points_write as _points_write  # noqa: E402, F401
# Phase A Batch 1 (2026-04-23): Stats 확장 Read tool 10개
from monglepick.tools.admin_tools import stats_extended as _stats_extended  # noqa: E402, F401
# Phase A Batch 2 (2026-04-23): AI 운영·시스템·설정·채팅칩 Read-only tool 14개 추가
from monglepick.tools.admin_tools import ai_ops_read as _ai_ops_read  # noqa: E402, F401
from monglepick.tools.admin_tools import system_read as _system_read  # noqa: E402, F401
from monglepick.tools.admin_tools import settings_read as _settings_read  # noqa: E402, F401
from monglepick.tools.admin_tools import chat_suggestions_read as _chat_suggestions_read  # noqa: E402, F401
# Phase A Batch 3 (2026-04-23): Dashboard Read tool 3개
from monglepick.tools.admin_tools import dashboard as _dashboard  # noqa: E402, F401
# Phase B (2026-04-23): Draft tool 10개 — Backend 호출 없이 form_prefill payload 만 반환
from monglepick.tools.admin_tools import drafts as _drafts  # noqa: E402, F401
# Phase C (2026-04-23): Navigate tool 12개 — GET 으로 대상 검색 + 관리 화면 링크 반환
from monglepick.tools.admin_tools import navigation as _navigation  # noqa: E402, F401
