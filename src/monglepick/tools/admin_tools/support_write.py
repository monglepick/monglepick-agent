"""
관리자 AI 에이전트 — Tier 2 Support Write Tool (1개).

설계서: docs/관리자_AI에이전트_설계서.md §4 Tier 2 (경량 쓰기, HITL 1차 확인)

Backend `AdminSupportController`:
- faq_create — POST /api/v1/admin/faq

요청 body (FaqCreateRequest):
  { category: string, question: string(≤500), answer: string, sortOrder?: int }

Role matrix (§4.2):
  support 쓰기 → SUPER_ADMIN, ADMIN, SUPPORT_ADMIN

Step 5a 주의:
- 이 tool 은 `tier=2` 로 등록되므로 반드시 `risk_gate` 에서 사용자 승인이 난 뒤에만 실행된다.
  tool_executor 자체가 Tier≥2 를 자동 차단하는 가드는 Step 5a 에서 제거됐고, 대신 graph 레벨에서
  risk_gate → (approve 시에만) tool_executor 흐름이 보장한다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from monglepick.api.admin_backend_client import (
    AdminApiResult,
    post_admin_json,
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

_SUPPORT_WRITE_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN", "SUPPORT_ADMIN"}


# ============================================================
# Args Schemas
# ============================================================

class _FaqCreateArgs(BaseModel):
    """FAQ 등록 — Backend FaqCreateRequest 와 필드 매칭."""

    category: str = Field(..., min_length=1, description="FAQ 카테고리 (예: '계정', '결제', 'AI').")
    question: str = Field(
        ..., min_length=1, max_length=500, description="FAQ 질문 (최대 500자).",
    )
    answer: str = Field(..., min_length=1, description="FAQ 답변 본문.")
    sortOrder: int | None = Field(
        default=None, description="정렬 순서 (낮을수록 상단). 미지정 시 Backend 기본값.",
    )


# ============================================================
# Handlers
# ============================================================

async def _handle_faq_create(
    ctx: ToolContext,
    category: str,
    question: str,
    answer: str,
    sortOrder: int | None = None,
) -> AdminApiResult:
    body: dict = {"category": category, "question": question, "answer": answer}
    if sortOrder is not None:
        body["sortOrder"] = sortOrder
    raw = await post_admin_json(
        "/api/v1/admin/faq",
        admin_jwt=ctx.admin_jwt,
        json_body=body,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="faq_create",
    tier=2,
    required_roles=_SUPPORT_WRITE_ROLES,
    description=(
        "FAQ 한 건 등록. category/question(≤500자)/answer/sortOrder 필수·선택 매칭. "
        "쓰기 작업이므로 실행 전 사용자 승인이 반드시 필요하다(Tier 2). "
        "'FAQ 등록해줘', 'FAQ 추가' 같은 발화에 사용."
    ),
    example_questions=[
        "FAQ 등록해줘 — 카테고리 결제, 질문 '환불 받으려면?', 답변 '고객센터 문의'",
        "'계정 비번 변경' FAQ 하나 추가해줘",
    ],
    args_schema=_FaqCreateArgs,
    handler=_handle_faq_create,
))
