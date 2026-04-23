"""
관리자 AI 에이전트 Pydantic/TypedDict 모델 정의.

설계서: docs/관리자_AI에이전트_설계서.md §3.3 (State), §10.2 (Structured Output)

Step 1 범위 (2026-04-23):
- AdminAssistantState (TypedDict, LangGraph State)
- AdminIntent (LLM 구조화 출력용 — 6종 intent)
- AdminRoleEnum (8종, backend AdminRole 과 일치)
- ChartSpec/TableSpec (후속 Step 에서 tool 결과 렌더용 — 스텁만 정의)
- ToolCall/NarrationOutput (후속 Step 에서 사용 예정 — 스텁만 정의)

후속 Step 에서 추가될 모델:
- ToolCard (Tool RAG 카드), AggOp (pandas 집계 연산), PlanStep
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

# ============================================================
# AdminRole — Backend AdminRole enum 과 일치 (8종)
# ============================================================

# Backend domain/admin/entity/AdminRole.java 기준 8종.
# 설계서 §4.2 Role × Tool 매트릭스의 열 이름과 정렬된다.
AdminRoleEnum = Literal[
    "SUPER_ADMIN",
    "ADMIN",
    "MODERATOR",
    "FINANCE_ADMIN",
    "SUPPORT_ADMIN",
    "DATA_ADMIN",
    "AI_OPS_ADMIN",
    "STATS_ADMIN",
]

# Agent 에서 허용하는 확장 집합 — "ADMIN" 이 들어오면 SUPER_ADMIN 에 준하는 권한으로 대우.
# Backend 가 현재 단순 "ADMIN" 문자열을 JWT role 클레임에 넣고 있어 8종 세분화가
# 아직 이뤄지지 않았다(설계서 §13 "Backend @PreAuthorize 정합화는 별도 과제").
# Agent 레벨에서는 "ADMIN"=SUPER_ADMIN 으로 해석해 모든 도구를 허용한다.
ADMIN_ROLE_ALIAS: dict[str, str] = {
    "ADMIN": "SUPER_ADMIN",
}


def normalize_admin_role(raw: str | None) -> str:
    """
    JWT role 클레임 문자열을 AdminRoleEnum 값으로 정규화한다.

    - None 또는 빈 문자열 → "" (비관리자로 간주, 진입 차단)
    - "USER" 등 일반 역할 → "" (비관리자)
    - "ADMIN" → "SUPER_ADMIN" (Backend 가 단순 "ADMIN" 만 발급하는 현 상황 대응)
    - 8종 AdminRoleEnum 값 그대로 매칭되면 그대로 반환
    - 알 수 없는 값 → "" (안전하게 차단)

    Args:
        raw: JWT "role" 클레임 원문

    Returns:
        정규화된 AdminRole 문자열 (빈 문자열이면 관리자 권한 없음)
    """
    if not raw:
        return ""
    value = raw.strip().upper()
    # 별칭 치환 ("ADMIN" → "SUPER_ADMIN")
    value = ADMIN_ROLE_ALIAS.get(value, value)
    allowed: set[str] = {
        "SUPER_ADMIN", "ADMIN", "MODERATOR",
        "FINANCE_ADMIN", "SUPPORT_ADMIN", "DATA_ADMIN",
        "AI_OPS_ADMIN", "STATS_ADMIN",
    }
    return value if value in allowed else ""


# ============================================================
# Intent 분류 (Step 1 범위)
# ============================================================

# 6종 Intent — 관리자가 던질 수 있는 발화 종류.
# 설계서 §10.2 AdminIntent 참조.
#
# - query     : 단건/목록 조회 ("user_id=xxx 의 결제 내역")
# - action    : CRUD 쓰기 ("겨울 프로모 배너 등록해줘")
# - stats     : 통계·집계 요청 ("지난 7일 DAU")
# - report    : 장문 보고서 ("주간 운영 리포트 만들어줘")
# - sql       : 자유 SELECT — v2 에서는 지원 안함. 분류 후 "지원하지 않음" 안내만.
# - smalltalk : 인사·사용법 질문 ("너 뭐 할 수 있어?")
AdminIntentKind = Literal["query", "action", "stats", "report", "sql", "smalltalk"]


class AdminIntent(BaseModel):
    """
    관리자 발화 Intent 분류 결과 (Solar Pro structured output).

    intent_classifier 노드가 이 모델로 구조화 출력을 받아 state 에 저장한다.
    후속 tool_selector 가 intent 에 따라 후보 tool 을 좁힌다.
    """

    kind: AdminIntentKind = Field(
        default="smalltalk",
        description=(
            "발화 종류 6종 중 하나. "
            "query=조회, action=쓰기, stats=통계, report=보고서, sql=자유쿼리, smalltalk=일반대화"
        ),
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="분류 신뢰도 (0.0~1.0). 0.5 미만이면 smalltalk 로 보정한다.",
    )
    reason: str = Field(
        default="",
        description="왜 이 intent 로 분류했는지 한 줄 근거 (로깅/디버깅용).",
    )


# ============================================================
# Tool 실행 관련 모델 (후속 Step 에서 사용 — 현재는 스텁)
# ============================================================

class ToolCall(BaseModel):
    """
    LLM 이 선택한 단일 tool 호출 의도 (후속 Step 에서 구현).

    Step 1 에서는 선언만 두고 사용하지 않는다.
    """

    tool_name: str = Field(..., description="호출할 tool 의 레지스트리 이름")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="tool 에 전달할 인자 (Pydantic 스키마 검증 후 execute)",
    )
    tier: int = Field(default=0, ge=0, le=4, description="Tool Tier (0~4). 2 이상은 HITL 승인 필요.")
    rationale: str = Field(default="", description="이 tool 을 선택한 이유 (LLM 생성).")


# ============================================================
# HITL 승인 (Step 5a) — risk_gate interrupt payload / resume decision
# ============================================================

class ConfirmationPayload(BaseModel):
    """
    risk_gate 가 interrupt() 로 발동할 때 SSE 에 실려 나가는 페이로드.

    Client(Admin UI) 는 이 payload 로 승인 모달을 그리고, 사용자가 승인/거절하면
    `POST /api/v1/admin/assistant/resume` 로 `ConfirmationDecision` 을 돌려보낸다.
    """

    tool_name: str = Field(..., description="승인 대상 tool 이름 (레지스트리 키).")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="실행될 인자. 사용자가 모달에서 검토할 수 있게 전체 전송.",
    )
    tier: int = Field(..., ge=2, le=4, description="Tier 값 (2 이상만 risk_gate 통과).")
    plan_summary: str = Field(
        default="",
        description="한국어 한줄 요약. '공지사항 FAQ 1건 등록' 처럼 모달 상단 타이틀.",
    )
    rationale: str = Field(
        default="",
        description="선택 근거 (LLM 이 제공한 것). 사용자 의사결정 보조 정보.",
    )
    # Step 6b (2026-04-23): Tier 3 tool 에 확인 키워드 타이핑 강제.
    # Tool 레지스트리의 ToolSpec.confirm_keyword 를 risk_gate 가 그대로 주입한다.
    # 빈 문자열이면 키워드 검증을 요구하지 않는다(Tier 2 까지). Admin UI ConfirmationDialog
    # 는 이 값이 비어있지 않으면 "아래 키워드를 직접 입력하세요: 정지" 같은 input 을 노출하고
    # 정확 일치할 때만 [승인하고 실행] 버튼을 활성화한다.
    required_keyword: str = Field(
        default="",
        description=(
            "Tier 3 위험 쓰기 작업의 확인 키워드. 비어있지 않으면 사용자가 모달에서 "
            "이 문자열을 정확히 타이핑해야 승인 가능. 예: '정지', '환불', '포인트 조정'."
        ),
    )


class ConfirmationDecision(BaseModel):
    """
    `/resume` 요청 본문. risk_gate 가 interrupt 로 기다리던 지점으로 이 값이 전달된다.

    - decision='approve': 실행 계속 → tool_executor
    - decision='reject' : 실행 중단 → response_formatter 가 거절 안내 응답
    """

    decision: Literal["approve", "reject"]
    comment: str = Field(
        default="",
        description="사용자가 남긴 메모 (선택). Audit log 에 기록될 수 있다.",
    )


class ToolResult(BaseModel):
    """Tool 실행 결과의 축약 표현 (후속 Step)."""

    tool_name: str
    ok: bool = True
    latency_ms: int = 0
    row_count: int | None = None
    ref_id: str | None = Field(
        default=None,
        description="state.tool_results_cache 에 저장된 원본 결과의 참조 키. "
                    "LLM 컨텍스트에는 ref_id 만 전달하고 raw rows 는 재주입하지 않는다.",
    )
    error: str | None = None


# ============================================================
# 응답 출력 모델 (Narration / Chart / Table)
# ============================================================

class ChartSpec(BaseModel):
    """
    차트 페이로드 스키마 (라이브러리 중립).

    Admin UI ChartRenderer 가 Recharts/chart.js 로 렌더한다.
    Step 1 에서는 사용하지 않음 — 선언만 유지.
    """

    type: Literal["line", "bar", "pie", "area", "heatmap"] = "line"
    title: str = ""
    data: list[dict[str, Any]] = Field(default_factory=list)
    x_key: str = ""
    y_keys: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class TableSpec(BaseModel):
    """표 페이로드 스키마 (Step 1 스텁)."""

    title: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    caption: str = Field(default="", description="출처 tool 및 조회 범위 서술 (근거 표시용).")


class NarrationOutput(BaseModel):
    """
    최종 응답 본문 구조화 출력 (response_formatter 에서 사용 — 후속 Step).

    수치는 절대 LLM 이 생성하지 않는다. summary / key_findings 는 집계된
    dict 값을 "인용·서술" 하는 역할만 맡는다 (설계서 §6.1).
    """

    summary: str = Field(default="", description="3~5문장 한국어 요약.")
    key_findings: list[str] = Field(default_factory=list, description="주요 발견 3~5개 (bullet).")
    referenced_refs: list[str] = Field(
        default_factory=list,
        description="근거 ref_id 목록 — 어떤 tool_result 에 기반했는지.",
    )


# ============================================================
# AdminAssistantState (LangGraph TypedDict State)
# ============================================================

class MessageTurn(TypedDict, total=False):
    """최근 대화 히스토리 한 턴."""

    role: Literal["user", "assistant"]
    content: str


class AdminAssistantState(TypedDict, total=False):
    """
    Admin Assistant LangGraph State.

    §3.3 설계서 기준 State. TypedDict(total=False) — 초기 state 에는 일부만 존재.
    Chat Agent 와 다르게 이 state 는 요청-스코프 ephemeral (세션 persistence 는 별도
    Step 에서 `admin_assistant_session_archive` 테이블 도입).

    Step 1 에서는 아래 필드 중 일부만 실제 사용하고 나머지는 후속 Step 에서 채운다.
    """

    # ── 입력 ──
    admin_id: str                      # JWT sub (user_id). 빈 문자열이면 비관리자 (진입 차단)
    admin_role: str                    # AdminRoleEnum 정규화 값 (빈 문자열이면 권한 없음)
    admin_jwt: str                     # Backend forwarding 용 — 요청 수명 동안만 메모리 보유
    session_id: str                    # 세션 식별자 (자동 생성)
    user_message: str                  # 현재 턴 사용자 발화
    history: list[MessageTurn]         # 최근 8턴 슬라이딩 윈도우 (후속 Step)

    # ── Intent 분류 결과 ──
    intent: AdminIntent | None

    # ── Tool 관련 ──
    candidate_tools: list[dict[str, Any]]      # Tool RAG top-5 (ToolCard) — Step 3 도입 예정
    # Step 2 신규: tool_selector 가 LLM 응답에서 추출한 단일 ToolCall.
    # Step 2 는 single-tool 호출만 지원하므로 ToolCall 객체 하나만 유지.
    # 후속 Step 에서 list[ToolCall] 로 확장해 병렬 실행 도입 예정.
    pending_tool_call: ToolCall | None
    # ref_id → (AdminApiResult 또는 dict). narrator 는 summarize_for_llm 축약본만 보고,
    # raw rows 는 LLM 컨텍스트에 재주입하지 않는다 (§6.1).
    tool_results_cache: dict[str, Any]
    # narrator 가 가장 최근 참조한 ref_id — 여러 도구 호출 시 종합 보고용 (Step 3).
    latest_tool_ref_id: str
    analysis_outputs: list[dict[str, Any]]     # pandas aggregate 결과 (Step 3 도입)

    # ── HITL 승인 (Step 5a 활성화) ──
    # awaiting_confirmation: risk_gate 에서 interrupt 발동 직전에 True 로 설정.
    #   SSE 레이어가 이 플래그를 보고 confirmation_required 이벤트 발행 여부를 결정.
    # confirmation_payload: ConfirmationPayload.model_dump() — SSE 이벤트에 그대로 실림.
    # confirmation_decision: /resume 호출 시 넘어온 ConfirmationDecision.model_dump().
    #   risk_gate 가 재개되면서 decision='reject' 면 response_text 를 직접 채우고
    #   pending_tool_call=None 으로 비워서 실행 경로를 차단.
    awaiting_confirmation: bool
    confirmation_payload: dict[str, Any] | None
    confirmation_decision: dict[str, Any] | None

    # ── 최종 응답 ──
    response_text: str
    chart_payloads: list[ChartSpec]
    table_payloads: list[TableSpec]

    # ── 제어 ──
    iteration_count: int               # tool-call 루프 횟수 (최대 5)
    error: str | None
