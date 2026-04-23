"""
관리자 에이전트 Tool Selector 체인.

설계서: docs/관리자_AI에이전트_설계서.md §3.2 (tool_selector), §10 (프롬프트)

역할:
- Intent 가 stats/query/action 등으로 분류된 뒤, admin_role 에 허용된 Tool 중 **가장 적합한
  하나** 를 Solar Pro 의 `bind_tools()` 로 선택받는다.
- Step 2 범위: Tool RAG 도입 전이라 admin_role 필터된 전체 tool (최대 8~10개) 을 프롬프트에
  주입. 이후 Step 3 에서 Qdrant 기반 top-5 retrieval 로 확장될 예정.

반환:
- `SelectedTool(name, arguments, rationale)` — LLM 이 tool_call 을 내뱉은 경우
- `None` — 적절한 tool 이 없거나 호출 실패. 상위에서 narrator 가 "적절한 도구 없음" 안내.

주의:
- LangChain `bind_tools()` 는 `BaseTool` 인스턴스를 원한다. 레지스트리의 ToolSpec 을
  `StructuredTool.from_function` 으로 감싸 넘긴다. `coroutine` 은 no-op 더미를 준다 — 실제
  실행은 `tool_executor` 노드의 레지스트리 조회로 수행한다 (LLM 이 직접 실행하지 않음).
- `chain = prompt | llm` 대신 prompt 를 미리 messages 로 렌더한 뒤 llm.ainvoke 에 직접
  넘긴다. MagicMock 호환 (question_chain/admin_intent_chain 과 동일 이유).
"""

from __future__ import annotations

import time
import traceback
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from monglepick.llm.factory import get_solar_api_llm, guarded_ainvoke
from monglepick.tools.admin_tools import ToolSpec, list_tools_for_role

logger = structlog.get_logger()


# ============================================================
# 반환 모델
# ============================================================

class SelectedTool(BaseModel):
    """tool_selector 가 LLM 응답에서 추출한 단일 tool-call."""

    name: str = Field(..., description="선택된 tool 의 레지스트리 이름")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="LLM 이 제안한 인자 (args_schema 검증 전)",
    )
    rationale: str = Field(default="", description="선택 이유 (디버깅용, 옵셔널)")


# ============================================================
# 프롬프트
# ============================================================

_SELECTOR_SYSTEM_PROMPT = """당신은 몽글픽 관리자 AI 어시스턴트의 Tool Selector 다.
관리자의 자연어 발화를 읽고, 제공된 tool 중 **정확히 하나** 를 선택해 호출한다.

규칙:
1. 제공된 tool 외에는 호출하지 않는다.
2. 발화에 명시된 인자만 채우고, 언급되지 않은 인자는 **기본값** 그대로 둔다(예: `period` 기본 "7d").
3. 발화가 여러 작업을 요구해도 가장 핵심인 **하나만** 호출한다. 여러 tool 을 연달아 호출하지 않는다.
4. 적절한 tool 이 없다면 tool 을 호출하지 말고 평문으로 "적절한 도구가 없음" 이라고 답한다.
5. 관리자 권한 {admin_role} 에 허용된 tool 만 나열되어 있으므로 보이는 것만 선택 가능하다.
6. 수치를 만들어내지 않는다. tool 호출만 수행하고 결과 해석은 이 체인의 역할이 아니다.
"""


_SELECTOR_HUMAN_PROMPT = """관리자 역할: {admin_role}
분류된 의도: {intent_kind}
발화: {user_message}

위 발화에 가장 적합한 tool 을 정확히 하나 호출하라. 적절한 tool 이 없으면 호출하지 말 것.
"""


_selector_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SELECTOR_SYSTEM_PROMPT),
        ("human", _SELECTOR_HUMAN_PROMPT),
    ]
)


# ============================================================
# ToolSpec → LangChain StructuredTool 변환
# ============================================================

async def _selector_noop(**kwargs: Any) -> str:
    """
    LLM 이 tool_call 을 내뱉기만 하면 되므로 실제 실행은 이 no-op 이 받아 폐기한다.

    tool_executor 노드가 ADMIN_TOOL_REGISTRY 에서 원본 handler 를 다시 조회해 실행한다.
    이 분리 덕분에 LLM 이 직접 Backend 를 호출하지 않고, 실행은 에이전트 본체에 남는다.
    """
    return ""


def _to_structured_tool(spec: ToolSpec) -> StructuredTool:
    """
    ToolSpec → LangChain StructuredTool 변환.

    - name / description: ToolSpec 값 그대로
    - args_schema: ToolSpec.args_schema (Pydantic BaseModel)
    - coroutine: 더미 no-op (LLM 은 선택만 하고 실제 실행은 tool_executor 가 한다)
    """
    return StructuredTool.from_function(
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_schema,
        coroutine=_selector_noop,
    )


# ============================================================
# 핵심 함수
# ============================================================

async def select_admin_tool(
    user_message: str,
    admin_role: str,
    intent_kind: str,
    request_id: str = "",
) -> SelectedTool | None:
    """
    관리자 발화 + intent 를 보고 최적 Tool 하나를 선택한다.

    Args:
        user_message: 관리자 자연어 입력
        admin_role: 정규화된 AdminRoleEnum — 레지스트리 필터 기준
        intent_kind: AdminIntent.kind (query/action/stats/report/sql/smalltalk)
        request_id: 동시성 슬롯 / 로그 식별자

    Returns:
        SelectedTool: LLM 이 tool 을 선택한 경우
        None: tool 없음 / 권한 없음 / LLM 에러 (어떤 경우든 graceful)
    """
    start = time.perf_counter()

    # 1) admin_role 필터 — 하나도 허용 없으면 조기 반환
    allowed = list_tools_for_role(admin_role)
    if not allowed:
        logger.info(
            "admin_tool_selector_no_allowed_tools",
            admin_role=admin_role or "(blank)",
        )
        return None

    # 2) Solar API LLM + bind_tools
    try:
        llm = get_solar_api_llm(temperature=0.1)
        tools = [_to_structured_tool(s) for s in allowed]
        # tool_choice="auto": LLM 이 자율 판단 (tool 없으면 text 반환 가능)
        llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

        # `to_messages()` 대신 ChatPromptValue 자체를 넘긴다 — admin_intent_chain.py 와 동일.
        # bind_tools(...) 결과는 ToolCalling Runnable 로 ChatPromptValue/list 양쪽 모두 받지만,
        # LangChain 향후 변경에 대비해 admin_intent_chain.py 와 입력 형식 통일.
        prompt_value = _selector_prompt.format_prompt(
            user_message=user_message.strip(),
            admin_role=admin_role or "UNKNOWN",
            intent_kind=intent_kind or "unknown",
        )
        response = await guarded_ainvoke(
            llm_with_tools,
            prompt_value,
            model="solar_api",
            request_id=request_id or "admin_tool_selector",
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "admin_tool_selector_failed",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        return None

    # 3) tool_calls 추출 — LangChain 0.3 규약: response.tool_calls 는 list[dict]
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        elapsed_ms = (time.perf_counter() - start) * 1000
        content_preview = (getattr(response, "content", "") or "")[:120]
        logger.info(
            "admin_tool_selector_no_tool_call",
            content_preview=content_preview,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return None

    first = tool_calls[0]
    name = first.get("name") or ""
    arguments = first.get("args") or {}

    # 4) 허용 tool 집합 내부인지 재검증 (프롬프트 지시 위반 방어)
    allowed_names = {s.name for s in allowed}
    if name not in allowed_names:
        logger.warning(
            "admin_tool_selector_disallowed_tool_rejected",
            tool_name=name,
            allowed=sorted(allowed_names),
        )
        return None

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "admin_tool_selected",
        tool_name=name,
        args_keys=sorted(arguments.keys()),
        elapsed_ms=round(elapsed_ms, 1),
    )
    return SelectedTool(
        name=name,
        arguments=arguments,
        rationale=f"bind_tools[{len(allowed)}] auto-selected",
    )
