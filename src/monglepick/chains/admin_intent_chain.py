"""
관리자 에이전트 Intent 분류 체인.

Solar Pro API 의 structured output(`method="json_schema"`) 로 관리자 발화를
6종 AdminIntentKind(query/action/stats/report/sql/smalltalk) 중 하나로 분류한다.

설계서: docs/관리자_AI에이전트_설계서.md §10.2 AdminIntent

구조:
    발화 → ChatPromptTemplate → structured Solar LLM → AdminIntent

Fallback 정책:
- LLM 호출 실패(타임아웃/API 오류) 시 smalltalk intent 로 폴백하여
  에이전트가 대화를 중단하지 않도록 한다. (실제 작업은 수행 안함)
- confidence < 0.5 는 smalltalk 로 보정 — "애매하면 안전하게 안내만" 정책.
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.admin_assistant.models import AdminIntent
from monglepick.llm.factory import get_structured_llm, guarded_ainvoke
from monglepick.prompts.admin_assistant import (
    INTENT_HUMAN_PROMPT,
    INTENT_SYSTEM_PROMPT,
)

logger = structlog.get_logger()


# ============================================================
# 프롬프트 템플릿 (모듈 레벨 싱글턴)
# ============================================================

_intent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", INTENT_SYSTEM_PROMPT),
        ("human", INTENT_HUMAN_PROMPT),
    ]
)


# ============================================================
# 신뢰도 하한 — 아래면 smalltalk 로 강제 보정
# ============================================================

_MIN_CONFIDENCE: float = 0.5


# ============================================================
# 체인 호출
# ============================================================


async def classify_admin_intent(
    user_message: str,
    admin_role: str,
    request_id: str = "",
) -> AdminIntent:
    """
    관리자 발화를 AdminIntent(6종 kind) 로 분류한다.

    Solar Pro API 의 `with_structured_output(method="json_schema")` 로
    AdminIntent Pydantic 모델을 직접 받는다. Ollama fallback 경로는
    Step 1 범위 밖이다 — LLM_MODE=hybrid/api_only 에서만 정상 동작하며
    local_only 에서는 Ollama structured output 에 자동 전환된다
    (factory.get_structured_llm 이 처리).

    Args:
        user_message: 관리자가 입력한 발화 (1줄 이상)
        admin_role: 정규화된 AdminRoleEnum 값 (프롬프트에 포함되어 역할별 맥락 유지)
        request_id: 세마포어/로그용 요청 식별자 (선택)

    Returns:
        AdminIntent — kind/confidence/reason 3필드. 실패 시 smalltalk 로 폴백.
    """
    start = time.perf_counter()
    try:
        llm = get_structured_llm(schema=AdminIntent, temperature=0.1, use_api=True)

        # `_intent_prompt | llm` chain 패턴을 쓰지 않는 이유:
        # 단위 테스트에서 `with_structured_output()` 이 반환하는 MagicMock 에 `|` 연산자가
        # 없어 RunnableSequence 구성이 깨지기 때문 (MagicMock 자동 propagation 으로
        # chain 자체가 MagicMock 이 되어 ainvoke 결과가 Pydantic 이 아닌 MagicMock 이
        # 반환되는 검증 에러 발생). llm 직접 호출은 mock 과 실 모듈 모두 호환.
        #
        # 단, 입력은 ChatPromptValue (`format_prompt()` 결과) 를 그대로 넘긴다.
        # `to_messages()` 로 list[BaseMessage] 변환 시 LangChain 0.3+ 의
        # `with_structured_output(method="json_schema")` 가 내부에서 추가하는
        # ChatPromptTemplate input validation 에 걸려 "Expected mapping type" 에러 발생.
        # ChatPromptValue 자체는 dict-like / messages-like 양쪽 인터페이스를 모두 만족.
        prompt_value = _intent_prompt.format_prompt(
            user_message=user_message.strip(),
            admin_role=admin_role or "UNKNOWN",
        )
        result = await guarded_ainvoke(
            llm,
            prompt_value,
            model="solar_api",
            request_id=request_id or "admin_intent",
        )

        # LLM 이 BaseModel 이 아닌 dict 를 돌려주는 경로 대비 graceful 파싱
        if not isinstance(result, AdminIntent):
            result = AdminIntent.model_validate(result)

        # confidence 하한 보정
        if result.confidence < _MIN_CONFIDENCE and result.kind != "smalltalk":
            logger.info(
                "admin_intent_low_confidence_fallback_to_smalltalk",
                kind=result.kind,
                confidence=result.confidence,
                reason=result.reason,
            )
            result = AdminIntent(
                kind="smalltalk",
                confidence=result.confidence,
                reason=f"low_confidence_fallback (원래: {result.kind}, {result.reason})",
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "admin_intent_classified",
            kind=result.kind,
            confidence=round(result.confidence, 2),
            reason=result.reason[:80],
            elapsed_ms=round(elapsed_ms, 1),
        )
        return result

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "admin_intent_classify_failed_fallback_smalltalk",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        # 에러 전파 금지 (설계서 §3 "에러: 모든 노드/체인 try/except, 실패 시 fallback 반환")
        return AdminIntent(
            kind="smalltalk",
            confidence=0.0,
            reason=f"classify_error:{type(e).__name__}",
        )
