"""
고객센터 AI 챗봇 — vLLM EXAONE 1.2B 2단계 응답 체인 (v3.2).

### 배경
v3.1 은 Solar/Ollama 로 1회 호출이었으나 사용자 요구에 따라 VM4 의 vLLM EXAONE
4.0 1.2B 로 전환해야 한다. 이 모델은 max_model_len=2048 이라 FAQ 30건의 full answer
까지 한 번에 컨텍스트로 실을 수 없다(→ v3.1 1회 호출 불가).

### v3.2 해결책 — 2단계 분리
**Step 1 (PLAN)**: FAQ **question 만** 실어 SupportPlan 구조화 출력
    입력: 사용자 발화 + [id, question] 30건 (~1K tok)
    출력: {"kind": ..., "matched_faq_ids": [...]}
    → 가볍게 분류 + 참조할 FAQ 선정만 수행

**Step 2 (ANSWER)**: kind 별 분기
    - faq / partial   → 선정된 FAQ **answer 본문만** 실어 자연어 답변 생성
    - smalltalk       → FAQ 없이 짧은 몽글이 응답 생성
    - complaint       → LLM 호출 없이 Python 템플릿
    - out_of_scope    → LLM 호출 없이 Python 템플릿

LLM 호출 수: kind=faq/partial/smalltalk 는 2회, complaint/out_of_scope 는 1회.
vLLM EXAONE 1.2B 는 GPU 로컬이라 1회 ~500ms. 2회여도 전체 1~1.5초 안에 응답.

### 구조화 출력
with_structured_output 을 쓰는 대신 JSON mode + 수동 파싱(JsonOutputParser) 조합.
1.2B 모델이 function calling 스키마를 엄격하게 따르지 못하는 케이스가 많아,
프롬프트로 JSON 포맷을 직접 지시하고 실패 시 정규식 복원까지 내려간다.

### 실패 대응
Step 1 이 완전히 실패하면 SupportReply(kind="complaint", needs_human=True) 로 graceful fallback.
사용자는 "지금은 확인이 어렵다, 1:1 문의" 안내를 받는다.
"""

from __future__ import annotations

import json
import re
import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.support_assistant.models import (
    FaqDoc,
    SupportPlan,
    SupportReply,
)
from monglepick.llm.factory import get_vllm_llm, guarded_ainvoke
from monglepick.prompts.support_assistant import (
    SUPPORT_ANSWER_FROM_FAQ_HUMAN_PROMPT,
    SUPPORT_ANSWER_FROM_FAQ_SYSTEM_PROMPT,
    SUPPORT_COMPLAINT_TEMPLATE,
    SUPPORT_OUT_OF_SCOPE_TEMPLATE,
    SUPPORT_PLAN_HUMAN_PROMPT,
    SUPPORT_PLAN_SYSTEM_PROMPT,
    SUPPORT_SMALLTALK_HUMAN_PROMPT,
    SUPPORT_SMALLTALK_SYSTEM_PROMPT,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Step 1 — FAQ 매칭 + kind 분류 (SupportPlan JSON)
# =============================================================================

# Step 1 은 질문 목록만 받아 짧은 JSON 으로 응답하면 됨. FAQ 한 건당 "[id=1] 질문..."
# 포맷으로 축약해 컨텍스트 1K tok 이내 유지.
_FAQ_QUESTION_BLOCK_LIMIT = 120  # 한 건당 최대 문자 (질문이 극단적으로 길 경우 자름)


def _build_faq_titles(faqs: list[FaqDoc]) -> str:
    """Step 1 프롬프트용 — question 만 포함한 짧은 목록."""
    if not faqs:
        return "(등록된 FAQ 가 없습니다)"
    lines = []
    for faq in faqs:
        q = (faq.question or "").strip()
        if len(q) > _FAQ_QUESTION_BLOCK_LIMIT:
            q = q[:_FAQ_QUESTION_BLOCK_LIMIT] + "..."
        lines.append(f"[id={faq.faq_id}] {q}")
    return "\n".join(lines)


# JSON 을 모델이 코드블록 `​``json ...​``` 으로 감쌌을 때 본문만 추출하기 위한 정규식.
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _parse_plan_json(raw_text: str) -> SupportPlan | None:
    """
    LLM 응답에서 SupportPlan JSON 을 파싱한다.

    1.2B 모델이 앞뒤에 코드블록/설명을 섞을 수 있으므로 중괄호 블록을 추출해 파싱.
    실패하면 None 반환 — 호출 측이 fallback 경로로 분기한다.
    """
    if not raw_text:
        return None
    text = raw_text.strip()
    # 코드펜스 제거
    if text.startswith("```"):
        text = text.strip("`")
        # "json" prefix 가 붙는 경우 제거
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
    # 첫 중괄호 블록만 추출
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        return SupportPlan.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.debug("support_plan_validate_failed", error=str(exc), data=data)
        return None


async def _classify_and_match(
    user_message: str, faqs: list[FaqDoc]
) -> SupportPlan:
    """
    Step 1: vLLM EXAONE 1.2B 에게 FAQ 질문 목록만 주고 SupportPlan 을 받아온다.

    실패 시 SupportPlan(kind="complaint", []) 로 graceful degrade.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SUPPORT_PLAN_SYSTEM_PROMPT), ("human", SUPPORT_PLAN_HUMAN_PROMPT)]
    )
    # 분류는 결정적이어야 정확도 ↑ — temperature 0.0
    llm = get_vllm_llm(temperature=0.0)

    inputs = {
        "user_message": user_message,
        "faq_titles": _build_faq_titles(faqs),
    }

    try:
        prompt_value = await prompt.ainvoke(inputs)
        response = await guarded_ainvoke(
            llm,
            prompt_value,
            model="vllm_exaone_1_2b",
            request_id="support_plan",
        )
        # content 가 빈 문자열이면 그대로 "" 유지 (`or str(response)` 로 SimpleNamespace
        # 문자열 표현이 끼어드는 버그 방지).
        raw = (getattr(response, "content", "") or "").strip()
        plan = _parse_plan_json(raw)
        if plan is None:
            logger.warning(
                "support_plan_unparsable",
                raw_preview=raw[:200],
            )
            return SupportPlan(kind="complaint", matched_faq_ids=[])
        return plan
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "support_plan_error",
            error=str(exc),
            error_type=type(exc).__name__,
            stack_trace=traceback.format_exc(),
        )
        return SupportPlan(kind="complaint", matched_faq_ids=[])


# =============================================================================
# Step 2-A — FAQ 기반 답변 생성 (kind ∈ {faq, partial})
# =============================================================================

# Step 2 는 선정된 FAQ (최대 3건) 의 full answer 를 싣는다. 한 건당 400자로 제한해도
# 3건 × 400 = 1200자 ≈ 600 tok. system+user 포함 1.2B 의 2048 tok 에 여유.
_FAQ_ANSWER_TRUNCATE = 400


def _build_faq_answer_context(faqs: list[FaqDoc], matched_ids: list[int]) -> str:
    """Step 2-A 프롬프트용 — 선정된 FAQ 의 full answer 만 추려 직렬화."""
    by_id = {f.faq_id: f for f in faqs}
    blocks: list[str] = []
    for fid in matched_ids:
        faq = by_id.get(int(fid))
        if faq is None:
            continue
        answer = (faq.answer or "").strip()
        if len(answer) > _FAQ_ANSWER_TRUNCATE:
            answer = answer[:_FAQ_ANSWER_TRUNCATE] + "..."
        blocks.append(f"[id={faq.faq_id}] 질문: {faq.question}\n답변: {answer}")
    if not blocks:
        return "(근거 FAQ 없음)"
    return "\n\n".join(blocks)


async def _generate_answer_from_faq(
    user_message: str,
    faqs: list[FaqDoc],
    matched_ids: list[int],
    match_mode: str,  # "faq" | "partial"
) -> str:
    """Step 2-A: 선정된 FAQ 근거로 몽글이 톤 답변 자유 텍스트 생성."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPPORT_ANSWER_FROM_FAQ_SYSTEM_PROMPT),
            ("human", SUPPORT_ANSWER_FROM_FAQ_HUMAN_PROMPT),
        ]
    )
    llm = get_vllm_llm(temperature=0.3)
    inputs = {
        "user_message": user_message,
        "match_mode": match_mode,
        "faq_context": _build_faq_answer_context(faqs, matched_ids),
    }
    try:
        prompt_value = await prompt.ainvoke(inputs)
        response = await guarded_ainvoke(
            llm,
            prompt_value,
            model="vllm_exaone_1_2b",
            request_id=f"support_answer_{match_mode}",
        )
        text = (getattr(response, "content", "") or "").strip()
        return text
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "support_answer_error",
            error=str(exc),
            error_type=type(exc).__name__,
            match_mode=match_mode,
        )
        # LLM 실패 — FAQ 원문을 그대로 노출 (정확한 정보 우선)
        by_id = {f.faq_id: f for f in faqs}
        for fid in matched_ids:
            faq = by_id.get(int(fid))
            if faq is not None:
                return faq.answer
        return ""


# =============================================================================
# Step 2-B — 스몰토크 응답 (kind == smalltalk)
# =============================================================================


async def _generate_smalltalk(user_message: str) -> str:
    """짧은 몽글이 응대 — FAQ 없이 1~2문장."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPPORT_SMALLTALK_SYSTEM_PROMPT),
            ("human", SUPPORT_SMALLTALK_HUMAN_PROMPT),
        ]
    )
    llm = get_vllm_llm(temperature=0.3)
    try:
        prompt_value = await prompt.ainvoke({"user_message": user_message})
        response = await guarded_ainvoke(
            llm,
            prompt_value,
            model="vllm_exaone_1_2b",
            request_id="support_smalltalk",
        )
        return (getattr(response, "content", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "support_smalltalk_error",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return "안녕하세요! 궁금한 점이 있으면 편하게 말씀해 주세요."


# =============================================================================
# 퍼블릭 — 2단계 통합 dispatch
# =============================================================================


async def generate_support_reply(
    user_message: str,
    faqs: list[FaqDoc],
) -> SupportReply:
    """
    2단계 체인 실행 + kind 별 Step 2 분기.

    - Step 1: SupportPlan(kind, matched_faq_ids)
    - Step 2 분기:
        · faq / partial → _generate_answer_from_faq (LLM 1회)
        · smalltalk    → _generate_smalltalk (LLM 1회)
        · complaint    → SUPPORT_COMPLAINT_TEMPLATE (LLM 호출 없음)
        · out_of_scope → SUPPORT_OUT_OF_SCOPE_TEMPLATE (LLM 호출 없음)
    """
    started = time.perf_counter()
    logger.info(
        "support_reply_start",
        input_preview=user_message[:100],
        faq_count=len(faqs),
    )

    plan = await _classify_and_match(user_message, faqs)

    # plan.matched_faq_ids 중 실제 존재하지 않는 id 는 미리 드롭 (환각 방지)
    valid_ids = {f.faq_id for f in faqs}
    cleaned_ids = [fid for fid in plan.matched_faq_ids if int(fid) in valid_ids]

    # kind 가 faq/partial 인데 유효 id 가 0건이면 complaint 로 강등
    if plan.kind in ("faq", "partial") and not cleaned_ids:
        logger.info(
            "support_reply_demote_to_complaint_no_matches",
            original_kind=plan.kind,
            raw_ids=plan.matched_faq_ids,
        )
        plan = SupportPlan(kind="complaint", matched_faq_ids=[])
        cleaned_ids = []

    # kind 가 faq 인데 LLM 이 matched_ids 를 비워 보냈다면 partial 로 격하 불가 →
    # complaint 로 강등.
    # (위 조건에서 이미 처리됨)

    # Step 2 분기
    if plan.kind in ("faq", "partial"):
        answer = await _generate_answer_from_faq(
            user_message=user_message,
            faqs=faqs,
            matched_ids=cleaned_ids,
            match_mode=plan.kind,
        )
        if not answer.strip():
            # LLM 이 빈 답변을 주면 FAQ 원문 그대로 노출
            by_id = {f.faq_id: f for f in faqs}
            first = by_id.get(cleaned_ids[0]) if cleaned_ids else None
            answer = first.answer if first else SUPPORT_COMPLAINT_TEMPLATE
        needs_human = plan.kind == "partial"
        reply = SupportReply(
            kind=plan.kind,
            matched_faq_ids=cleaned_ids,
            answer=answer,
            needs_human=needs_human,
        )
    elif plan.kind == "smalltalk":
        answer = await _generate_smalltalk(user_message)
        if not answer.strip():
            answer = "안녕하세요! 궁금한 점이 있으면 편하게 말씀해 주세요."
        reply = SupportReply(
            kind="smalltalk",
            matched_faq_ids=[],
            answer=answer,
            needs_human=False,
        )
    elif plan.kind == "out_of_scope":
        reply = SupportReply(
            kind="out_of_scope",
            matched_faq_ids=[],
            answer=SUPPORT_OUT_OF_SCOPE_TEMPLATE,
            needs_human=False,
        )
    else:
        # complaint (또는 분류 실패 폴백)
        reply = SupportReply(
            kind="complaint",
            matched_faq_ids=[],
            answer=SUPPORT_COMPLAINT_TEMPLATE,
            needs_human=True,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "support_reply_done",
        kind=reply.kind,
        matched_count=len(reply.matched_faq_ids),
        needs_human=reply.needs_human,
        elapsed_ms=round(elapsed_ms, 1),
    )
    return reply
