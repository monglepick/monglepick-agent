"""
support_reply_chain v3.2 — vLLM EXAONE 1.2B 2단계 체인 단위 테스트.

vLLM 실 호출 없이 `guarded_ainvoke` 를 monkeypatch 로 대체해 Step 1/Step 2 의
입출력과 분기 동작을 검증한다.

검증 범위:
- `_parse_plan_json`: JSON 포맷/코드펜스/잘못된 텍스트 파싱
- `_classify_and_match`: 응답 파싱 실패 시 complaint fallback
- `generate_support_reply` 5가지 kind 분기:
    · faq / partial → Step 2-A 답변 생성 호출
    · smalltalk     → Step 2-B 스몰토크 호출
    · complaint     → LLM 호출 없이 템플릿
    · out_of_scope  → LLM 호출 없이 템플릿
- faq 인데 matched_ids 가 비면 complaint 로 강등
- 환각 ID (실제 존재하지 않는 faq_id) 는 드롭
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from monglepick.agents.support_assistant.models import FaqDoc, SupportPlan
from monglepick.chains import support_reply_chain as chain_mod
from monglepick.chains.support_reply_chain import (
    _parse_plan_json,
    generate_support_reply,
)
from monglepick.prompts.support_assistant import (
    SUPPORT_COMPLAINT_TEMPLATE,
    SUPPORT_OUT_OF_SCOPE_TEMPLATE,
)


# =============================================================================
# 픽스처 — FAQ 목록 공통
# =============================================================================


@pytest.fixture
def sample_faqs() -> list[FaqDoc]:
    return [
        FaqDoc(
            faq_id=1,
            category="GENERAL",
            question="고객센터 전화번호와 연락처가 어떻게 되나요?",
            answer="이메일 contact@monglepick.com 과 1:1 문의 창구로 운영됩니다.",
            sort_order=50,
        ),
        FaqDoc(
            faq_id=2,
            category="ACCOUNT",
            question="비밀번호를 잊어버렸어요. 어떻게 재설정하나요?",
            answer="로그인 페이지 하단 '비밀번호 찾기' 링크로 이메일 재설정 링크를 받으세요.",
            sort_order=20,
        ),
    ]


# =============================================================================
# vLLM stub — `guarded_ainvoke` 를 순서대로 응답하도록 교체
# =============================================================================


class _FakeInvokeRecorder:
    """호출 순서대로 cued response 를 반환. 호출 횟수·request_id 를 기록."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []  # request_id 기록

    async def __call__(self, llm, prompt_value, model, request_id=""):
        self.calls.append(request_id)
        if not self._responses:
            raise RuntimeError(f"no stub response left for request_id={request_id}")
        text = self._responses.pop(0)
        return SimpleNamespace(content=text)


def _install_stub(monkeypatch, responses: list[str]) -> _FakeInvokeRecorder:
    """`guarded_ainvoke` 를 recorder 로 치환하고 인스턴스를 반환."""
    rec = _FakeInvokeRecorder(responses)
    monkeypatch.setattr(chain_mod, "guarded_ainvoke", rec)
    # get_vllm_llm 은 실제로 호출되지 않지만 LLM_MODE 판정을 안전하게 우회하기 위해 stub.
    monkeypatch.setattr(chain_mod, "get_vllm_llm", lambda temperature=0.0: object())
    return rec


# =============================================================================
# 1) JSON 파서
# =============================================================================


class TestParsePlanJson:
    def test_plain_json(self):
        p = _parse_plan_json('{"kind": "faq", "matched_faq_ids": [1, 2]}')
        assert p is not None
        assert p.kind == "faq"
        assert p.matched_faq_ids == [1, 2]

    def test_code_fence_wrapped(self):
        p = _parse_plan_json(
            '```json\n{"kind": "smalltalk", "matched_faq_ids": []}\n```'
        )
        assert p is not None
        assert p.kind == "smalltalk"

    def test_with_surrounding_text(self):
        p = _parse_plan_json(
            '여기 결과입니다: {"kind": "complaint", "matched_faq_ids": []} 참고하세요.'
        )
        assert p is not None
        assert p.kind == "complaint"

    def test_non_json_returns_none(self):
        assert _parse_plan_json("그냥 텍스트") is None
        assert _parse_plan_json("") is None


# =============================================================================
# 2) generate_support_reply — 5가지 kind 분기
# =============================================================================


@pytest.mark.asyncio
class TestGenerateSupportReply:
    async def test_kind_faq_calls_step2_answer(self, monkeypatch, sample_faqs):
        """faq 분류 → Step 2-A 답변 생성 호출, needs_human=False."""
        rec = _install_stub(
            monkeypatch,
            [
                # Step 1 — plan JSON
                '{"kind": "faq", "matched_faq_ids": [2]}',
                # Step 2-A — 자유 텍스트 답변
                "비밀번호는 '비밀번호 찾기' 에서 이메일 인증으로 재설정하실 수 있어요.",
            ],
        )
        reply = await generate_support_reply(
            user_message="비밀번호 변경하고 싶어요",
            faqs=sample_faqs,
        )
        assert reply.kind == "faq"
        assert reply.matched_faq_ids == [2]
        assert "비밀번호" in reply.answer
        assert reply.needs_human is False
        # Step 1 + Step 2-A 두 번 호출됨
        assert rec.calls == ["support_plan", "support_answer_faq"]

    async def test_kind_partial_sets_needs_human_true(
        self, monkeypatch, sample_faqs
    ):
        rec = _install_stub(
            monkeypatch,
            [
                '{"kind": "partial", "matched_faq_ids": [2]}',
                "완전히 일치하는 안내는 아니지만 비슷한 내용이 있어요... '문의하기' 탭",
            ],
        )
        reply = await generate_support_reply(
            user_message="비밀번호 관련 문의",
            faqs=sample_faqs,
        )
        assert reply.kind == "partial"
        assert reply.matched_faq_ids == [2]
        assert reply.needs_human is True
        assert rec.calls == ["support_plan", "support_answer_partial"]

    async def test_kind_smalltalk_calls_step2_smalltalk(
        self, monkeypatch, sample_faqs
    ):
        rec = _install_stub(
            monkeypatch,
            [
                '{"kind": "smalltalk", "matched_faq_ids": []}',
                "안녕하세요! 도와드릴 일이 있으면 편하게 말씀해 주세요.",
            ],
        )
        reply = await generate_support_reply(
            user_message="안녕?",
            faqs=sample_faqs,
        )
        assert reply.kind == "smalltalk"
        assert reply.matched_faq_ids == []
        assert reply.needs_human is False
        assert reply.answer.startswith("안녕하세요")
        assert rec.calls == ["support_plan", "support_smalltalk"]

    async def test_kind_complaint_uses_template_no_step2_llm_call(
        self, monkeypatch, sample_faqs
    ):
        """complaint 는 Step 2 LLM 호출 없이 고정 템플릿."""
        rec = _install_stub(
            monkeypatch,
            ['{"kind": "complaint", "matched_faq_ids": []}'],
        )
        reply = await generate_support_reply(
            user_message="결제 에러 계속 나요 긴급",
            faqs=sample_faqs,
        )
        assert reply.kind == "complaint"
        assert reply.matched_faq_ids == []
        assert reply.needs_human is True
        assert reply.answer == SUPPORT_COMPLAINT_TEMPLATE
        # Step 1 한 번만 호출돼야 함
        assert rec.calls == ["support_plan"]

    async def test_kind_out_of_scope_uses_template_no_step2_llm_call(
        self, monkeypatch, sample_faqs
    ):
        rec = _install_stub(
            monkeypatch,
            ['{"kind": "out_of_scope", "matched_faq_ids": []}'],
        )
        reply = await generate_support_reply(
            user_message="봉준호 감독 영화 추천해줘",
            faqs=sample_faqs,
        )
        assert reply.kind == "out_of_scope"
        assert reply.answer == SUPPORT_OUT_OF_SCOPE_TEMPLATE
        assert reply.needs_human is False
        assert rec.calls == ["support_plan"]


# =============================================================================
# 3) Graceful degrade
# =============================================================================


@pytest.mark.asyncio
class TestGracefulDegrade:
    async def test_step1_unparsable_response_falls_back_to_complaint(
        self, monkeypatch, sample_faqs
    ):
        """Step 1 이 JSON 이 아닌 헛소리를 돌려주면 complaint 템플릿으로 폴백."""
        rec = _install_stub(
            monkeypatch,
            ["알 수 없는 텍스트"],  # JSON 아님
        )
        reply = await generate_support_reply(
            user_message="테스트",
            faqs=sample_faqs,
        )
        assert reply.kind == "complaint"
        assert reply.answer == SUPPORT_COMPLAINT_TEMPLATE
        assert reply.needs_human is True
        assert rec.calls == ["support_plan"]

    async def test_hallucinated_faq_ids_are_dropped_and_demoted(
        self, monkeypatch, sample_faqs
    ):
        """
        LLM 이 faq 라고 해놓고 실제 존재하지 않는 id(999) 만 돌려주면
        cleaned_ids 가 [] → complaint 로 강등.
        """
        rec = _install_stub(
            monkeypatch,
            ['{"kind": "faq", "matched_faq_ids": [999]}'],
        )
        reply = await generate_support_reply(
            user_message="뭔가 찾는 거",
            faqs=sample_faqs,
        )
        assert reply.kind == "complaint"
        assert reply.matched_faq_ids == []
        # Step 2 호출 안 됨 (강등 후 템플릿 사용)
        assert rec.calls == ["support_plan"]

    async def test_faq_kind_with_partially_valid_ids_keeps_valid_only(
        self, monkeypatch, sample_faqs
    ):
        """faq_id=2 는 존재, 999 는 환각 → 2만 살리고 Step 2-A 호출."""
        rec = _install_stub(
            monkeypatch,
            [
                '{"kind": "faq", "matched_faq_ids": [999, 2]}',
                "비밀번호는 '비밀번호 찾기' 에서 재설정하실 수 있어요.",
            ],
        )
        reply = await generate_support_reply(
            user_message="비밀번호",
            faqs=sample_faqs,
        )
        assert reply.kind == "faq"
        assert reply.matched_faq_ids == [2]  # 999 는 드롭
        assert "비밀번호" in reply.answer
        assert rec.calls == ["support_plan", "support_answer_faq"]

    async def test_step2_empty_answer_falls_back_to_faq_raw(
        self, monkeypatch, sample_faqs
    ):
        """LLM 이 빈 답변을 돌려주면 FAQ 원문을 그대로 노출."""
        rec = _install_stub(
            monkeypatch,
            [
                '{"kind": "faq", "matched_faq_ids": [2]}',
                "",  # 빈 응답
            ],
        )
        reply = await generate_support_reply(
            user_message="비밀번호 잊어버렸어요",
            faqs=sample_faqs,
        )
        assert reply.kind == "faq"
        assert reply.matched_faq_ids == [2]
        # FAQ 원문 answer 가 그대로 노출되어야 함
        assert "비밀번호 찾기" in reply.answer
