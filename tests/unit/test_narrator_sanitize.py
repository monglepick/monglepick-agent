"""
narrator 응답 후처리 `_sanitize_narrator_output` 단위 테스트 (2026-04-23 운영 발견).

사건: 실제 운영에서 Solar Pro narrator 가 시스템 프롬프트의 규칙과 자체 검증 체크리스트를
본문에 섞어 노출했다. 예:

    지난 7일간 DAU 추이는 모든 날짜에 0명으로 기록되었습니다.
    [근거: stats_trends · 기간=7d]

    (※ 실제 응답 시 "새 수치를 만들어내지 말 것" 규칙에 따라 평균/추세 분석 등을 추가하지 않음)

    ---
    **검증 사항**
    1. tool_result의 dau 값(0)만 인용
    2. __list_total__=7, truncated=true 사실을 명시
    ...

`_sanitize_narrator_output` 은 이런 메타 잔존을 공격적으로 잘라낸다.
"""

from __future__ import annotations

from monglepick.agents.admin_assistant.nodes import _sanitize_narrator_output


class TestSanitizeNarratorOutput:
    def test_passthrough_clean_text(self):
        text = (
            "지난 7일 DAU 는 1,234명이에요.\n"
            "[출처: stats_overview · 기간=7d]"
        )
        assert _sanitize_narrator_output(text) == text

    def test_cuts_triple_dash_divider_and_following_meta(self):
        """
        실제 운영 로그에서 관찰된 패턴:
        본문 → [근거: ...] → (※ ...) → --- → **검증 사항** → 번호 리스트
        """
        raw = (
            "지난 7일간 DAU 추이는 모든 날짜에 0명으로 기록되었습니다.\n"
            "[근거: stats_trends · 기간=7d]\n\n"
            "(※ 실제 응답 시 \"새 수치를 만들어내지 말 것\" 규칙에 따라 평균/추세 분석 등을 추가하지 않음)\n\n"
            "---\n"
            "**검증 사항**\n"
            "1. tool_result의 dau 값(0)만 인용\n"
            "2. __list_total__=7, truncated=true 사실을 명시\n"
            "3. \"요청 실패\" 조건 미해당 (ok=true)\n"
            "4. 3문장 이내로 요약\n"
            "5. 숫자 포맷팅 없음(원본이 0이므로)\n"
        )
        cleaned = _sanitize_narrator_output(raw)
        # 본문 + 근거 라인은 보존
        assert "지난 7일간 DAU" in cleaned
        assert "[근거: stats_trends" in cleaned
        # 메타 블록 전면 제거
        assert "---" not in cleaned
        assert "검증 사항" not in cleaned
        assert "__list_total__" not in cleaned
        assert "tool_result의 dau" not in cleaned
        assert "실제 응답 시" not in cleaned
        # "(※ ...)" 괄호 문단도 제거
        assert "※" not in cleaned

    def test_cuts_verification_header_even_without_dash(self):
        """`---` 없이도 '**검증 사항**' 만으로도 이후 섹션이 잘려야 한다."""
        raw = (
            "FAQ 12건 등록돼 있어요.\n"
            "[출처: faqs_list]\n\n"
            "**검증 사항**\n"
            "1. 수치 인용 OK"
        )
        cleaned = _sanitize_narrator_output(raw)
        assert "FAQ 12건" in cleaned
        assert "검증 사항" not in cleaned

    def test_inline_parenthetical_meta_removed(self):
        """본문 사이에 낀 '(※ 실제 응답 시 ...)' 괄호 문단도 제거된다."""
        raw = (
            "총 주문 5건을 찾았어요.\n"
            "(※ 실제 응답 시 내부 규칙 적용)\n"
            "[출처: orders_list · status=REFUNDED]"
        )
        cleaned = _sanitize_narrator_output(raw)
        assert "총 주문 5건" in cleaned
        assert "[출처: orders_list" in cleaned
        assert "실제 응답 시" not in cleaned

    def test_empty_string_noop(self):
        assert _sanitize_narrator_output("") == ""

    def test_collapses_excess_blank_lines(self):
        raw = "첫 줄\n\n\n\n두 번째 줄"
        cleaned = _sanitize_narrator_output(raw)
        # 3개 이상 연속 개행은 2개로 축약
        assert "\n\n\n" not in cleaned
        assert "첫 줄\n\n두 번째 줄" == cleaned

    def test_sagu_gwajeong_header_cut(self):
        """'사고 과정' 헤더로 시작하는 메타 블록도 제거."""
        raw = (
            "DAU 1,234명.\n"
            "[출처: stats_overview]\n"
            "사고 과정\n"
            "- tool_result 에서 dau 만 인용"
        )
        cleaned = _sanitize_narrator_output(raw)
        assert "사고 과정" not in cleaned
        assert "DAU 1,234명" in cleaned
