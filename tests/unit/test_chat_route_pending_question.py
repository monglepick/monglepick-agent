"""
Chat agent route_after_intent 의 pending_question 우선순위 테스트 (2026-05-07 회귀 픽스).

대상:
 - monglepick.agents.chat.graph.route_after_intent

핵심 계약:
 1) pending_question="awaiting_location" 이 set 돼 있으면 LLM 의 의도 분류 결과와 무관하게
    무조건 tool_executor_node 로 라우팅된다 (단일 토큰 "강남역" 답변이 general/info 로 잘못
    분류돼도 위치 해소 흐름이 끊기지 않게 보장).
 2) pending_question 이 None / 미설정 / 다른 값이면 기존 의도 기반 라우팅 동작 유지.
"""

from __future__ import annotations

from monglepick.agents.chat.graph import route_after_intent
from monglepick.agents.chat.models import IntentResult


def _state(intent: str, pending_question: str | None = None) -> dict:
    return {
        "intent": IntentResult(intent=intent, confidence=0.9),
        "pending_question": pending_question,
    }


class TestPendingLocationOverride:
    def test_pending_location_overrides_general_intent(self):
        """LLM 이 'general' 로 분류해도 pending_question 이 있으면 tool_executor_node."""
        state = _state("general", pending_question="awaiting_location")
        assert route_after_intent(state) == "tool_executor_node"

    def test_pending_location_overrides_search_intent(self):
        """search 로 잘못 분류돼도 우선 처리."""
        state = _state("search", pending_question="awaiting_location")
        assert route_after_intent(state) == "tool_executor_node"

    def test_pending_location_overrides_recommend_intent(self):
        """recommend 로 잘못 분류돼도 우선 처리."""
        state = _state("recommend", pending_question="awaiting_location")
        assert route_after_intent(state) == "tool_executor_node"

    def test_pending_location_with_theater_intent_still_routes_to_tool(self):
        """이미 theater 로 분류된 케이스도 tool_executor_node 로 라우팅 — 거짓양성 없음."""
        state = _state("theater", pending_question="awaiting_location")
        assert route_after_intent(state) == "tool_executor_node"

    def test_no_pending_question_uses_intent_based_routing(self):
        """pending_question 없으면 기존 라우팅 그대로 — recommend → preference_refiner."""
        state = _state("recommend", pending_question=None)
        assert route_after_intent(state) == "preference_refiner"

    def test_unrelated_pending_question_doesnt_override(self):
        """pending_question 값이 awaiting_location 이 아니면 무시 — 기존 라우팅 유지."""
        state = _state("general", pending_question="some_other_value")
        assert route_after_intent(state) == "general_responder"

    def test_pending_question_missing_key_falls_through(self):
        """state 에 pending_question 키 자체가 없어도 안전하게 fall-through."""
        state = {"intent": IntentResult(intent="general", confidence=0.8)}
        assert route_after_intent(state) == "general_responder"
