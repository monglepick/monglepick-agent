"""
external_search_node + route_after_retrieval 분기 단위 테스트 (2026-04-23 추가).

대상:
 - monglepick.agents.chat.nodes.external_search_node
 - monglepick.agents.chat.graph._has_recency_signal
 - monglepick.agents.chat.graph.route_after_retrieval (최신 시그널 분기)

핵심 계약:
 1) "최신/올해/2026년" 키워드 OR dynamic_filters[release_year>=N] 이 current_year-1 이상이면
    _has_recency_signal → True
 2) 후보 0건 + recency_signal True → route_after_retrieval 이 "external_search_node" 반환
 3) external_search_node 는 DDGS 결과를 RankedMovie 스텁으로 변환해 ranked_movies 에 담는다
 4) DDGS 결과 0건이어도 ranked_movies=[] 로 graceful degrade
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from monglepick.agents.chat.graph import _has_recency_signal, route_after_retrieval
from monglepick.agents.chat.models import ExtractedPreferences, FilterCondition
from monglepick.agents.chat.nodes import external_search_node


# ============================================================
# 1. _has_recency_signal
# ============================================================


class TestHasRecencySignal:
    """_has_recency_signal: 최신 영화 시그널 판정."""

    def test_recent_filter_returns_true(self):
        """dynamic_filters[release_year>=current_year-1] 이면 True."""
        current_year = datetime.now().year
        prefs = ExtractedPreferences(
            dynamic_filters=[
                FilterCondition(field="release_year", operator="gte", value=str(current_year)),
            ],
        )
        state = {"current_input": "", "preferences": prefs}
        assert _has_recency_signal(state) is True

    def test_old_filter_returns_false(self):
        """release_year>=2010 같은 오래된 필터는 최신 시그널이 아님."""
        prefs = ExtractedPreferences(
            dynamic_filters=[
                FilterCondition(field="release_year", operator="gte", value="2010"),
            ],
        )
        state = {"current_input": "추천해줘", "preferences": prefs}
        assert _has_recency_signal(state) is False

    @pytest.mark.parametrize("keyword", ["최신", "최근", "올해", "신작", "요즘"])
    def test_recency_keywords_in_input(self, keyword):
        """원문 입력에 최신 키워드가 있으면 True."""
        state = {"current_input": f"{keyword} 영화 추천"}
        assert _has_recency_signal(state) is True

    def test_future_year_in_input(self):
        """current_year / current_year+1 숫자가 직접 언급되면 True."""
        current_year = datetime.now().year
        state = {"current_input": f"{current_year}년 개봉 영화"}
        assert _has_recency_signal(state) is True

    def test_plain_request_returns_false(self):
        """아무 시그널도 없는 일반 요청은 False."""
        state = {"current_input": "재밌는 영화 추천해줘"}
        assert _has_recency_signal(state) is False

    def test_invalid_filter_value_does_not_crash(self):
        """release_year value 가 숫자가 아니어도 graceful."""
        prefs = ExtractedPreferences(
            dynamic_filters=[
                FilterCondition(field="release_year", operator="gte", value="not-a-number"),
            ],
        )
        state = {"current_input": "", "preferences": prefs}
        # 파싱 실패 → False (크래시 X)
        assert _has_recency_signal(state) is False


# ============================================================
# 2. route_after_retrieval: 최신 시그널 분기
# ============================================================


class TestRouteAfterRetrievalExternal:
    """route_after_retrieval: 후보 0건 + 최신 시그널 → external_search_node."""

    def test_zero_candidates_with_recency_routes_external(self):
        state = {
            "candidate_movies": [],
            "current_input": "2026년 영화 추천",
            "turn_count": 1,
        }
        assert route_after_retrieval(state) == "external_search_node"

    def test_zero_candidates_without_recency_routes_question(self):
        """최신 시그널 없이 후보 0건이면 기존대로 question_generator."""
        state = {
            "candidate_movies": [],
            "current_input": "영화 추천",
            "turn_count": 1,
        }
        assert route_after_retrieval(state) == "question_generator"

    def test_has_candidates_ignores_external_branch(self):
        """후보가 있으면 최신 시그널 여부와 무관하게 external 로 가지 않는다."""
        # rrf_score 0.1 이상 후보 5개 → quality_passed True
        from monglepick.agents.chat.models import CandidateMovie

        candidates = [
            CandidateMovie(id=str(i), title=f"M{i}", rrf_score=0.1) for i in range(5)
        ]
        state = {
            "candidate_movies": candidates,
            "current_input": "최신 영화 추천",  # 시그널 있음
            "turn_count": 1,
        }
        result = route_after_retrieval(state)
        assert result != "external_search_node"


# ============================================================
# 3. external_search_node
# ============================================================


class TestExternalSearchNode:
    """external_search_node: DDGS 스텁 결과 → RankedMovie 변환."""

    @pytest.mark.asyncio
    async def test_returns_ranked_stubs_from_ddg(self):
        """DDGS 결과가 있으면 RankedMovie 스텁이 ranked_movies 에 담긴다."""
        fake_ddg = [
            {
                "title": "2026 영화 목록",
                "body": "「괴물 리턴즈」(2026) 봉준호 감독의 신작.",
                "href": "https://namu.wiki/w/2026",
            },
        ]
        prefs = ExtractedPreferences(
            user_intent="2026년 최신 영화",
            dynamic_filters=[
                FilterCondition(field="release_year", operator="gte", value="2026"),
            ],
        )
        state = {
            "preferences": prefs,
            "current_input": "2026년 영화 추천",
            "session_id": "s1",
            "user_id": "u1",
        }

        with patch(
            "monglepick.utils.movie_info_enricher._search_duckduckgo",
            return_value=fake_ddg,
        ):
            result = await external_search_node(state)

        ranked = result["ranked_movies"]
        assert len(ranked) == 1
        assert ranked[0].title == "괴물 리턴즈"
        assert ranked[0].release_year == 2026
        assert ranked[0].id.startswith("external_")
        assert ranked[0].rank == 1
        assert "외부 웹" in ranked[0].explanation
        # 외부 출처 URL 이 overview 에 포함되어야 한다
        assert "namu.wiki" in ranked[0].overview

    @pytest.mark.asyncio
    async def test_empty_ddg_returns_empty_ranked(self):
        """DDGS 결과가 없어도 graceful — ranked_movies=[]."""
        state = {
            "preferences": ExtractedPreferences(user_intent=""),
            "current_input": "신작 영화",
            "session_id": "s1",
            "user_id": "u1",
        }

        with patch(
            "monglepick.utils.movie_info_enricher._search_duckduckgo",
            return_value=[],
        ):
            result = await external_search_node(state)

        assert result["ranked_movies"] == []

    @pytest.mark.asyncio
    async def test_extracts_release_year_from_dynamic_filters(self):
        """dynamic_filters 의 release_year 하한이 search_external_movies 로 전달된다."""
        prefs = ExtractedPreferences(
            dynamic_filters=[
                FilterCondition(field="release_year", operator="gte", value="2026"),
            ],
        )
        state = {
            "preferences": prefs,
            "current_input": "추천",
            "session_id": "s1",
            "user_id": "u1",
        }

        captured_args: dict = {}

        async def fake_search(**kwargs):
            captured_args.update(kwargs)
            return []

        with patch(
            "monglepick.agents.chat.nodes.search_external_movies",
            side_effect=fake_search,
        ):
            await external_search_node(state)

        assert captured_args.get("release_year_gte") == 2026

    @pytest.mark.asyncio
    async def test_ddg_exception_returns_empty_gracefully(self):
        """search_external_movies 가 예외를 던져도 빈 리스트로 graceful."""
        state = {
            "preferences": ExtractedPreferences(),
            "current_input": "최신 영화",
            "session_id": "s1",
            "user_id": "u1",
        }

        async def raise_exc(**kwargs):
            raise RuntimeError("network down")

        with patch(
            "monglepick.agents.chat.nodes.search_external_movies",
            side_effect=raise_exc,
        ):
            result = await external_search_node(state)

        assert result == {"ranked_movies": []}
