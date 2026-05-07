"""
tool_executor_node 단위 테스트 (Phase 6 외부 지도 연동).

대상:
 - monglepick.agents.chat.nodes.tool_executor_node
 - monglepick.agents.chat.nodes._extract_location_hint
 - monglepick.agents.chat.nodes._format_tool_response

핵심 계약:
 1) intent 가 info/theater/booking 외 → 도구 미실행, 안내 메시지만
 2) theater 의도 + state.location 있음 → execute_tool 호출 (geocoding skip)
 3) theater 의도 + location 없음 + 지명 추출 hit → geocoding 후 execute_tool 호출
 4) theater 의도 + location 없음 + 지명도 없음 → 위치 재질의 메시지 (execute_tool 호출 X)
 5) execute_tool 결과는 state.tool_results 에 저장
 6) 도구 호출 중 예외 → 에러 전파 X, fallback 응답
 7) _extract_location_hint: "강남역", "홍대 근처" 등 패턴별 매칭
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monglepick.agents.chat.models import IntentResult, Location
from monglepick.agents.chat.nodes import (
    _extract_location_hint,
    _format_tool_response,
    tool_executor_node,
)


def _make_mock_tool(return_value=None):
    """LangChain @tool 객체(StructuredTool)를 흉내내는 mock — ainvoke 만 AsyncMock 으로."""
    fake = MagicMock()
    fake.ainvoke = AsyncMock(return_value=return_value)
    return fake


# ============================================================
# 1. _extract_location_hint
# ============================================================


class TestExtractLocationHint:
    @pytest.mark.parametrize("text,expected", [
        ("강남역 근처 영화관", "강남역"),
        ("홍대입구역 알려줘", "홍대입구역"),
        ("강남 근처 CGV", "강남"),
        ("신촌 주변 영화관", "신촌"),
        ("잠실동 영화관 어디 있어", "잠실동"),
        ("서울시 강남구 영화관", "강남구"),
    ])
    def test_pattern_matches(self, text, expected):
        assert _extract_location_hint(text) == expected

    @pytest.mark.parametrize("text", [
        "",
        "영화 추천해줘",
        "오늘 박스오피스 1위 영화",
    ])
    def test_no_match_returns_none(self, text):
        assert _extract_location_hint(text) is None


# ============================================================
# 2. _format_tool_response
# ============================================================


class TestFormatToolResponse:
    def test_theater_with_results(self):
        results = {
            "theater_search": [
                {"name": "CGV 강남", "distance_m": 320},
                {"name": "메가박스 강남", "distance_m": 540},
            ],
            "kobis_now_showing": [
                {"rank": 1, "movie_nm": "테스트 영화"},
                {"rank": 2, "movie_nm": "또다른 영화"},
            ],
        }
        text = _format_tool_response("theater", results, location_address="강남역")
        assert "강남역" in text
        assert "2곳" in text
        assert "CGV 강남" in text
        assert "박스오피스" in text
        assert "테스트 영화" in text

    def test_theater_with_string_fallback(self):
        """도구가 안내 문자열을 직접 반환한 경우 그대로 노출."""
        results = {"theater_search": "영화관 검색이 잠시 안 돼요"}
        text = _format_tool_response("theater", results, location_address=None)
        assert "잠시 안 돼요" in text

    def test_info_with_detail(self):
        results = {
            "movie_detail": {
                "title": "기생충",
                "director": "봉준호",
                "runtime": 132,
                "overview": "한 가족의 이야기",
            },
            "ott_availability": ["Netflix", "Disney+"],
            "similar_movies": [{"title": "옥자"}, {"title": "마더"}],
        }
        text = _format_tool_response("info", results, location_address=None)
        assert "기생충" in text
        assert "봉준호" in text
        assert "Netflix" in text
        assert "옥자" in text

    def test_empty_results_returns_fallback(self):
        text = _format_tool_response("theater", {}, location_address=None)
        # 빈 도구 결과 → fallback 안내
        assert "정보를 가져오지 못" in text


# ============================================================
# 3. tool_executor_node — 통합 흐름
# ============================================================


def _state_with_intent(intent: str, **extra) -> dict:
    """ChatAgentState dict 생성 헬퍼."""
    return {
        "intent": IntentResult(intent=intent, confidence=0.9),
        "session_id": "test-session",
        "user_id": "",
        "current_input": "",
        **extra,
    }


class TestToolExecutorNode:
    @pytest.mark.asyncio
    async def test_unsupported_intent_returns_message(self):
        """recommend 의도는 본 노드 책임 밖 → 라우팅 안전망 메시지."""
        state = _state_with_intent("recommend")
        result = await tool_executor_node(state)
        assert "response" in result
        assert "tool_results" not in result

    @pytest.mark.asyncio
    async def test_theater_with_location_calls_execute_tool(self):
        """state.location 있으면 geocoding 호출 없이 execute_tool 만 호출."""
        location = Location(latitude=37.5, longitude=127.0, address="강남역")
        state = _state_with_intent(
            "theater",
            current_input="근처 영화관",
            location=location,
        )
        mock_geocoding = _make_mock_tool(return_value=None)
        with patch(
            "monglepick.agents.chat.nodes.execute_tool",
            new=AsyncMock(return_value={"theater_search": [{"name": "CGV 강남", "distance_m": 100}]}),
        ) as mock_exec, patch(
            "monglepick.agents.chat.nodes.geocoding",
            new=mock_geocoding,
        ):
            result = await tool_executor_node(state)

        mock_exec.assert_awaited_once()
        mock_geocoding.ainvoke.assert_not_awaited()  # location 이미 있음 → geocoding 호출 X
        assert "tool_results" in result
        assert "theater_search" in result["tool_results"]
        # 인자에 latitude/longitude 가 그대로 전달되었는지
        kwargs = mock_exec.await_args.kwargs
        assert kwargs["intent"] == "theater"
        assert kwargs["location"]["latitude"] == 37.5

    @pytest.mark.asyncio
    async def test_theater_without_location_uses_geocoding(self):
        """location 없음 + 메시지에 '강남역' 있음 → geocoding hit → execute_tool 호출."""
        state = _state_with_intent(
            "theater",
            current_input="강남역 근처 영화관 알려줘",
            location=None,
        )
        geo_result = {
            "latitude": 37.4979,
            "longitude": 127.0276,
            "address": "강남역",
            "place_name": "강남역 2호선",
            "source": "keyword",
        }
        mock_geocoding = _make_mock_tool(return_value=geo_result)
        with patch(
            "monglepick.agents.chat.nodes.geocoding",
            new=mock_geocoding,
        ), patch(
            "monglepick.agents.chat.nodes.execute_tool",
            new=AsyncMock(return_value={"theater_search": [{"name": "CGV 강남", "distance_m": 100}]}),
        ) as mock_exec:
            result = await tool_executor_node(state)

        mock_geocoding.ainvoke.assert_awaited_once_with({"query": "강남역"})
        mock_exec.assert_awaited_once()
        # state 에 location 도 함께 채워져 반환됨 (다음 턴 활용 가능)
        assert "location" in result
        assert result["location"].latitude == 37.4979

    @pytest.mark.asyncio
    async def test_theater_without_location_no_hint_asks_again(self):
        """location 없고 메시지에서도 지명을 못 뽑으면 위치 재질의."""
        state = _state_with_intent(
            "theater",
            current_input="영화관 알려줘",  # 지명 없음
            location=None,
        )
        with patch(
            "monglepick.agents.chat.nodes.execute_tool",
            new=AsyncMock(),
        ) as mock_exec:
            result = await tool_executor_node(state)

        mock_exec.assert_not_awaited()
        assert "지하철역" in result["response"] or "동네" in result["response"]
        assert "tool_results" not in result  # 도구 미호출 → state 미갱신

    @pytest.mark.asyncio
    async def test_exception_returns_fallback(self):
        """execute_tool 자체가 예외를 throw 해도 친절한 fallback 응답."""
        state = _state_with_intent(
            "theater",
            current_input="강남역",
            location=Location(latitude=37.5, longitude=127.0, address="강남역"),
        )
        with patch(
            "monglepick.agents.chat.nodes.execute_tool",
            new=AsyncMock(side_effect=RuntimeError("upstream boom")),
        ):
            result = await tool_executor_node(state)
        assert "response" in result
        # 에러를 전파하지 않고 빈 tool_results 와 안내 응답을 함께 반환
        assert result["tool_results"] == {}


# ============================================================
# 4. 멀티턴 보류 질문 (awaiting_location) 회귀 픽스 — 2026-05-07
# ============================================================


class TestPendingLocationFlow:
    """
    회귀 시나리오:
    Turn 1) 사용자 "근처영화관에서 최신 상영영화 찾아줘" → 지명 추출 실패 → 위치 재질의
            tool_executor_node 가 pending_question="awaiting_location" 을 set 해야 함
    Turn 2) 사용자 "강남역" → LLM 이 의도를 잘못(general/info) 분류해도 awaiting_location
            덕분에 강제 theater 흐름으로 진행 + short_fallback 으로 단일 토큰도 geocoding
    """

    @pytest.mark.asyncio
    async def test_location_required_sets_pending_question(self):
        """위치 재질의 시 pending_question='awaiting_location' 가 state 업데이트로 반환된다."""
        state = _state_with_intent(
            "theater",
            current_input="근처영화관에서 최신 상영영화 찾아줘",
            location=None,
        )
        result = await tool_executor_node(state)
        assert "어느 지역" in result["response"]
        assert result.get("pending_question") == "awaiting_location"
        # 도구 결과는 비어있어야 함 (호출되지 않음)
        assert "tool_results" not in result

    @pytest.mark.asyncio
    async def test_pending_location_overrides_misclassified_intent(self):
        """
        사용자가 "강남역" 만 보내서 LLM 이 general 로 잘못 분류했어도
        pending_question 이 set 돼 있으면 theater 로 강제 분기된다.
        """
        geo_result = {
            "latitude": 37.4979,
            "longitude": 127.0276,
            "address": "강남역",
            "place_name": "강남역",
            "source": "keyword",
        }
        # 의도는 일부러 general — pending_question 이 우선이어야 함
        state = _state_with_intent(
            "general",
            current_input="강남역",
            location=None,
            pending_question="awaiting_location",
        )
        with patch(
            "monglepick.agents.chat.nodes.geocoding",
            new=_make_mock_tool(return_value=geo_result),
        ) as mock_geo, patch(
            "monglepick.agents.chat.nodes.execute_tool",
            new=AsyncMock(return_value={"theater_search": [{"name": "CGV 강남", "distance_m": 100}]}),
        ) as mock_exec:
            result = await tool_executor_node(state)

        mock_geo.ainvoke.assert_awaited_once_with({"query": "강남역"})
        mock_exec.assert_awaited_once()
        kwargs = mock_exec.await_args.kwargs
        # 강제로 theater 로 실행됐는지 확인
        assert kwargs["intent"] == "theater"
        assert "tool_results" in result
        # 위치 해소 성공 시 보류 플래그는 명시적으로 비워져 다음 턴엔 다시 LLM 분류로 정상 복귀
        assert result.get("pending_question") is None

    @pytest.mark.asyncio
    async def test_short_fallback_for_single_token_location(self):
        """
        pending_question 컨텍스트에서 사용자가 정규식 패턴 어디에도 안 걸리는 단일 지명
        ("강남", "홍대", "이태원") 으로만 답해도 입력 자체를 카카오 키워드 검색에 던져
        위치 해소를 이어간다.
        """
        from monglepick.agents.chat.nodes import _extract_location_hint

        # 정규식 매칭 실패하지만 short_fallback 켜면 그대로 반환
        assert _extract_location_hint("강남", allow_short_fallback=True) == "강남"
        assert _extract_location_hint("홍대", allow_short_fallback=True) == "홍대"
        assert _extract_location_hint("이태원", allow_short_fallback=True) == "이태원"

        # short_fallback 꺼져 있으면 None (기존 동작 유지)
        assert _extract_location_hint("강남", allow_short_fallback=False) is None
        assert _extract_location_hint("홍대") is None  # default False

    @pytest.mark.asyncio
    async def test_short_fallback_rejects_long_or_garbage(self):
        """short_fallback 은 짧고 깔끔한 응답만 통과 — 긴 자유문/특수문자는 거부."""
        from monglepick.agents.chat.nodes import _extract_location_hint

        # 길이 초과
        assert (
            _extract_location_hint("이건너무긴자유문장이고지명이아니에요", allow_short_fallback=True)
            is None
        )
        # 줄바꿈/특수문자 다수
        assert _extract_location_hint("@#$%^&*", allow_short_fallback=True) is None
        # 한국어 자연문(긴) — 정규식 매칭도 실패하고 길이도 초과
        assert (
            _extract_location_hint("그냥아무거나추천해줘오늘기분이별로야", allow_short_fallback=True)
            is None
        )

    @pytest.mark.asyncio
    async def test_pending_location_uses_short_fallback_for_single_token(self):
        """
        end-to-end: pending_question="awaiting_location" + current_input="강남" (단일 지명, 역/근처/동·구 미포함)
        → short_fallback 활성 → geocoding({"query":"강남"}) 호출.
        """
        geo_result = {
            "latitude": 37.498,
            "longitude": 127.027,
            "address": "서울 강남구",
            "place_name": "강남",
            "source": "keyword",
        }
        state = _state_with_intent(
            "theater",
            current_input="강남",  # 정규식 패턴 미매칭 — short_fallback 만 잡을 수 있음
            location=None,
            pending_question="awaiting_location",
        )
        with patch(
            "monglepick.agents.chat.nodes.geocoding",
            new=_make_mock_tool(return_value=geo_result),
        ) as mock_geo, patch(
            "monglepick.agents.chat.nodes.execute_tool",
            new=AsyncMock(return_value={"theater_search": [{"name": "CGV 강남", "distance_m": 100}]}),
        ):
            result = await tool_executor_node(state)

        mock_geo.ainvoke.assert_awaited_once_with({"query": "강남"})
        assert "tool_results" in result
        assert result.get("pending_question") is None  # 성공 시 비워짐

    @pytest.mark.asyncio
    async def test_pending_location_geocoding_failure_keeps_flag(self):
        """
        pending_question 컨텍스트에서 단일 토큰을 받았지만 geocoding 도 실패하면
        pending_question 을 그대로 유지(다시 awaiting_location)하여 사용자가 또 다른 지명을
        보낼 수 있게 한다.
        """
        state = _state_with_intent(
            "theater",
            current_input="강남",
            location=None,
            pending_question="awaiting_location",
        )
        with patch(
            "monglepick.agents.chat.nodes.geocoding",
            new=_make_mock_tool(return_value=None),  # geocoding 실패
        ):
            result = await tool_executor_node(state)

        assert "어느 지역" in result["response"]
        assert result.get("pending_question") == "awaiting_location"  # 유지
