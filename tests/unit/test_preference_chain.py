"""
선호 추출 체인 단위 테스트 (Task 7).

테스트 대상:
- Mock LLM: "SF 영화 추천해줘" → genre_preference="SF"
- 병합: 새 값이 이전 값 덮어쓰기
- 병합: None은 이전 값 유지
- reference_movies 합집합 (중복 제거)
- 에러 → 이전 선호 반환
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import ExtractedPreferences
from monglepick.chains.preference_chain import (
    _format_existing_preferences,
    extract_preferences,
)


@pytest.mark.asyncio
async def test_extract_genre_preference(mock_ollama):
    """SF 영화 추천 → genre_preference='SF'가 추출된다."""
    mock_ollama.set_structured_response(
        ExtractedPreferences(genre_preference="SF"),
    )
    result = await extract_preferences("SF 영화 추천해줘")
    assert result.genre_preference == "SF"


@pytest.mark.asyncio
async def test_merge_overrides_prev(mock_ollama):
    """새 값이 이전 값을 덮어쓴다."""
    prev = ExtractedPreferences(genre_preference="액션", mood="유쾌")
    mock_ollama.set_structured_response(
        ExtractedPreferences(genre_preference="SF"),
    )
    result = await extract_preferences("SF로 바꿔줘", previous_preferences=prev)
    # Phase ML-3: genre는 합집합 누적 ("액션" + "SF" = "액션, SF")
    assert result.genre_preference == "액션, SF"
    # mood는 LLM이 None 반환 → 이전 값 유지
    assert result.mood == "유쾌"


@pytest.mark.asyncio
async def test_merge_preserves_prev_on_none(mock_ollama):
    """LLM이 모두 None 반환 → 이전 값 유지."""
    prev = ExtractedPreferences(platform="넷플릭스", era="2020년대")
    mock_ollama.set_structured_response(ExtractedPreferences())
    result = await extract_preferences("음...", previous_preferences=prev)
    assert result.platform == "넷플릭스"
    assert result.era == "2020년대"


@pytest.mark.asyncio
async def test_merge_reference_movies_union(mock_ollama):
    """reference_movies는 합집합 (중복 제거)."""
    prev = ExtractedPreferences(reference_movies=["인셉션"])
    mock_ollama.set_structured_response(
        ExtractedPreferences(reference_movies=["인셉션", "테넷"]),
    )
    result = await extract_preferences(
        "테넷도 좋았어",
        previous_preferences=prev,
    )
    assert "인셉션" in result.reference_movies
    assert "테넷" in result.reference_movies
    # 중복 제거 확인
    assert len(result.reference_movies) == 2


@pytest.mark.asyncio
async def test_error_returns_previous(mock_ollama):
    """LLM 에러 → 이전 선호 그대로 반환."""
    prev = ExtractedPreferences(genre_preference="코미디")
    mock_ollama.set_error(RuntimeError("LLM error"))
    result = await extract_preferences("아무거나", previous_preferences=prev)
    assert result.genre_preference == "코미디"


@pytest.mark.asyncio
async def test_error_returns_empty_on_first_turn(mock_ollama):
    """첫 턴에서 LLM 에러 → 빈 ExtractedPreferences."""
    mock_ollama.set_error(RuntimeError("LLM error"))
    result = await extract_preferences("영화 추천해줘")
    assert result.genre_preference is None
    assert result.reference_movies == []


class TestFormatExistingPreferences:
    """_format_existing_preferences 유틸 함수 테스트."""

    def test_none_input(self):
        """None → '(아직 파악된 선호 조건 없음)'."""
        result = _format_existing_preferences(None)
        assert "없음" in result

    def test_empty_preferences(self):
        """빈 ExtractedPreferences → '(아직 파악된 선호 조건 없음)'."""
        result = _format_existing_preferences(ExtractedPreferences())
        assert "없음" in result

    def test_filled_preferences(self):
        """채워진 필드가 포맷된다."""
        prefs = ExtractedPreferences(
            genre_preference="SF",
            platform="넷플릭스",
        )
        result = _format_existing_preferences(prefs)
        assert "SF" in result
        assert "넷플릭스" in result
