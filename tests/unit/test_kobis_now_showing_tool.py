"""
kobis_now_showing 도구 단위 테스트 (Phase 6 외부 지도 연동).

대상: monglepick.tools.kobis_now_showing.kobis_now_showing (LangChain @tool)

핵심 계약:
 1) KOBIS_API_KEY 미설정 → 안내 문자열 반환
 2) 정상 응답 (boxOfficeResult.dailyBoxOfficeList) → top_n 슬라이싱 + 필드 정규화
 3) faultInfo (잘못된 키) → 안내 문자열
 4) dailyBoxOfficeList 빈 응답 → 안내 문자열
 5) httpx.TimeoutException → 안내 문자열
 6) top_n 범위 보정 (0 → 1, 999 → 10)
 7) 항목 단위 파싱 실패 시 해당 항목만 skip, 나머지는 살림
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from monglepick.tools import kobis_now_showing


def _build_response(daily_list: list[dict] | None = None, fault: dict | None = None) -> MagicMock:
    """KOBIS 응답 모킹 헬퍼."""
    payload: dict = {}
    if daily_list is not None:
        payload["boxOfficeResult"] = {"dailyBoxOfficeList": daily_list}
    if fault is not None:
        payload["faultInfo"] = fault
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def _patch_kobis_key(value: str = "test-kobis-key"):
    return patch("monglepick.tools.kobis_now_showing._KOBIS_API_KEY", value)


_SAMPLE_ITEM = {
    "rank": "1",
    "movieCd": "20240001",
    "movieNm": "테스트 영화",
    "audiAcc": "1234567",
    "openDt": "2026-04-01",
    "rankInten": "0",
    "rankOldAndNew": "OLD",
}


class TestKobisNowShowing:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_message(self):
        with _patch_kobis_key(""):
            result = await kobis_now_showing.ainvoke({"top_n": 5})
        assert isinstance(result, str)
        assert "현재 상영작" in result

    @pytest.mark.asyncio
    async def test_normal_response_parses_and_slices(self):
        """정상 응답에서 top_n 슬라이싱 + 필드 정규화."""
        items = [{**_SAMPLE_ITEM, "rank": str(i + 1), "movieCd": f"2024000{i + 1}"} for i in range(10)]
        resp = _build_response(daily_list=items)

        with _patch_kobis_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=resp)
            with patch("monglepick.tools.kobis_now_showing.httpx.AsyncClient", return_value=mock_client):
                result = await kobis_now_showing.ainvoke({"top_n": 3})

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["rank"] == 1
        assert result[0]["movie_cd"] == "20240001"
        assert result[0]["audi_acc"] == 1234567
        # openDt 의 하이픈은 제거되어야 한다
        assert result[0]["open_dt"] == "20260401"

    @pytest.mark.asyncio
    async def test_fault_info_returns_message(self):
        """KOBIS 가 200 으로 faultInfo 만 반환 (잘못된 키) → 안내 문자열."""
        resp = _build_response(fault={"errorCode": "320012", "message": "Invalid Key"})
        with _patch_kobis_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=resp)
            with patch("monglepick.tools.kobis_now_showing.httpx.AsyncClient", return_value=mock_client):
                result = await kobis_now_showing.ainvoke({"top_n": 5})
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_empty_list_returns_message(self):
        """dailyBoxOfficeList 가 빈 배열 (휴무 등) → 안내 문자열."""
        resp = _build_response(daily_list=[])
        with _patch_kobis_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=resp)
            with patch("monglepick.tools.kobis_now_showing.httpx.AsyncClient", return_value=mock_client):
                result = await kobis_now_showing.ainvoke({"top_n": 5})
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_timeout_returns_message(self):
        with _patch_kobis_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("slow"))
            with patch("monglepick.tools.kobis_now_showing.httpx.AsyncClient", return_value=mock_client):
                result = await kobis_now_showing.ainvoke({"top_n": 5})
        assert isinstance(result, str)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("requested,expected_len", [(0, 1), (1, 1), (5, 5), (10, 10), (999, 10)])
    async def test_top_n_clamped(self, requested, expected_len):
        """top_n 은 [1, 10] 범위로 클램프된다."""
        items = [{**_SAMPLE_ITEM, "rank": str(i + 1), "movieCd": f"X{i:04d}"} for i in range(10)]
        resp = _build_response(daily_list=items)
        with _patch_kobis_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=resp)
            with patch("monglepick.tools.kobis_now_showing.httpx.AsyncClient", return_value=mock_client):
                result = await kobis_now_showing.ainvoke({"top_n": requested})
        assert isinstance(result, list)
        assert len(result) == expected_len

    @pytest.mark.asyncio
    async def test_item_parse_failure_skipped(self):
        """한 항목 파싱 실패 (audiAcc='abc') 는 skip, 나머지는 살아남는다."""
        items = [
            {**_SAMPLE_ITEM, "rank": "1", "audiAcc": "abc"},  # 파싱 실패
            {**_SAMPLE_ITEM, "rank": "2", "movieCd": "OK002"},  # 정상
        ]
        resp = _build_response(daily_list=items)
        with _patch_kobis_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=resp)
            with patch("monglepick.tools.kobis_now_showing.httpx.AsyncClient", return_value=mock_client):
                result = await kobis_now_showing.ainvoke({"top_n": 5})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["movie_cd"] == "OK002"
