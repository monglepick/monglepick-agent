"""
geocoding 도구 단위 테스트 (Phase 6 외부 지도 연동).

대상: monglepick.tools.geocoding.geocoding (LangChain @tool)

핵심 계약:
 1) API 키 누락 → None 반환 (에러 전파 X)
 2) 빈/공백 입력 → 호출 전 None 반환
 3) 주소 검색 hit → source="address" 결과 반환 (키워드 fallback 호출 X)
 4) 주소 0건 → 키워드 검색 fallback → source="keyword"
 5) 두 경로 모두 0건 → None
 6) httpx.TimeoutException → None
 7) 카카오 응답의 x/y 가 누락되어도 0.0 으로 안전하게 파싱
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from monglepick.tools import geocoding


def _build_response(documents: list[dict]) -> MagicMock:
    """카카오 응답을 흉내내는 httpx.Response 모킹 헬퍼."""
    resp = MagicMock()
    resp.json.return_value = {"documents": documents}
    resp.raise_for_status = MagicMock()
    return resp


def _patch_kakao_key(value: str = "test-kakao-key"):
    """모듈 전역 _KAKAO_API_KEY 를 패치 — 도구 모듈은 import 시점 캡처라 직접 교체."""
    return patch("monglepick.tools.geocoding._KAKAO_API_KEY", value)


class TestGeocodingTool:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_none(self):
        """KAKAO_API_KEY 미설정 → 호출 자체를 안 하고 None."""
        with _patch_kakao_key(""):
            result = await geocoding.ainvoke({"query": "강남역"})
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_query_returns_none(self):
        """빈/공백 입력은 카카오를 두드리지 않고 None."""
        with _patch_kakao_key():
            result = await geocoding.ainvoke({"query": "   "})
        assert result is None

    @pytest.mark.asyncio
    async def test_address_hit_returns_address_source(self):
        """주소 검색 1건 hit → source='address' + road 주소 우선."""
        address_resp = _build_response([{
            "address_name": "서울 강남구 역삼동",
            "road_address": {"address_name": "서울 강남구 테헤란로 152"},
            "x": "127.0276",
            "y": "37.4979",
        }])

        with _patch_kakao_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=address_resp)

            with patch("monglepick.tools.geocoding.httpx.AsyncClient", return_value=mock_client):
                result = await geocoding.ainvoke({"query": "테헤란로 152"})

        assert result is not None
        assert result["source"] == "address"
        assert result["address"] == "서울 강남구 테헤란로 152"  # road 우선
        assert abs(result["latitude"] - 37.4979) < 1e-6
        assert abs(result["longitude"] - 127.0276) < 1e-6
        # 주소 검색 hit 시 키워드 검색은 호출되지 않음 (1번만 호출)
        assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_address_empty_falls_back_to_keyword(self):
        """주소 검색 0건 → 키워드 검색 fallback → source='keyword'."""
        empty_resp = _build_response([])
        keyword_resp = _build_response([{
            "place_name": "강남역 2호선",
            "address_name": "서울 강남구 역삼동 858",
            "road_address_name": "",
            "x": "127.0276",
            "y": "37.4979",
        }])

        with _patch_kakao_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            # 1차 주소 검색은 빈 응답, 2차 키워드 검색은 hit
            mock_client.get = AsyncMock(side_effect=[empty_resp, keyword_resp])

            with patch("monglepick.tools.geocoding.httpx.AsyncClient", return_value=mock_client):
                result = await geocoding.ainvoke({"query": "강남역"})

        assert result is not None
        assert result["source"] == "keyword"
        assert result["place_name"] == "강남역 2호선"
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_both_empty_returns_none(self):
        """주소/키워드 모두 0건 → None."""
        empty_resp = _build_response([])
        with _patch_kakao_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(side_effect=[empty_resp, empty_resp])
            with patch("monglepick.tools.geocoding.httpx.AsyncClient", return_value=mock_client):
                result = await geocoding.ainvoke({"query": "지구상에없는장소xyz"})
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        """httpx.TimeoutException → None (에러 전파 금지)."""
        with _patch_kakao_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("slow"))
            with patch("monglepick.tools.geocoding.httpx.AsyncClient", return_value=mock_client):
                result = await geocoding.ainvoke({"query": "강남역"})
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_xy_safe_parsing(self):
        """카카오 응답에 x/y 가 비어있어도 0.0 으로 파싱되어 KeyError/ValueError 안 터진다."""
        weird_resp = _build_response([{
            "address_name": "이상한 응답",
            "road_address": None,
            # x/y 누락
        }])
        with _patch_kakao_key():
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=weird_resp)
            with patch("monglepick.tools.geocoding.httpx.AsyncClient", return_value=mock_client):
                result = await geocoding.ainvoke({"query": "이상한곳"})
        # 주소 hit 로 처리되되 좌표는 0.0 (호출자가 falsy 검사로 사용 가능)
        assert result is not None
        assert result["latitude"] == 0.0
        assert result["longitude"] == 0.0
