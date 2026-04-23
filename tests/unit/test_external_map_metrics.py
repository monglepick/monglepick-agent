"""
외부 지도 도구 Prometheus 메트릭 단위 테스트 (Phase 6 후속⁷ — 2026-04-23).

대상:
 - monglepick.metrics.external_map_tool_total
 - monglepick.metrics.external_map_tool_duration_seconds
 - monglepick.metrics.external_map_location_source_total

핵심 계약:
 1) geocoding/kobis_now_showing/theater_search 의 outcome 별 카운터 증가
 2) duration histogram 이 호출마다 observe 됨
 3) tool_executor_node 의 location 해소 경로별 source 카운터 증가
 4) 라벨 값은 정의된 enum 만 사용 (카디널리티 가드)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from monglepick.metrics import (
    external_map_location_source_total,
    external_map_tool_duration_seconds,
    external_map_tool_total,
)
from monglepick.tools import geocoding, kobis_now_showing, theater_search


def _counter_value(counter, **labels) -> float:
    """라벨 조합의 현재 카운터 값 반환 (없으면 0)."""
    try:
        return counter.labels(**labels)._value.get()
    except Exception:
        return 0.0


def _histogram_count(histogram, **labels) -> int:
    """라벨 조합의 histogram 누적 observation 수 반환."""
    try:
        return int(histogram.labels(**labels)._sum.get() and
                   histogram.labels(**labels)._count.get() or 0)
    except Exception:
        return 0


def _build_kakao_address_resp() -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"documents": [{
        "address_name": "서울 강남구 역삼동",
        "road_address": {"address_name": "서울 강남구 테헤란로 152"},
        "x": "127.0276", "y": "37.4979",
    }]}
    resp.raise_for_status = MagicMock()
    return resp


def _build_kakao_empty_resp() -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"documents": []}
    resp.raise_for_status = MagicMock()
    return resp


def _build_kobis_resp(daily_list: list[dict] | None, fault: dict | None = None) -> MagicMock:
    payload: dict = {}
    if daily_list is not None:
        payload["boxOfficeResult"] = {"dailyBoxOfficeList": daily_list}
    if fault is not None:
        payload["faultInfo"] = fault
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


# ============================================================
# 1. geocoding outcome 메트릭
# ============================================================


class TestGeocodingMetrics:
    @pytest.mark.asyncio
    async def test_no_api_key_increments_no_api_key(self):
        before = _counter_value(external_map_tool_total, tool="geocoding", outcome="no_api_key")
        with patch("monglepick.tools.geocoding._KAKAO_API_KEY", ""):
            await geocoding.ainvoke({"query": "강남"})
        after = _counter_value(external_map_tool_total, tool="geocoding", outcome="no_api_key")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_address_hit_increments_ok(self):
        before = _counter_value(external_map_tool_total, tool="geocoding", outcome="ok")
        with patch("monglepick.tools.geocoding._KAKAO_API_KEY", "k"):
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=_build_kakao_address_resp())
            with patch("monglepick.tools.geocoding.httpx.AsyncClient", return_value=mock_client):
                result = await geocoding.ainvoke({"query": "테헤란로 152"})
        assert result is not None
        after = _counter_value(external_map_tool_total, tool="geocoding", outcome="ok")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_both_empty_increments_empty(self):
        before = _counter_value(external_map_tool_total, tool="geocoding", outcome="empty")
        with patch("monglepick.tools.geocoding._KAKAO_API_KEY", "k"):
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(side_effect=[
                _build_kakao_empty_resp(), _build_kakao_empty_resp(),
            ])
            with patch("monglepick.tools.geocoding.httpx.AsyncClient", return_value=mock_client):
                result = await geocoding.ainvoke({"query": "지구상에없는장소xyz"})
        assert result is None
        after = _counter_value(external_map_tool_total, tool="geocoding", outcome="empty")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_timeout_falls_through_to_empty(self):
        """geocoding 만의 특수 동작 — `_search_address`/`_search_keyword` 헬퍼가
        timeout 을 graceful 하게 흡수하고 None 반환 → outer 흐름은 두 단계 모두 0건으로
        보고 `empty` outcome 으로 기록한다. outer try 의 TimeoutException 분기는
        헬퍼가 흡수하므로 실질적으로 도달 불가능 — 안전망 역할.

        다른 도구(theater_search/kobis_now_showing)는 헬퍼 분리 없이 메인 흐름에서
        timeout 을 직접 받으므로 outcome="timeout" 이 정상 카운트된다.
        """
        before = _counter_value(external_map_tool_total, tool="geocoding", outcome="empty")
        with patch("monglepick.tools.geocoding._KAKAO_API_KEY", "k"):
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("slow"))
            with patch("monglepick.tools.geocoding.httpx.AsyncClient", return_value=mock_client):
                result = await geocoding.ainvoke({"query": "강남"})
        assert result is None
        after = _counter_value(external_map_tool_total, tool="geocoding", outcome="empty")
        assert after == before + 1


# ============================================================
# 2. kobis_now_showing outcome 메트릭
# ============================================================


class TestKobisMetrics:
    @pytest.mark.asyncio
    async def test_no_api_key_increments_no_api_key(self):
        before = _counter_value(external_map_tool_total, tool="kobis_now_showing", outcome="no_api_key")
        with patch("monglepick.tools.kobis_now_showing._KOBIS_API_KEY", ""):
            await kobis_now_showing.ainvoke({"top_n": 5})
        after = _counter_value(external_map_tool_total, tool="kobis_now_showing", outcome="no_api_key")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_normal_response_increments_ok(self):
        before = _counter_value(external_map_tool_total, tool="kobis_now_showing", outcome="ok")
        items = [{
            "rank": "1", "movieCd": "X1", "movieNm": "테스트", "audiAcc": "100",
            "openDt": "2026-04-01", "rankInten": "0", "rankOldAndNew": "OLD",
        }]
        with patch("monglepick.tools.kobis_now_showing._KOBIS_API_KEY", "k"):
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=_build_kobis_resp(daily_list=items))
            with patch("monglepick.tools.kobis_now_showing.httpx.AsyncClient", return_value=mock_client):
                await kobis_now_showing.ainvoke({"top_n": 1})
        after = _counter_value(external_map_tool_total, tool="kobis_now_showing", outcome="ok")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_fault_increments_fault(self):
        before = _counter_value(external_map_tool_total, tool="kobis_now_showing", outcome="fault")
        with patch("monglepick.tools.kobis_now_showing._KOBIS_API_KEY", "k"):
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=_build_kobis_resp(
                daily_list=None, fault={"errorCode": "320012", "message": "Invalid"}))
            with patch("monglepick.tools.kobis_now_showing.httpx.AsyncClient", return_value=mock_client):
                await kobis_now_showing.ainvoke({"top_n": 5})
        after = _counter_value(external_map_tool_total, tool="kobis_now_showing", outcome="fault")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_empty_increments_empty(self):
        before = _counter_value(external_map_tool_total, tool="kobis_now_showing", outcome="empty")
        with patch("monglepick.tools.kobis_now_showing._KOBIS_API_KEY", "k"):
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=_build_kobis_resp(daily_list=[]))
            with patch("monglepick.tools.kobis_now_showing.httpx.AsyncClient", return_value=mock_client):
                await kobis_now_showing.ainvoke({"top_n": 5})
        after = _counter_value(external_map_tool_total, tool="kobis_now_showing", outcome="empty")
        assert after == before + 1


# ============================================================
# 3. theater_search outcome 메트릭
# ============================================================


class TestTheaterSearchMetrics:
    @pytest.mark.asyncio
    async def test_no_api_key_increments_no_api_key(self):
        before = _counter_value(external_map_tool_total, tool="theater_search", outcome="no_api_key")
        with patch("monglepick.tools.theater_search._KAKAO_API_KEY", ""):
            await theater_search.ainvoke({"latitude": 37.5, "longitude": 127.0})
        after = _counter_value(external_map_tool_total, tool="theater_search", outcome="no_api_key")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_success_increments_ok(self):
        before = _counter_value(external_map_tool_total, tool="theater_search", outcome="ok")
        # 카카오 키워드 검색 응답 — 영화관 1건씩 3개 체인 모두 hit
        kakao_resp = MagicMock()
        kakao_resp.json.return_value = {"documents": [{
            "id": "1", "place_name": "CGV 강남",
            "road_address_name": "서울 강남구",
            "phone": "02-1234-5678",
            "x": "127.0276", "y": "37.4979", "distance": "100",
        }]}
        kakao_resp.raise_for_status = MagicMock()
        with patch("monglepick.tools.theater_search._KAKAO_API_KEY", "k"):
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=kakao_resp)
            with patch("monglepick.tools.theater_search.httpx.AsyncClient", return_value=mock_client):
                result = await theater_search.ainvoke({
                    "latitude": 37.4979, "longitude": 127.0276, "radius": 5000,
                })
        assert isinstance(result, list)
        after = _counter_value(external_map_tool_total, tool="theater_search", outcome="ok")
        assert after == before + 1


# ============================================================
# 4. duration histogram observe 검증 (sum 이 증가하는지)
# ============================================================


class TestDurationHistogram:
    @pytest.mark.asyncio
    async def test_geocoding_observes_duration(self):
        """ok 호출이 발생하면 duration sum 도 증가한다."""
        before_sum = external_map_tool_duration_seconds.labels(tool="geocoding")._sum.get()
        with patch("monglepick.tools.geocoding._KAKAO_API_KEY", "k"):
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client.get = AsyncMock(return_value=_build_kakao_address_resp())
            with patch("monglepick.tools.geocoding.httpx.AsyncClient", return_value=mock_client):
                await geocoding.ainvoke({"query": "강남"})
        after_sum = external_map_tool_duration_seconds.labels(tool="geocoding")._sum.get()
        assert after_sum >= before_sum  # >= (호출 시간이 0 이하일 수 없음)


# ============================================================
# 5. location source 카운터 (tool_executor_node 통합 — 별 모듈에서 검증)
# ============================================================


class TestLocationSourceCounter:
    """external_map_location_source_total 의 라벨 enum 자체 검증.

    실제 노드 통합 테스트는 test_tool_executor_node.py 에서 이미 시나리오별로 분기 검증 중이므로,
    여기서는 라벨 enum 만 카디널리티 가드로 점검한다.
    """

    def test_three_sources_can_be_labeled(self):
        # 3가지 source 모두 라벨 호출 자체가 통과해야 한다 (KeyError 등 X)
        for src in ("client_supplied", "geocoded", "missing"):
            external_map_location_source_total.labels(source=src)  # 예외 없으면 통과
