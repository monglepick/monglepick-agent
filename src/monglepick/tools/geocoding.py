"""
카카오 Local API 기반 지오코딩 도구 (Phase 6 Tool 9 — 외부 지도 연동).

사용자가 "강남역", "홍대입구역 근처", "서울시 강남구 테헤란로 152" 처럼
주소/지역명만 던졌을 때 좌표(위도/경도)를 얻어 theater_search 등의 위치 기반
도구로 위임할 수 있도록 한다. tool_executor_node 가 location 미제공 + 메시지에
지명이 포함된 경우 우선 호출한다.

카카오 Local REST API:
- 주소 검색:    GET https://dapi.kakao.com/v2/local/search/address.json?query=
- 키워드 검색:  GET https://dapi.kakao.com/v2/local/search/keyword.json?query=

전략:
1) 주소 검색을 우선 시도 (도로명 / 지번 주소가 정확히 매칭될 때 가장 신뢰).
2) 0건이면 키워드 검색으로 fallback (지하철역명·지명·랜드마크 등).
3) 두 경로 모두 0건이거나 에러면 에러 전파 금지하고 None 반환.
"""

from __future__ import annotations

import time

import httpx
import structlog
from langchain_core.tools import tool

from monglepick.config import settings
from monglepick.metrics import (
    external_map_tool_duration_seconds,
    external_map_tool_total,
)

logger = structlog.get_logger()

_TOOL_NAME = "geocoding"

# 카카오 Local API 설정
_KAKAO_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
_KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
_KAKAO_API_KEY = settings.KAKAO_API_KEY  # .env 에서 로드 (REST API 키)
_KAKAO_TIMEOUT_SEC = 5.0                 # 카카오 API 응답 타임아웃 (초)


@tool
async def geocoding(query: str) -> dict | None:
    """
    카카오 Local API 로 주소/지역명을 좌표(위도, 경도) 로 변환한다.

    "강남역", "홍대입구역 근처", "서울시 강남구 테헤란로 152" 등 자유 형식 지명을
    위경도 좌표로 매핑한다. 이후 theater_search 같은 위치 기반 도구의 입력으로 사용.

    Args:
        query: 자유 형식 지역명 / 주소 (예: "강남역", "홍대 입구")

    Returns:
        성공 시:
        {
            "query": str,        # 원본 입력
            "address": str,      # 정규화된 주소 (도로명 우선, 없으면 지번)
            "place_name": str,   # 매칭된 장소명 (키워드 검색일 때만 채워짐)
            "latitude": float,   # 위도
            "longitude": float,  # 경도
            "source": str,       # "address" | "keyword" — 어느 경로로 매칭됐는지
        }
        매칭 실패 / API 키 누락 / 에러 시: None
        (호출자는 None 일 때 사용자에게 위치 재질의를 띄우면 된다)
    """
    # API 키 누락 시 조기 반환 (에러 전파 X)
    if not _KAKAO_API_KEY:
        logger.warning("geocoding_tool_no_api_key")
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="no_api_key").inc()
        return None

    # 빈/공백 입력은 호출조차 하지 않는다 (메트릭 기록도 스킵 — 도구 호출이 아님)
    safe_query = (query or "").strip()
    if not safe_query:
        return None

    headers = {"Authorization": f"KakaoAK {_KAKAO_API_KEY}"}
    started = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=_KAKAO_TIMEOUT_SEC) as client:
            # 1) 주소 검색 우선 — 도로명/지번 주소면 가장 정확
            address_hit = await _search_address(client, headers, safe_query)
            if address_hit:
                logger.info(
                    "geocoding_tool_address_hit",
                    query=safe_query,
                    address=address_hit.get("address"),
                )
                external_map_tool_total.labels(tool=_TOOL_NAME, outcome="ok").inc()
                external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
                    time.perf_counter() - started
                )
                return address_hit

            # 2) 키워드 검색으로 fallback — 지하철역·지명·랜드마크 등
            keyword_hit = await _search_keyword(client, headers, safe_query)
            if keyword_hit:
                logger.info(
                    "geocoding_tool_keyword_hit",
                    query=safe_query,
                    place_name=keyword_hit.get("place_name"),
                )
                external_map_tool_total.labels(tool=_TOOL_NAME, outcome="ok").inc()
                external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
                    time.perf_counter() - started
                )
                return keyword_hit

        # 두 경로 모두 0건
        logger.info("geocoding_tool_no_match", query=safe_query)
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="empty").inc()
        external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
            time.perf_counter() - started
        )
        return None

    except httpx.TimeoutException:
        logger.error(
            "geocoding_tool_timeout",
            query=safe_query,
            timeout_sec=_KAKAO_TIMEOUT_SEC,
        )
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="timeout").inc()
        external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
            time.perf_counter() - started
        )
        return None

    except Exception as e:
        # 에러 전파 금지 — 호출 측은 None 만 보고 위치 재질의로 분기
        logger.error(
            "geocoding_tool_error",
            query=safe_query,
            error=str(e),
            error_type=type(e).__name__,
        )
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="exception").inc()
        external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
            time.perf_counter() - started
        )
        return None


async def _search_address(
    client: httpx.AsyncClient,
    headers: dict,
    query: str,
) -> dict | None:
    """카카오 주소 검색 헬퍼. 첫 매칭만 반환 (가장 신뢰도 높은 후보)."""
    try:
        resp = await client.get(
            _KAKAO_ADDRESS_URL,
            headers=headers,
            params={"query": query, "size": 1},
        )
        resp.raise_for_status()
        documents = resp.json().get("documents", [])
        if not documents:
            return None

        doc = documents[0]
        # 카카오 주소 검색 응답: x=경도, y=위도 (모두 문자열)
        # address_name 은 지번 주소, road_address.address_name 은 도로명 주소
        road_addr = (doc.get("road_address") or {}).get("address_name", "")
        jibun_addr = doc.get("address_name", "")
        return {
            "query": query,
            "address": road_addr or jibun_addr,
            "place_name": "",  # 주소 검색은 장소명이 비어있다
            "latitude": float(doc.get("y", 0) or 0),
            "longitude": float(doc.get("x", 0) or 0),
            "source": "address",
        }
    except Exception as e:
        # 단계별 실패는 상위 _search 흐름에서 keyword fallback 으로 자연스럽게 진행되므로
        # 여기서는 warning 으로만 남기고 None 반환.
        logger.warning("geocoding_address_request_error", query=query, error=str(e))
        return None


async def _search_keyword(
    client: httpx.AsyncClient,
    headers: dict,
    query: str,
) -> dict | None:
    """카카오 키워드 검색 헬퍼. 지하철역/지명/랜드마크 fallback 용."""
    try:
        resp = await client.get(
            _KAKAO_KEYWORD_URL,
            headers=headers,
            params={"query": query, "size": 1},
        )
        resp.raise_for_status()
        documents = resp.json().get("documents", [])
        if not documents:
            return None

        doc = documents[0]
        # 키워드 검색 응답: place_name (장소명), road_address_name / address_name
        return {
            "query": query,
            "address": doc.get("road_address_name") or doc.get("address_name", ""),
            "place_name": doc.get("place_name", ""),
            "latitude": float(doc.get("y", 0) or 0),
            "longitude": float(doc.get("x", 0) or 0),
            "source": "keyword",
        }
    except Exception as e:
        logger.warning("geocoding_keyword_request_error", query=query, error=str(e))
        return None
