"""
카카오맵 API 기반 근처 영화관 검색 도구 (Phase 6 Tool 3).

카카오 로컬 API(키워드 검색)를 사용해 사용자 위치 기반 영화관을 검색한다.
theater 의도 처리 시 tool_executor_node에서 호출된다.

카카오 로컬 API 문서:
- 키워드로 장소 검색: GET https://dapi.kakao.com/v2/local/search/keyword.json
- Authorization: KakaoAK {REST_API_KEY}
"""

from __future__ import annotations

import asyncio
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

_TOOL_NAME = "theater_search"

# 카카오 로컬 API 설정
_KAKAO_API_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
_KAKAO_API_KEY = settings.KAKAO_API_KEY   # .env에서 로드
_KAKAO_TIMEOUT_SEC = 5.0                  # 카카오 API 응답 타임아웃 (초)

# 영화관 체인명 키워드 목록 — 카카오 키워드 검색에 순차적으로 시도
# 카카오 API는 단일 키워드만 지원하므로 여러 번 호출 후 병합한다
_THEATER_KEYWORDS = ["CGV", "롯데시네마", "메가박스"]

# 검색 반경 최대값 (미터) — 카카오 API 최대 20,000m
_MAX_RADIUS_M = 20_000

# 영화관 체인 식별 + 모바일 예매 페이지 검색 딥링크 패턴
# 영화관별 시간표 API 가 공식 공개되지 않아 체인 검색 페이지로 점프시키고
# 사용자가 영화관명으로 다시 한 번 선택하도록 유도한다 (스크래핑 회피).
# 키 = 식별 토큰(소문자), 값 = (chain_label, booking_url_template)
# booking_url_template 의 {q} 자리에 영화관명을 URL 인코딩하여 삽입한다.
_CHAIN_PATTERNS: list[tuple[str, str, str]] = [
    # (식별 토큰, chain 라벨, booking URL 템플릿)
    ("cgv",     "CGV",      "https://m.cgv.co.kr/cgvsearch/Search.aspx?query={q}"),
    ("롯데",    "롯데시네마", "https://www.lottecinema.co.kr/NLCHS/Cinema/Detail?cinemaQuery={q}"),
    ("메가박스", "메가박스",  "https://www.megabox.co.kr/theater?searchKeyword={q}"),
]


def _detect_chain_and_booking_url(name: str) -> tuple[str, str]:
    """
    영화관명에서 체인을 식별하고 해당 체인의 모바일 검색 딥링크를 생성한다.

    예: "CGV 강남" → ("CGV", "https://m.cgv.co.kr/cgvsearch/Search.aspx?query=CGV%20%EA%B0%95%EB%82%A8")

    매칭 실패 시 ("기타", "") 반환 — 비체인 상영관(예: 독립영화관) 대응.

    Args:
        name: 카카오 응답의 place_name (예: "CGV 강남", "롯데시네마 월드타워")

    Returns:
        (chain_label, booking_url) — 매칭 실패 시 ("기타", "")
    """
    # urllib import 는 모듈 상단 의존을 줄이기 위해 함수 레벨에서 지연 로드
    from urllib.parse import quote

    lowered = name.lower()
    for token, chain_label, url_template in _CHAIN_PATTERNS:
        if token in lowered:
            return chain_label, url_template.format(q=quote(name))
    return "기타", ""


@tool
async def theater_search(
    latitude: float,
    longitude: float,
    radius: int = 5000,
) -> list[dict] | str:
    """
    카카오맵 API로 사용자 위치 기반 근처 영화관을 검색한다.

    CGV, 롯데시네마, 메가박스 3대 체인을 동시 검색하여 거리순으로 정렬한다.

    Args:
        latitude: 사용자 위도 (예: 37.5665)
        longitude: 사용자 경도 (예: 126.9780)
        radius: 검색 반경 (미터, 기본 5,000 / 최대 20,000)

    Returns:
        성공 시 영화관 정보 dict 목록 (거리 오름차순):
        [
            {
                "theater_id": str,   # 카카오 장소 ID
                "name": str,         # 영화관명 (예: "CGV 강남")
                "address": str,      # 도로명 주소
                "phone": str,        # 전화번호
                "latitude": float,   # 위도
                "longitude": float,  # 경도
                "distance_m": int,   # 검색 위치로부터 거리 (미터)
                "place_url": str,    # 카카오맵 상세 URL
            }
        ]
        API 키 누락 또는 에러 시: "영화관 검색이 잠시 안 돼요" 문자열 반환
    """
    # API 키 누락 시 조기 반환
    if not _KAKAO_API_KEY:
        logger.warning("theater_search_tool_no_api_key")
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="no_api_key").inc()
        return "영화관 검색이 잠시 안 돼요"

    # 반경 범위 보정 (최소 100m, 최대 20,000m)
    safe_radius = max(100, min(int(radius), _MAX_RADIUS_M))
    started = time.perf_counter()

    try:
        # 공통 요청 헤더 (카카오 REST API 인증)
        headers = {"Authorization": f"KakaoAK {_KAKAO_API_KEY}"}

        async with httpx.AsyncClient(timeout=_KAKAO_TIMEOUT_SEC) as client:
            # 3개 체인 키워드 병렬 검색 — asyncio.gather로 동시 요청
            tasks = [
                _search_keyword(client, headers, keyword, latitude, longitude, safe_radius)
                for keyword in _THEATER_KEYWORDS
            ]
            results_per_keyword = await asyncio.gather(*tasks, return_exceptions=True)

        # 체인별 결과 병합 + 중복 제거 (theater_id 기준)
        seen_ids: set[str] = set()
        merged: list[dict] = []

        for result in results_per_keyword:
            # gather 중 예외 발생 시 해당 체인 결과 건너뜀
            if isinstance(result, Exception):
                logger.warning(
                    "theater_search_keyword_error",
                    error=str(result),
                )
                continue
            for theater in result:
                tid = theater.get("theater_id", "")
                if tid and tid not in seen_ids:
                    seen_ids.add(tid)
                    merged.append(theater)

        # 거리 오름차순 정렬
        merged.sort(key=lambda x: x.get("distance_m", 999_999))

        logger.info(
            "theater_search_tool_done",
            latitude=latitude,
            longitude=longitude,
            radius_m=safe_radius,
            result_count=len(merged),
            top_names=[t.get("name", "") for t in merged[:3]],
        )
        # 결과 0건은 "정상 응답이지만 매칭 0건" 으로 empty 라벨, 아니면 ok.
        outcome = "ok" if merged else "empty"
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome=outcome).inc()
        external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
            time.perf_counter() - started
        )
        return merged

    except httpx.TimeoutException:
        logger.error(
            "theater_search_tool_timeout",
            latitude=latitude,
            longitude=longitude,
            timeout_sec=_KAKAO_TIMEOUT_SEC,
        )
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="timeout").inc()
        external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
            time.perf_counter() - started
        )
        return "영화관 검색이 잠시 안 돼요"

    except Exception as e:
        # 예상치 못한 에러 (에러 전파 금지)
        logger.error(
            "theater_search_tool_error",
            error=str(e),
            error_type=type(e).__name__,
            latitude=latitude,
            longitude=longitude,
        )
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="exception").inc()
        external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
            time.perf_counter() - started
        )
        return "영화관 검색이 잠시 안 돼요"


async def _search_keyword(
    client: httpx.AsyncClient,
    headers: dict,
    keyword: str,
    latitude: float,
    longitude: float,
    radius: int,
) -> list[dict]:
    """
    카카오 로컬 키워드 검색 API 단일 호출 헬퍼.

    하나의 영화관 체인 키워드(CGV 등)로 검색 후 파싱 결과를 반환한다.
    에러 시 빈 리스트를 반환하여 gather 전체를 깨뜨리지 않는다.

    Args:
        client: 재사용 httpx.AsyncClient 인스턴스
        headers: Authorization 헤더
        keyword: 검색 키워드 (예: "CGV")
        latitude: 위도 (y 파라미터)
        longitude: 경도 (x 파라미터)
        radius: 검색 반경 (미터)

    Returns:
        영화관 dict 목록 (파싱 완료)
    """
    try:
        resp = await client.get(
            _KAKAO_API_URL,
            headers=headers,
            params={
                "query": keyword,
                "x": str(longitude),       # 경도 (카카오: x=경도)
                "y": str(latitude),        # 위도 (카카오: y=위도)
                "radius": str(radius),
                "sort": "distance",        # 거리 오름차순 정렬
                "size": 5,                 # 체인당 최대 5개
                "category_group_code": "CT1",  # CT1: 문화시설 (영화관 포함)
            },
        )
        resp.raise_for_status()
        data = resp.json()

        theaters: list[dict] = []
        for doc in data.get("documents", []):
            # 카카오 응답 필드: id, place_name, road_address_name, phone, x(경도), y(위도), distance
            place_name = doc.get("place_name", "")
            chain_label, booking_url = _detect_chain_and_booking_url(place_name)
            theaters.append({
                "theater_id": doc.get("id", ""),
                "name": place_name,
                "chain": chain_label,                # "CGV" / "롯데시네마" / "메가박스" / "기타"
                "address": doc.get("road_address_name") or doc.get("address_name", ""),
                "phone": doc.get("phone", ""),
                "latitude": float(doc.get("y", 0)),
                "longitude": float(doc.get("x", 0)),
                # distance: 카카오 API가 문자열로 반환 ("1234")
                "distance_m": int(doc.get("distance", 0) or 0),
                "place_url": doc.get("place_url", ""),
                # 체인 모바일 사이트 검색 딥링크 — 영화관별 시간표 API 가 없어
                # 사용자가 한 번 더 영화관을 클릭해야 하는 한계는 있지만 스크래핑 회피.
                "booking_url": booking_url,
            })
        return theaters

    except Exception as e:
        logger.warning(
            "theater_search_keyword_request_error",
            keyword=keyword,
            error=str(e),
        )
        return []
