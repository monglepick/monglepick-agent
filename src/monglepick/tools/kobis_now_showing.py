"""
KOBIS 일별 박스오피스 기반 "현재 상영중 영화" 조회 도구 (Phase 6 Tool 10 — 외부 지도 연동).

KOBIS Open API 의 일별 박스오피스 Top-10 을 "지금 상영중 영화" 의 권위 데이터로 사용한다.
영화관(theater_search) 결과와 결합해 "근처 영화관 + 지금 박스오피스" 카드 묶음을
사용자에게 보여준다.

전제:
- KOBIS 는 영화관 마스터 / 영화관별 시간표 API 를 공개하지 않는다.
  → 영화관 정보는 카카오 Local 로, 영화 콘텐츠는 KOBIS 로 역할 분담.
- 일별 박스오피스는 D-1 기준 집계 (당일 데이터는 미집계). 내부적으로 어제 날짜 사용.

KOBIS Open API:
- 일별 박스오피스: GET {KOBIS_BASE_URL}/boxoffice/searchDailyBoxOfficeList.json
  필수: key, targetDt (YYYYMMDD)
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import httpx
import structlog
from langchain_core.tools import tool

from monglepick.config import settings
from monglepick.metrics import (
    external_map_tool_duration_seconds,
    external_map_tool_total,
)

logger = structlog.get_logger()

_TOOL_NAME = "kobis_now_showing"

# KOBIS API 설정
_KOBIS_BASE_URL = settings.KOBIS_BASE_URL  # http://www.kobis.or.kr/kobisopenapi/webservice/rest
_KOBIS_API_KEY = settings.KOBIS_API_KEY    # .env 에서 로드
_KOBIS_TIMEOUT_SEC = 5.0                   # KOBIS 응답 타임아웃 (초)


@tool
async def kobis_now_showing(top_n: int = 10) -> list[dict] | str:
    """
    KOBIS 일별 박스오피스 Top-N 으로 "현재 상영중 영화" 목록을 조회한다.

    영화관별 정확한 시간표는 KOBIS 가 제공하지 않으므로, 박스오피스 상위 영화를
    "이 영화관에서도 상영 중일 가능성이 높은 영화" 로 간주하여 안내한다.
    영화관 카드와 함께 표시하면 "근처 영화관 N곳 / 지금 인기 영화 Top-N" UX 구성.

    Args:
        top_n: 반환할 영화 수 (기본 10, 최대 10 — KOBIS 응답이 Top-10 고정)

    Returns:
        성공 시 영화 dict 목록 (rank 오름차순):
        [
            {
                "rank": int,             # 박스오피스 순위 (1~10)
                "movie_cd": str,         # KOBIS 영화 코드 (8자리)
                "movie_nm": str,         # 영화명 (한국어)
                "audi_acc": int,         # 누적 관객수
                "open_dt": str,          # 개봉일 (YYYYMMDD)
                "rank_inten": int,       # 전일 대비 순위 변동 (+/-)
                "rank_old_and_new": str, # "NEW"=신규진입 / "OLD"=기존
            }
        ]
        API 키 누락 / 빈 응답 / 에러 시: "현재 상영작 정보를 잠시 불러올 수 없어요" 문자열
    """
    # API 키 누락 시 조기 반환
    if not _KOBIS_API_KEY:
        logger.warning("kobis_now_showing_tool_no_api_key")
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="no_api_key").inc()
        return "현재 상영작 정보를 잠시 불러올 수 없어요"

    # top_n 범위 보정 (1~10) — KOBIS 일별 박스오피스 응답 자체가 Top-10 고정
    safe_top_n = max(1, min(int(top_n), 10))

    # 어제 날짜를 targetDt 로 사용 — 당일은 집계 전이라 빈 응답이 나올 수 있다
    target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    started = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=_KOBIS_TIMEOUT_SEC) as client:
            resp = await client.get(
                f"{_KOBIS_BASE_URL}/boxoffice/searchDailyBoxOfficeList.json",
                params={"key": _KOBIS_API_KEY, "targetDt": target_date},
            )
            resp.raise_for_status()
            data = resp.json()

        # KOBIS 응답 구조: { "boxOfficeResult": { "dailyBoxOfficeList": [ ... ] } }
        # 키 누락(잘못된 키) 시 KOBIS 는 200 에 faultInfo 만 반환 → boxOfficeResult 없음
        result = data.get("boxOfficeResult")
        if not result:
            fault = data.get("faultInfo") or {}
            logger.warning(
                "kobis_now_showing_tool_fault",
                target_date=target_date,
                fault_message=fault.get("message", "unknown"),
            )
            external_map_tool_total.labels(tool=_TOOL_NAME, outcome="fault").inc()
            external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
                time.perf_counter() - started
            )
            return "현재 상영작 정보를 잠시 불러올 수 없어요"

        raw_list = result.get("dailyBoxOfficeList", [])
        if not raw_list:
            # 박스오피스 데이터가 비어있을 때 — 신년 등 휴무 기간에 가끔 발생
            logger.info("kobis_now_showing_tool_empty", target_date=target_date)
            external_map_tool_total.labels(tool=_TOOL_NAME, outcome="empty").inc()
            external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
                time.perf_counter() - started
            )
            return "현재 상영작 정보를 잠시 불러올 수 없어요"

        # 응답 정규화 + Top-N 슬라이싱
        movies: list[dict] = []
        for item in raw_list[:safe_top_n]:
            try:
                movies.append({
                    "rank": int(item.get("rank", 0)),
                    "movie_cd": item.get("movieCd", ""),
                    "movie_nm": item.get("movieNm", ""),
                    # audi_acc/rank_inten 은 KOBIS 가 문자열로 내려준다 → int 변환
                    "audi_acc": int(item.get("audiAcc", 0) or 0),
                    # open_dt 는 YYYY-MM-DD 형식으로 내려옴 → 하이픈 제거
                    "open_dt": (item.get("openDt", "") or "").replace("-", ""),
                    "rank_inten": int(item.get("rankInten", 0) or 0),
                    "rank_old_and_new": item.get("rankOldAndNew", "OLD"),
                })
            except (ValueError, TypeError) as parse_err:
                # 한 항목 파싱 실패가 전체 응답을 깨뜨리지 않도록 항목 단위 skip
                logger.warning(
                    "kobis_now_showing_tool_item_parse_error",
                    item=item,
                    error=str(parse_err),
                )
                continue

        logger.info(
            "kobis_now_showing_tool_done",
            target_date=target_date,
            top_n=safe_top_n,
            result_count=len(movies),
            top_titles=[m["movie_nm"] for m in movies[:3]],
        )
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="ok").inc()
        external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
            time.perf_counter() - started
        )
        return movies

    except httpx.TimeoutException:
        logger.error(
            "kobis_now_showing_tool_timeout",
            target_date=target_date,
            timeout_sec=_KOBIS_TIMEOUT_SEC,
        )
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="timeout").inc()
        external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
            time.perf_counter() - started
        )
        return "현재 상영작 정보를 잠시 불러올 수 없어요"

    except Exception as e:
        # 에러 전파 금지 — 호출 측은 안내 문자열만 보고 다른 카드(영화관) 만 노출
        logger.error(
            "kobis_now_showing_tool_error",
            target_date=target_date,
            error=str(e),
            error_type=type(e).__name__,
        )
        external_map_tool_total.labels(tool=_TOOL_NAME, outcome="exception").inc()
        external_map_tool_duration_seconds.labels(tool=_TOOL_NAME).observe(
            time.perf_counter() - started
        )
        return "현재 상영작 정보를 잠시 불러올 수 없어요"
