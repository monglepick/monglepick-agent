"""
LangChain Tools 패키지 — Phase 6 도구 모음.

tool_executor_node에서 info/theater/booking 의도에 대해 호출되는 10개 도구.

도구 목록:
1. search_movies      — 내부 RAG(Qdrant+ES+Neo4j) 기반 영화 검색
2. movie_detail       — TMDB API 영화 상세 정보 조회
3. theater_search     — 카카오맵 API 기반 근처 영화관 검색 (체인 식별 + 예매 딥링크)
4. ott_availability   — TMDB Watch Providers API OTT 시청 가능 여부
5. similar_movies     — Qdrant 코사인 유사도 기반 유사 영화 검색
6. user_history       — MySQL watch_history 사용자 시청 이력 조회
7. graph_explorer     — Neo4j 영화 관계 그래프 탐색
8. web_search_movie   — DuckDuckGo 외부 검색으로 영화 줄거리/정보 보강
9. geocoding          — 카카오 Local: 지역명/주소 → 위경도 좌표 (외부 지도 연동)
10. kobis_now_showing — KOBIS 일별 박스오피스: 현재 상영중 영화 Top-N (외부 지도 연동)

각 도구는 @tool 데코레이터가 적용된 async 함수이며,
에러 발생 시 빈 값([], {}, 안내 문자열)을 반환하고 절대 에러를 전파하지 않는다.

사용 예시 (tool_executor_node에서):
    from monglepick.tools import TOOL_REGISTRY
    func = TOOL_REGISTRY.get("movie_detail")
    result = await func.ainvoke({"movie_id": "157336"})
"""

from __future__ import annotations

from monglepick.tools.geocoding import geocoding
from monglepick.tools.graph_explorer import graph_explorer
from monglepick.tools.kobis_now_showing import kobis_now_showing
from monglepick.tools.movie_detail import movie_detail
from monglepick.tools.ott_availability import ott_availability
from monglepick.tools.search_movies import search_movies
from monglepick.tools.similar_movies import similar_movies
from monglepick.tools.theater_search import theater_search
from monglepick.tools.user_history import user_history
from monglepick.tools.web_search_movie import web_search_movie

# 도구 이름 → LangChain Tool 인스턴스 매핑
# tool_executor_node에서 의도별 도구를 선택할 때 사용한다.
TOOL_REGISTRY: dict[str, object] = {
    "search_movies": search_movies,
    "movie_detail": movie_detail,
    "theater_search": theater_search,
    "ott_availability": ott_availability,
    "similar_movies": similar_movies,
    "user_history": user_history,
    "graph_explorer": graph_explorer,
    "web_search_movie": web_search_movie,
    "geocoding": geocoding,
    "kobis_now_showing": kobis_now_showing,
}

# 의도별 기본 도구 매핑 (tool_executor_node의 intent → tool name)
# info 의도: 영화 상세 정보 + OTT 가용성 + 유사 영화 + 외부 검색 보강
# theater 의도: 위치 기반 영화관(카카오) + 현재 상영작(KOBIS 박스오피스 Top-N)
#   - geocoding 은 location 미제공 시 별도 분기에서 사전 호출되므로 INTENT_TOOL_MAP 에는 포함하지 않는다.
# booking 의도: 영화관 + 현재 상영작 + 영화 검색 (체인 검색 페이지 딥링크는 theater_search 응답에 포함)
INTENT_TOOL_MAP: dict[str, list[str]] = {
    "info": ["movie_detail", "ott_availability", "similar_movies", "web_search_movie"],
    "theater": ["theater_search", "kobis_now_showing"],
    "booking": ["theater_search", "kobis_now_showing", "search_movies"],
    "search": ["search_movies", "graph_explorer"],
}

__all__ = [
    "search_movies",
    "movie_detail",
    "theater_search",
    "ott_availability",
    "similar_movies",
    "user_history",
    "graph_explorer",
    "web_search_movie",
    "geocoding",
    "kobis_now_showing",
    "TOOL_REGISTRY",
    "INTENT_TOOL_MAP",
]
