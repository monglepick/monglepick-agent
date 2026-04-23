"""
Prometheus 커스텀 메트릭 중앙 정의 모듈.

prometheus-fastapi-instrumentator 가 기본 HTTP 메트릭(http_requests_total 등)을 자동으로 노출하지만,
LangGraph 노드·서브 에이전트·외부 의존(CF, 임베딩) 수준의 세밀한 메트릭은 여기에서 명시적으로 정의한다.

관찰하고 싶은 질문:
- "Movie Match 의 p95/p99 latency 는 얼마인가?"
- "LLM Reasoning 모드와 centroid 모드의 사용 비율은?"
- "Co-watched CF 호출 성공률·타임아웃률은?"
- "CF 결과가 비어 하이브리드만으로 매치된 비율은?"

### 라벨 컨벤션 (카디널리티 가드)
- 라벨에 사용자 ID, 영화 ID, free-form 에러 메시지를 절대 넣지 않는다 (카디널리티 폭증 방지).
- 고정된 enum 값만 라벨로 허용: mode (centroid/reasoning), outcome (success/error/timeout), source (hybrid/cf/both/none).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ============================================================
# Movie Match — 그래프 실행 메트릭
# ============================================================

# Movie Match 전체 그래프 실행 횟수. outcome 라벨로 분류.
#   outcome : "success" | "no_results" | "error"
# (2026-04-14 Match v3: mode toggle 폐기 — LLM 리랭커가 기본 경로로 통합됨)
match_requests_total: Counter = Counter(
    "monglepick_match_requests_total",
    "Movie Match 그래프 실행 횟수",
    labelnames=("outcome",),
)

# Movie Match 전체 그래프 실행 시간 (초). p50/p95/p99 산출 가능.
# 버킷은 LangGraph + LLM 호출(리랭커+설명)을 고려해 0.5s~30s 범위를 촘촘하게 잡았다.
match_duration_seconds: Histogram = Histogram(
    "monglepick_match_duration_seconds",
    "Movie Match 그래프 실행 소요 시간 (초)",
    labelnames=("outcome",),
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0),
)


# ============================================================
# Co-watched CF 클라이언트 — Recommend FastAPI 호출 메트릭
# ============================================================

# CF 클라이언트 요청 건수. outcome 라벨로 성공/실패/타임아웃/빈 결과 분리.
#   outcome : "ok" | "empty" | "timeout" | "http_error" | "non_200" | "exception"
match_cowatch_request_total: Counter = Counter(
    "monglepick_match_cowatch_request_total",
    "Co-watched CF (Recommend FastAPI) 요청 건수",
    labelnames=("outcome",),
)

# CF 호출 응답 시간 (초). 캐시 hit/miss 여부는 Recommend 측에서 별도 기록하므로
# 여기서는 최종 HTTP 응답 시간(엔드-투-엔드 네트워크 포함) 만 측정한다.
match_cowatch_duration_seconds: Histogram = Histogram(
    "monglepick_match_cowatch_duration_seconds",
    "Co-watched CF 호출 소요 시간 (초)",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
)


# ============================================================
# 매치 후보 소스 비율
# ============================================================

# 최종 후보가 어느 소스에서 왔는지 집계. RRF 병합 후 각 source 의 기여도 판단용.
#   source : "hybrid_only" | "cf_only" | "both" | "none"
match_candidate_source_total: Counter = Counter(
    "monglepick_match_candidate_source_total",
    "Movie Match 최종 후보의 데이터 소스 분류",
    labelnames=("source",),
)


# ============================================================
# 벡터 centroid 사용 빈도
# ============================================================

# query_vector 가 어떤 방식으로 생성됐는지 집계.
#   kind : "centroid"      — 두 영화 embedding 정상 → centroid 사용 (Level 1-A 기본)
#          "text_rerembed" — embedding 누락(MySQL fallback 등) → 텍스트 재임베딩 fallback
match_query_vector_kind_total: Counter = Counter(
    "monglepick_match_query_vector_kind_total",
    "Movie Match rag_retriever 에서 사용된 query_vector 생성 방식",
    labelnames=("kind",),
)


# ============================================================
# LLM 리랭커 (Match v3) — Solar 호출 메트릭
# ============================================================

# LLM 리랭커 호출 횟수. outcome 라벨로 성공/실패/파싱 실패/타임아웃 분리.
#   outcome : "ok"              — JSON 파싱 성공 + 유효 점수 획득
#             "parse_error"     — LLM 응답 JSON 파싱 실패 → harmonic+cf fallback
#             "empty_response"  — LLM 이 빈 응답 반환
#             "timeout"         — 호출 타임아웃
#             "exception"       — 기타 예외
match_llm_reranker_total: Counter = Counter(
    "monglepick_match_llm_reranker_total",
    "Match v3 LLM 리랭커(Solar) 호출 결과",
    labelnames=("outcome",),
)

# LLM 리랭커 호출 지연시간 (초). Solar API 응답 + JSON 파싱 포함 E2E 측정.
match_llm_reranker_duration_seconds: Histogram = Histogram(
    "monglepick_match_llm_reranker_duration_seconds",
    "Match v3 LLM 리랭커 호출 소요 시간 (초)",
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0),
)


# ============================================================
# Phase 6 외부 지도 연동 — 도구 호출 메트릭 (2026-04-23)
# ============================================================
#
# tool_executor_node 가 INTENT_TOOL_MAP 으로 디스패치하는 외부 지도 도구들의
# 호출 횟수 / 지연 / 실패율을 추적한다. Grafana 에서 다음 질문에 답할 수 있다:
#   - "theater 의도 진입 시 location 미제공으로 geocoding 호출 비율은?"
#   - "카카오 Local API 의 p95 지연시간은?"
#   - "KOBIS 박스오피스 API 가 fault 응답을 돌려주는 빈도는?"
#   - "도구별 timeout 발생률은?"
#
# 도구 이름은 라벨로 노출 — INTENT_TOOL_MAP 의 도구 수가 10개 내외라 카디널리티 안전.
#
# outcome 라벨 enum:
#   "ok"          — 정상 결과 반환
#   "empty"       — 정상 응답이지만 결과가 0건 (geocoding 미매칭, KOBIS 빈 응답)
#   "no_api_key"  — API 키 미설정으로 호출 자체 스킵
#   "fault"       — 외부 API 가 의미있는 에러 페이로드 반환 (KOBIS faultInfo, 카카오 4xx 등)
#   "timeout"     — httpx.TimeoutException
#   "exception"   — 기타 예외 (네트워크 에러 등)

# 외부 지도 도구 호출 횟수
external_map_tool_total: Counter = Counter(
    "monglepick_external_map_tool_total",
    "외부 지도 도구 호출 결과 (theater_search / kobis_now_showing / geocoding)",
    labelnames=("tool", "outcome"),
)

# 외부 지도 도구 호출 지연시간 (초). 외부 API 왕복 + 파싱 포함 E2E.
# 카카오/KOBIS 모두 보통 100~500ms 안에 응답하므로 작은 버킷 위주.
external_map_tool_duration_seconds: Histogram = Histogram(
    "monglepick_external_map_tool_duration_seconds",
    "외부 지도 도구 호출 소요 시간 (초)",
    labelnames=("tool",),
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
)

# tool_executor_node 의 위치 해소 결과.
#   source : "client_supplied" — Client 가 navigator.geolocation 좌표를 본문 location 에 보냄
#            "geocoded"        — 메시지에서 지명 추출 → geocoding 도구로 좌표 획득 성공
#            "missing"         — 둘 다 실패, 위치 재질의 응답 발행
external_map_location_source_total: Counter = Counter(
    "monglepick_external_map_location_source_total",
    "tool_executor_node 의 location 해소 경로 분류",
    labelnames=("source",),
)
