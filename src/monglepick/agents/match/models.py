"""
Movie Match Agent Pydantic 모델 + LangGraph TypedDict State (§21-2).

모델 목록:
- MovieMatchState    : LangGraph TypedDict State (11개 필드)
- SharedFeatures     : 두 영화의 교집합 특성 분석 결과
- MatchScoreDetail   : 매칭 스코어 상세 내역 (sim_to_movie_1/2, match_score 등)
- MatchedMovie       : 매칭 추천 결과 영화 (MatchScoreDetail 포함)
- MovieMatchRequest  : API 요청 모델
- MovieMatchResponse : 동기 API 응답 모델 (디버그용)

유틸 함수:
- jaccard(set_a, set_b)                              : Jaccard 유사도 (0~1)
- cosine_similarity(vec_a, vec_b)                    : 코사인 유사도 (0~1, 음수 클램핑)
- calculate_similarity(candidate, movie)             : 4개 컴포넌트 가중합 유사도
- calculate_match_score(candidate, movie_1, movie_2) : min(simA, simB) 최종 매치 스코어
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from pydantic import BaseModel, Field


# ============================================================
# SharedFeatures — 두 영화의 교집합 특성 분석 결과
# ============================================================

class SharedFeatures(BaseModel):
    """
    두 영화의 교집합 특성 분석 결과 (feature_extractor 노드 출력).

    set intersection으로 계산한 공통 장르/무드/키워드와
    LLM이 생성한 유사성 요약 문장을 포함한다.
    SSE shared_features 이벤트로 프론트엔드에 전달된다.
    """

    common_genres: list[str] = Field(
        default_factory=list,
        description="두 영화의 공통 장르 목록 (set intersection)",
    )
    common_moods: list[str] = Field(
        default_factory=list,
        description="두 영화의 공통 무드 태그 목록 (set intersection)",
    )
    common_keywords: list[str] = Field(
        default_factory=list,
        description="두 영화의 공통 키워드 목록 (set intersection)",
    )
    common_directors: list[str] = Field(
        default_factory=list,
        description="공통 감독 (동일 감독이 연출한 경우)",
    )
    common_cast: list[str] = Field(
        default_factory=list,
        description="공통 출연진 목록 (두 영화 모두 출연한 배우)",
    )
    era_range: tuple[int, int] = Field(
        default=(1900, 2030),
        description="두 영화의 개봉연도 범위 [min_year-5, max_year+5]",
    )
    avg_rating: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="두 영화의 평균 TMDB 평점",
    )
    similarity_summary: str = Field(
        default="",
        description="두 영화 간 유사성 요약 (EXAONE 32B 생성)",
    )


# ============================================================
# MatchScoreDetail — 매칭 스코어 상세 내역
# ============================================================

class MatchScoreDetail(BaseModel):
    """
    후보 영화와 두 입력 영화 간의 유사도 상세 내역 (match_scorer 노드 출력).

    핵심 원칙:
    - match_score = min(sim_to_movie_1, sim_to_movie_2)
    - 한쪽에만 유사한 영화는 낮은 점수 → 양쪽 모두에 유사해야 높은 점수
    """

    sim_to_movie_1: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="첫 번째 선택 영화와의 유사도 (0~1)",
    )
    sim_to_movie_2: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="두 번째 선택 영화와의 유사도 (0~1)",
    )
    match_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="최종 매치 스코어 = min(sim_to_movie_1, sim_to_movie_2)",
    )
    genre_overlap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="후보 영화의 장르와 두 영화 공통 장르의 Jaccard 교집합 비율",
    )
    mood_overlap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="후보 영화의 무드와 두 영화 공통 무드의 Jaccard 교집합 비율",
    )
    keyword_overlap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="후보 영화의 키워드와 두 영화 공통 키워드의 Jaccard 교집합 비율",
    )


# ============================================================
# MatchedMovie — 매칭 추천 결과 영화
# ============================================================

class MatchedMovie(BaseModel):
    """
    Movie Match 추천 결과 영화 (match_scorer → explanation_generator 출력).

    MatchScoreDetail과 LLM 생성 explanation을 포함하며,
    SSE match_result 이벤트로 프론트엔드에 전달된다.
    """

    movie_id: str = Field(
        ...,
        description="영화 ID (TMDB ID 등, 예: 'tmdb_550')",
    )
    title: str = Field(
        default="",
        description="영화 제목 (한국어 또는 원제)",
    )
    title_en: str | None = Field(
        default=None,
        description="영문 제목",
    )
    genres: list[str] = Field(
        default_factory=list,
        description="장르 목록",
    )
    mood_tags: list[str] = Field(
        default_factory=list,
        description="무드 태그 목록",
    )
    release_year: int | None = Field(
        default=None,
        description="개봉 연도",
    )
    rating: float | None = Field(
        default=None,
        description="TMDB 평점 (0~10)",
    )
    poster_path: str | None = Field(
        default=None,
        description="포스터 이미지 경로 (Qdrant payload에서 추출)",
    )
    overview: str | None = Field(
        default=None,
        description="줄거리 (300자 내외)",
    )
    ott_platforms: list[str] = Field(
        default_factory=list,
        description="이용 가능한 OTT 플랫폼 목록",
    )
    director: str | None = Field(
        default=None,
        description="감독명",
    )
    cast_members: list[str] = Field(
        default_factory=list,
        description="주요 출연 배우 목록",
    )
    score_detail: MatchScoreDetail = Field(
        default_factory=MatchScoreDetail,
        description="매칭 스코어 상세 내역 (sim_to_movie_1/2, match_score 등)",
    )
    explanation: str = Field(
        default="",
        description="두 사람 모두 좋아할 이유 (EXAONE 32B 생성)",
    )
    rank: int = Field(
        default=0,
        description="추천 순위 (1~5)",
    )


# ============================================================
# MovieMatchRequest — API 요청 모델
# ============================================================

class MovieMatchRequest(BaseModel):
    """
    Movie Match API 요청 모델 (§21-7).

    movie_id_1과 movie_id_2는 반드시 서로 다른 영화 ID여야 한다.
    user_id는 선택적이며 로그 기록 및 추적용으로만 사용된다.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "movie_id_1": "tmdb_550",
                    "movie_id_2": "tmdb_680",
                    "user_id": "user123",
                }
            ]
        }
    }

    movie_id_1: str = Field(
        ...,
        min_length=1,
        description="첫 번째 선택 영화 ID (유효한 movie_id)",
    )
    movie_id_2: str = Field(
        ...,
        min_length=1,
        description="두 번째 선택 영화 ID (movie_id_1과 다를 것)",
    )
    user_id: str = Field(
        default="",
        description="사용자 ID (선택, 로그용). 빈 문자열이면 익명.",
    )


# ============================================================
# MovieMatchResponse — 동기 API 응답 모델 (디버그용)
# ============================================================

class MovieMatchResponse(BaseModel):
    """
    Movie Match API 동기 응답 모델 (POST /match/sync, 디버그용).

    shared_features: 두 영화의 공통 특성
    recommendations: 추천 영화 목록 (최대 5편)
    movie_1_title: 첫 번째 선택 영화 제목
    movie_2_title: 두 번째 선택 영화 제목
    """

    shared_features: SharedFeatures = Field(
        default_factory=SharedFeatures,
        description="두 영화의 공통 특성 분석 결과",
    )
    recommendations: list[MatchedMovie] = Field(
        default_factory=list,
        description="추천 영화 목록 (최대 5편, match_score 내림차순)",
    )
    movie_1_title: str = Field(
        default="",
        description="첫 번째 선택 영화 제목",
    )
    movie_2_title: str = Field(
        default="",
        description="두 번째 선택 영화 제목",
    )


# ============================================================
# LangGraph TypedDict State — MovieMatchState (§21-2)
# ============================================================

class MovieMatchState(TypedDict, total=False):
    """
    Movie Match Agent LangGraph TypedDict State (§21-2).

    6개 노드가 이 State를 읽고 쓰며 그래프를 진행한다.
    TypedDict(total=False): 모든 키가 Optional (초기 State에 일부만 존재).

    입력 필드 (API에서 설정):
        movie_id_1, movie_id_2, user_id

    노드 출력 필드 (순서대로 채워짐):
        movie_1, movie_2          → movie_loader
        shared_features           → feature_extractor
        search_query              → query_builder
        candidate_movies          → rag_retriever
        ranked_movies             → match_scorer, explanation_generator
        error                     → movie_loader (에러 시)
    """

    # ── 입력 필드 ──
    movie_id_1: str                    # 첫 번째 선택 영화 ID
    movie_id_2: str                    # 두 번째 선택 영화 ID
    user_id: str                       # 요청 사용자 ID (optional, 로그용)

    # ── movie_loader 출력 ──
    movie_1: dict[str, Any]            # 첫 번째 영화 전체 메타데이터 (embedding 벡터 포함)
    movie_2: dict[str, Any]            # 두 번째 영화 전체 메타데이터 (embedding 벡터 포함)

    # ── feature_extractor 출력 ──
    shared_features: SharedFeatures    # 두 영화의 교집합 특성 분석 결과

    # ── query_builder 출력 ──
    search_query: dict[str, Any]       # RAG 검색용 구조화 쿼리

    # ── rag_retriever 출력 ──
    candidate_movies: list[dict[str, Any]]  # RAG 검색 결과 후보 영화 목록 (embedding 포함)

    # ── match_scorer / explanation_generator 출력 ──
    ranked_movies: list[MatchedMovie]  # MMR 리랭킹 후 최종 Top 5

    # ── 에러 처리 ──
    error: str                         # 에러 메시지 (있을 경우, movie_loader에서 설정)


# ============================================================
# 유틸 함수 — 유사도 계산 (§21-5)
# ============================================================

def jaccard(set_a: set, set_b: set) -> float:
    """
    두 집합의 Jaccard 유사도를 계산한다.

    Jaccard = |A ∩ B| / |A ∪ B|
    둘 다 빈 집합이면 0.0을 반환한다.

    Args:
        set_a: 첫 번째 집합
        set_b: 두 번째 집합

    Returns:
        0.0 ~ 1.0 범위의 Jaccard 유사도
    """
    # 두 집합 모두 비어있으면 정의 불가 → 0.0 반환
    if not set_a and not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    # union이 0이면 (이론상 위 조건에서 잡히지만 안전하게 처리)
    if union == 0:
        return 0.0

    return intersection / union


def cosine_similarity(vec_a: list[float] | None, vec_b: list[float] | None) -> float:
    """
    두 벡터의 코사인 유사도를 계산한다.

    벡터가 None이거나 길이가 다르면 0.0을 반환한다.
    음수 값은 0.0으로 클램핑한다 (Qdrant Cosine은 [-1, 1] 범위이므로).

    Args:
        vec_a: 첫 번째 임베딩 벡터 (float 리스트)
        vec_b: 두 번째 임베딩 벡터 (float 리스트)

    Returns:
        0.0 ~ 1.0 범위의 코사인 유사도 (음수는 0으로 클램핑)
    """
    # 벡터가 없거나 길이가 다르면 계산 불가
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    try:
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)

        # L2 노름 계산
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # 영벡터 방지
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        # 코사인 유사도 계산 후 [-1, 1] → [0, 1] 클램핑
        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, similarity)

    except Exception:
        # numpy 계산 실패 시 0.0 반환 (에러 전파 금지)
        return 0.0


def calculate_similarity(candidate: dict[str, Any], movie: dict[str, Any]) -> float:
    """
    후보 영화와 선택 영화 간 종합 유사도를 계산한다 (§21-5).

    4개 컴포넌트 가중합:
    - genre_sim   (0.35): 장르 Jaccard 유사도 — 함께 볼 영화에서 장르가 가장 중요
    - mood_sim    (0.25): 무드태그 Jaccard 유사도 — 분위기 일치가 시청 만족도에 기여
    - keyword_sim (0.15): 키워드 Jaccard 유사도 — 세부 테마 유사성 보조 지표
    - vector_sim  (0.25): 임베딩 벡터 Cosine 유사도 — 의미적 유사성 (줄거리 반영)

    Args:
        candidate: 후보 영화 dict (genres, mood_tags, keywords, embedding 포함)
        movie: 선택 영화 dict (genres, mood_tags, keywords, embedding 포함)

    Returns:
        0.0 ~ 1.0 범위의 종합 유사도
    """
    # 장르 Jaccard 유사도
    genre_sim = jaccard(
        set(candidate.get("genres", [])),
        set(movie.get("genres", [])),
    )

    # 무드태그 Jaccard 유사도
    mood_sim = jaccard(
        set(candidate.get("mood_tags", [])),
        set(movie.get("mood_tags", [])),
    )

    # 키워드 Jaccard 유사도
    keyword_sim = jaccard(
        set(candidate.get("keywords", [])),
        set(movie.get("keywords", [])),
    )

    # 임베딩 벡터 코사인 유사도
    vector_sim = cosine_similarity(
        candidate.get("embedding"),
        movie.get("embedding"),
    )

    # 4개 컴포넌트 가중합
    return (
        0.35 * genre_sim
        + 0.25 * mood_sim
        + 0.15 * keyword_sim
        + 0.25 * vector_sim
    )


def calculate_match_score(
    candidate: dict[str, Any],
    movie_1: dict[str, Any],
    movie_2: dict[str, Any],
) -> MatchScoreDetail:
    """
    두 영화 모두에 유사한 정도를 min() 전략으로 계산한다 (§21-5).

    핵심 원리:
    - min(sim_to_movie_1, sim_to_movie_2) 사용
    - 한쪽에만 유사한 영화는 낮은 점수 → 양쪽 모두 유사해야 높은 점수

    예시:
    - A와 0.9, B와 0.8 → match_score = 0.8 (양쪽 모두 높음 → 좋은 추천)
    - A와 0.95, B와 0.2 → match_score = 0.2 (한쪽만 유사 → 나쁜 추천)

    Args:
        candidate: 후보 영화 dict
        movie_1  : 첫 번째 선택 영화 dict
        movie_2  : 두 번째 선택 영화 dict

    Returns:
        MatchScoreDetail (sim_to_movie_1/2, match_score, genre/mood/keyword_overlap 포함)
    """
    # 각 선택 영화와의 개별 유사도 계산
    sim_1 = calculate_similarity(candidate, movie_1)
    sim_2 = calculate_similarity(candidate, movie_2)

    # 공통 장르/무드/키워드와의 겹침 비율 계산
    # genre_overlap: 후보 장르 vs 두 영화의 공통 장르 교집합
    common_genres = set(movie_1.get("genres", [])) & set(movie_2.get("genres", []))
    genre_overlap = jaccard(set(candidate.get("genres", [])), common_genres)

    # mood_overlap: 후보 무드 vs 두 영화의 공통 무드 교집합
    common_moods = set(movie_1.get("mood_tags", [])) & set(movie_2.get("mood_tags", []))
    mood_overlap = jaccard(set(candidate.get("mood_tags", [])), common_moods)

    # keyword_overlap: 후보 키워드 vs 두 영화의 공통 키워드 교집합
    common_keywords = set(movie_1.get("keywords", [])) & set(movie_2.get("keywords", []))
    keyword_overlap = jaccard(set(candidate.get("keywords", [])), common_keywords)

    return MatchScoreDetail(
        sim_to_movie_1=round(sim_1, 4),
        sim_to_movie_2=round(sim_2, 4),
        match_score=round(min(sim_1, sim_2), 4),  # 핵심: min() 전략
        genre_overlap=round(genre_overlap, 4),
        mood_overlap=round(mood_overlap, 4),
        keyword_overlap=round(keyword_overlap, 4),
    )
