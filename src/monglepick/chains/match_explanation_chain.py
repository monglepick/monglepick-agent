"""
Movie Match 설명 생성 체인 (§21-3 노드 2, 6).

4개 공개 함수:
- generate_similarity_summary()        : feature_extractor용 — 두 영화 유사성 요약
- generate_match_explanation()         : explanation_generator용 — 단일 영화 추천 이유
- generate_match_explanations_batch()  : 배치 순차 실행 (Ollama 직렬 처리 특성상 순차가 효율적)
- _build_match_fallback_explanation()  : LLM 실패 시 메타데이터 기반 템플릿 폴백

LLM 모델:
- get_explanation_llm() 재사용 (EXAONE 4.0 32B, temperature=0.5)
- guarded_ainvoke() 재사용 (Ollama 세마포어 기반 동시 호출 제한)

에러 처리:
- 모든 함수 try/except, 실패 시 fallback 문자열 반환 (에러 전파 금지)
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.config import settings
from monglepick.llm.factory import get_explanation_llm, guarded_ainvoke
from monglepick.prompts.match_explanation import (
    MATCH_EXPLANATION_HUMAN_PROMPT,
    MATCH_EXPLANATION_SYSTEM_PROMPT,
    MATCH_SIMILARITY_HUMAN_PROMPT,
    MATCH_SIMILARITY_SYSTEM_PROMPT,
)

logger = structlog.get_logger()

# 배치 처리 시 LLM 설명을 생성할 최대 영화 수
# MAX_EXPLANATION_MOVIES 설정을 재사용하여 Ollama 과부하 방지
_MAX_MATCH_EXPLANATION_MOVIES: int = getattr(settings, "MAX_EXPLANATION_MOVIES", 3)


# ============================================================
# 내부 유틸 — 메타데이터 기반 폴백 설명
# ============================================================

def _build_match_fallback_explanation(
    movie: dict,
    movie_1_title: str = "",
    movie_2_title: str = "",
) -> str:
    """
    LLM 호출 실패 시 메타데이터 기반으로 기본 추천 이유를 생성한다.

    LLM이 응답하지 않거나 타임아웃이 발생할 때 항상 유효한 fallback을
    반환하여 사용자에게 빈 설명이 표시되지 않도록 한다.

    Args:
        movie          : 추천 영화 dict (title, genres, rating, director 등)
        movie_1_title  : 첫 번째 선택 영화 제목 (있으면 설명에 포함)
        movie_2_title  : 두 번째 선택 영화 제목 (있으면 설명에 포함)

    Returns:
        메타데이터 기반 기본 추천 이유 문자열 (2~3문장)
    """
    title = movie.get("title", "이 영화")
    genres = movie.get("genres", [])
    rating = movie.get("rating") or 0.0
    director = movie.get("director", "")

    # 장르 텍스트 구성 (최대 3개)
    genre_text = ", ".join(genres[:3]) if genres else "다양한 장르"

    # 기본 문장 구성
    parts = [f"<{title}>은(는) {genre_text} 장르의 영화예요."]

    # 선택 영화와의 연결 (두 제목이 모두 있을 때)
    if movie_1_title and movie_2_title:
        parts.append(
            f"{movie_1_title}와(과) {movie_2_title}의 팬 모두에게 "
            "어울리는 작품으로 함께 보기 좋아요."
        )
    elif rating > 0:
        # 제목이 없으면 평점 기반 문장으로 대체
        parts.append(f"평점 {rating:.1f}점으로 많은 분들이 좋아하는 작품이에요.")

    # 감독 정보가 있으면 추가
    if director and len(parts) < 3:
        parts.append(f"{director} 감독의 연출이 돋보여요.")

    return " ".join(parts)


# ============================================================
# 유사성 요약 생성 — feature_extractor 노드용
# ============================================================

async def generate_similarity_summary(
    movie_1: dict,
    movie_2: dict,
    common_genres: list[str] | None = None,
    common_moods: list[str] | None = None,
) -> str:
    """
    두 영화의 공통점을 분석하여 1~2문장 유사성 요약을 생성한다.

    EXAONE 4.0 32B (get_explanation_llm)로 실행하며,
    feature_extractor 노드의 SharedFeatures.similarity_summary 필드에 저장된다.

    LLM 실패 시 공통 장르/무드 기반의 규칙 기반 요약을 반환한다.

    Args:
        movie_1       : 첫 번째 영화 dict (title, genres, mood_tags, overview 등)
        movie_2       : 두 번째 영화 dict (title, genres, mood_tags, overview 등)
        common_genres : 공통 장르 목록 (feature_extractor에서 미리 계산)
        common_moods  : 공통 무드 목록 (feature_extractor에서 미리 계산)

    Returns:
        유사성 요약 문자열 (1~2문장, 한국어)
    """
    similarity_start = time.perf_counter()

    # LLM 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", MATCH_SIMILARITY_SYSTEM_PROMPT),
        ("human", MATCH_SIMILARITY_HUMAN_PROMPT),
    ])

    # 자유 텍스트 LLM (EXAONE 32B, temperature=0.5)
    llm = get_explanation_llm()

    # 입력 변수 구성
    inputs = {
        "title_1": movie_1.get("title", "영화 A"),
        "year_1": movie_1.get("release_year", ""),
        "genres_1": ", ".join(movie_1.get("genres", [])) or "미분류",
        "moods_1": ", ".join(movie_1.get("mood_tags", [])) or "정보 없음",
        "overview_1": (movie_1.get("overview", "") or "")[:200],
        "title_2": movie_2.get("title", "영화 B"),
        "year_2": movie_2.get("release_year", ""),
        "genres_2": ", ".join(movie_2.get("genres", [])) or "미분류",
        "moods_2": ", ".join(movie_2.get("mood_tags", [])) or "정보 없음",
        "overview_2": (movie_2.get("overview", "") or "")[:200],
        "common_genres": ", ".join(common_genres or []) or "없음",
        "common_moods": ", ".join(common_moods or []) or "없음",
    }

    logger.info(
        "similarity_summary_start",
        title_1=inputs["title_1"],
        title_2=inputs["title_2"],
        common_genres=inputs["common_genres"],
    )

    try:
        # 프롬프트 포맷 → guarded_ainvoke (세마포어로 Ollama 동시 호출 제한)
        prompt_value = await prompt.ainvoke(inputs)
        response = await guarded_ainvoke(
            llm, prompt_value, model=settings.EXPLANATION_MODEL,
        )
        elapsed_ms = (time.perf_counter() - similarity_start) * 1000

        # LangChain BaseMessage → 문자열 추출
        summary = response.content if hasattr(response, "content") else str(response)
        summary = summary.strip()

        logger.info(
            "similarity_summary_generated",
            title_1=inputs["title_1"],
            title_2=inputs["title_2"],
            summary_preview=summary[:80],
            elapsed_ms=round(elapsed_ms, 1),
        )
        return summary

    except Exception as e:
        elapsed_ms = (time.perf_counter() - similarity_start) * 1000
        logger.error(
            "similarity_summary_error",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )

        # LLM 실패 시 규칙 기반 fallback: 공통 장르/무드를 활용한 요약
        title_1 = movie_1.get("title", "영화 A")
        title_2 = movie_2.get("title", "영화 B")
        if common_genres:
            return (
                f"{title_1}와(과) {title_2}은(는) 모두 "
                f"{', '.join(common_genres[:2])} 장르를 공유하는 영화예요."
            )
        return (
            f"{title_1}와(과) {title_2}은(는) 비슷한 분위기와 감성을 지닌 영화예요."
        )


# ============================================================
# 단일 영화 추천 이유 생성 — explanation_generator 노드용
# ============================================================

async def generate_match_explanation(
    movie: dict,
    movie_1: dict,
    movie_2: dict,
    shared_features_summary: str = "",
    score_detail_dict: dict | None = None,
) -> str:
    """
    추천 영화에 대해 "두 사람 모두 좋아할 이유"를 생성한다.

    두 선택 영화와의 연결 포인트를 구체적으로 설명하여
    사용자가 추천 의도를 명확히 이해할 수 있도록 한다.

    Args:
        movie                   : 추천 영화 dict (title, genres, mood_tags, overview 등)
        movie_1                 : 첫 번째 선택 영화 dict
        movie_2                 : 두 번째 선택 영화 dict
        shared_features_summary : 두 영화 유사성 요약 (SharedFeatures.similarity_summary)
        score_detail_dict       : MatchScoreDetail dict
                                  (sim_to_movie_1, sim_to_movie_2 활용)

    Returns:
        2~3문장 한국어 추천 이유 문자열
        LLM 실패 시 _build_match_fallback_explanation() 결과 반환
    """
    explanation_start = time.perf_counter()

    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", MATCH_EXPLANATION_SYSTEM_PROMPT),
        ("human", MATCH_EXPLANATION_HUMAN_PROMPT),
    ])

    # 자유 텍스트 LLM (EXAONE 32B, temperature=0.5)
    llm = get_explanation_llm()

    # 유사도 값 추출 (없으면 0.0 기본값)
    sim_to_1 = (score_detail_dict or {}).get("sim_to_movie_1", 0.0)
    sim_to_2 = (score_detail_dict or {}).get("sim_to_movie_2", 0.0)

    # 입력 변수 구성
    inputs = {
        "movie_1_title": movie_1.get("title", "영화 A"),
        "movie_1_genres": ", ".join(movie_1.get("genres", [])) or "미분류",
        "movie_2_title": movie_2.get("title", "영화 B"),
        "movie_2_genres": ", ".join(movie_2.get("genres", [])) or "미분류",
        "shared_summary": shared_features_summary or "두 영화의 공통 특성",
        "recommended_title": movie.get("title", "이 영화"),
        "recommended_genres": ", ".join(movie.get("genres", [])) or "미분류",
        "recommended_moods": ", ".join(movie.get("mood_tags", [])) or "정보 없음",
        "recommended_overview": (movie.get("overview", "") or "")[:200],
        "sim_to_movie_1": sim_to_1,
        "sim_to_movie_2": sim_to_2,
    }

    logger.info(
        "match_explanation_start",
        recommended_title=inputs["recommended_title"],
        movie_1_title=inputs["movie_1_title"],
        movie_2_title=inputs["movie_2_title"],
        sim_to_movie_1=round(sim_to_1, 3),
        sim_to_movie_2=round(sim_to_2, 3),
    )

    try:
        # 프롬프트 포맷 → guarded_ainvoke (세마포어로 Ollama 동시 호출 제한)
        prompt_value = await prompt.ainvoke(inputs)
        response = await guarded_ainvoke(
            llm, prompt_value, model=settings.EXPLANATION_MODEL,
        )
        elapsed_ms = (time.perf_counter() - explanation_start) * 1000

        # LangChain BaseMessage → 문자열 추출
        explanation = response.content if hasattr(response, "content") else str(response)
        explanation = explanation.strip()

        logger.info(
            "match_explanation_generated",
            recommended_title=inputs["recommended_title"],
            explanation_preview=explanation[:60],
            elapsed_ms=round(elapsed_ms, 1),
            model=settings.EXPLANATION_MODEL,
        )
        return explanation

    except Exception as e:
        elapsed_ms = (time.perf_counter() - explanation_start) * 1000
        logger.error(
            "match_explanation_error",
            error=str(e),
            error_type=type(e).__name__,
            recommended_title=movie.get("title", ""),
            elapsed_ms=round(elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        # LLM 실패 시 메타데이터 기반 fallback
        return _build_match_fallback_explanation(
            movie=movie,
            movie_1_title=movie_1.get("title", ""),
            movie_2_title=movie_2.get("title", ""),
        )


# ============================================================
# 배치 설명 생성 — explanation_generator 노드용
# ============================================================

async def generate_match_explanations_batch(
    movies: list[dict],
    movie_1: dict,
    movie_2: dict,
    shared_features_summary: str = "",
    score_details: list[dict] | None = None,
) -> list[str]:
    """
    여러 추천 영화에 대해 "두 사람 모두 좋아할 이유"를 순차 생성한다.

    Ollama는 GPU 추론을 직렬 처리하므로 asyncio.gather 병렬 호출은
    Ollama 큐만 점유하고 실질적 병렬성이 없다. 따라서 순차 실행으로
    다른 요청이 Ollama에 접근할 수 있도록 공정하게 배분한다.

    _MAX_MATCH_EXPLANATION_MOVIES(기본 3)편까지만 LLM으로 생성하고,
    초과 영화는 _build_match_fallback_explanation() 템플릿을 사용한다.

    Args:
        movies                  : 추천 영화 dict 목록 (최대 5편)
        movie_1                 : 첫 번째 선택 영화 dict
        movie_2                 : 두 번째 선택 영화 dict
        shared_features_summary : 두 영화 유사성 요약 (SharedFeatures.similarity_summary)
        score_details           : MatchScoreDetail dict 목록 (movies와 순서 일치)

    Returns:
        추천 이유 문자열 목록 (movies와 동일한 길이, 순서 일치)
    """
    batch_start = time.perf_counter()
    max_llm = _MAX_MATCH_EXPLANATION_MOVIES

    # score_details가 없으면 빈 dict 리스트로 패딩
    if score_details is None:
        score_details = [{}] * len(movies)

    explanations: list[str] = []

    for i, (movie, score_detail) in enumerate(zip(movies, score_details)):
        # max_llm 초과 → 템플릿 fallback (LLM 호출 생략)
        if i >= max_llm:
            logger.info(
                "match_explanation_fallback_over_limit",
                title=movie.get("title", ""),
                index=i,
                max_llm=max_llm,
            )
            explanations.append(
                _build_match_fallback_explanation(
                    movie=movie,
                    movie_1_title=movie_1.get("title", ""),
                    movie_2_title=movie_2.get("title", ""),
                )
            )
            continue

        # 순차 LLM 호출 (세마포어는 generate_match_explanation 내부에서 적용)
        try:
            explanation = await generate_match_explanation(
                movie=movie,
                movie_1=movie_1,
                movie_2=movie_2,
                shared_features_summary=shared_features_summary,
                score_detail_dict=score_detail,
            )
            explanations.append(explanation)
        except Exception as e:
            # 개별 영화 실패 시 fallback으로 대체 (배치 전체 중단 금지)
            logger.error(
                "match_batch_explanation_error",
                title=movie.get("title", ""),
                index=i,
                error=str(e),
                error_type=type(e).__name__,
            )
            explanations.append(
                _build_match_fallback_explanation(
                    movie=movie,
                    movie_1_title=movie_1.get("title", ""),
                    movie_2_title=movie_2.get("title", ""),
                )
            )

    batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000

    logger.info(
        "match_explanations_batch_completed",
        movie_count=len(movies),
        llm_count=min(len(movies), max_llm),
        fallback_count=max(0, len(movies) - max_llm),
        elapsed_ms=round(batch_elapsed_ms, 1),
        model=settings.EXPLANATION_MODEL,
    )

    return explanations
