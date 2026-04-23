"""
Movie Match SSE/동기 엔드포인트 (§21-7).

2개 엔드포인트:
- POST /api/v1/match       — SSE 스트리밍 (EventSourceResponse)
- POST /api/v1/match/sync  — 동기 JSON (디버그/테스트용)

두 영화의 교집합 특성을 분석하여 함께 볼 영화 5편을 추천한다.
비로그인 사용 가능 (JWT optional).
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from monglepick.agents.match.graph import run_match_agent, run_match_agent_sync
from monglepick.agents.match.models import (
    MatchedMovie,
    MovieMatchRequest,
    MovieMatchResponse,
    SharedFeatures,
)
from monglepick.api.chat import _resolve_user_id
from monglepick.config import settings

logger = structlog.get_logger()

# ── 동시 실행 세마포어 — Chat Agent와 공유하여 Ollama 과부하 방지 ──
_graph_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)


# ============================================================
# 라우터 정의
# ============================================================

match_router = APIRouter(tags=["match"])


# ============================================================
# POST /match — SSE 스트리밍 엔드포인트
# ============================================================

@match_router.post(
    "/match",
    summary="Movie Match SSE 스트리밍 (⚠ Swagger 테스트 불가 → /match/sync 사용)",
    description=(
        "두 영화를 선택하면 교집합 특성을 분석하여 함께 볼 영화 5편을 추천한다.\n\n"
        "**⚠ Swagger UI에서는 SSE 응답이 끝나지 않습니다. `/match/sync` 동기 엔드포인트를 사용하세요.**\n\n"
        "### SSE 이벤트 (5종)\n"
        "- `status`: 각 노드 진행 상태 `{phase, message}`\n"
        "- `shared_features`: 공통 특성 분석 결과 `{common_genres, common_moods, ...}`\n"
        "- `match_result`: 추천 결과 `{movies: [MatchedMovie]}`\n"
        "- `error`: 에러 정보 `{error_code, message}`\n"
        "- `done`: 스트림 종료 `{}`"
    ),
    response_description="SSE 이벤트 스트림 (text/event-stream). Swagger UI에서는 응답이 끝나지 않으므로 /match/sync 동기 엔드포인트를 사용하세요.",
    responses={
        200: {
            "description": "SSE 이벤트 스트림",
            "content": {"text/event-stream": {}},
        },
        400: {
            "description": "잘못된 요청 (동일 영화 선택 등)",
            "content": {
                "application/json": {
                    "example": {"detail": "movie_id_1과 movie_id_2는 다른 영화여야 합니다."}
                }
            },
        },
    },
)
async def match_sse(request: MovieMatchRequest, raw_request: Request):
    """
    Movie Match SSE 스트리밍 엔드포인트.

    두 영화 ID를 받아 교집합 기반 추천을 수행하고,
    SSE로 분석 진행 상태와 추천 결과를 실시간 전송한다.

    Args:
        request: MovieMatchRequest (movie_id_1, movie_id_2, user_id)
        raw_request: FastAPI Request (JWT 추출용)

    Returns:
        EventSourceResponse (SSE 스트림)
    """
    # ── 1. 동일 영화 검증 ──
    if request.movie_id_1 == request.movie_id_2:
        logger.warning(
            "match_same_movie_rejected",
            movie_id=request.movie_id_1,
        )
        return JSONResponse(
            status_code=400,
            content={"detail": "movie_id_1과 movie_id_2는 다른 영화여야 합니다."},
        )

    # ── 2. user_id 결정 (JWT optional) ──
    user_id = _resolve_user_id(request.user_id, raw_request)

    # ── 2-1. 게스트(비로그인) 식별자 추출 (2026-04-22, Phase 3) ──
    # Chat Agent 와 동일한 평생 1회 정책 적용. 쿠키 없으면 downstream 에서 소비 스킵.
    from monglepick.api.guest_quota_client import (
        extract_client_ip as _extract_client_ip,
        parse_guest_cookie as _parse_guest_cookie,
    )
    guest_id = _parse_guest_cookie(raw_request.cookies.get("mongle_guest")) or ""
    client_ip = _extract_client_ip(raw_request)

    logger.info(
        "match_sse_request",
        movie_id_1=request.movie_id_1,
        movie_id_2=request.movie_id_2,
        user_id=user_id or "anonymous",
        guest_id=guest_id or "",
    )

    # ── 3. SSE 이벤트 생성기 ──
    async def _event_generator():
        """세마포어로 동시 실행 제한 후 Match Agent SSE 이벤트를 yield한다."""
        import json as _json

        # ── 비로그인 게스트 쿼터 체크 (차단 시 조기 종료) ──
        # 로그인 유저의 쿼터 체크는 Match 에서 현재 생략(대상 기능은 포인트 차감 대상 X),
        # 게스트만 평생 1회 정책 적용.
        if not user_id and guest_id:
            from monglepick.api.guest_quota_client import check_quota as _check_guest_quota

            guest_check = await _check_guest_quota(guest_id, client_ip)
            if not guest_check.allowed:
                logger.info(
                    "match_sse_guest_quota_exceeded",
                    guest_id=guest_id,
                    client_ip=client_ip,
                    reason=guest_check.reason,
                )
                yield {
                    "event": "error",
                    "data": _json.dumps({
                        "message": "무료 체험 1회를 모두 사용하셨어요. 더 많은 추천을 받으려면 로그인해주세요.",
                        "error_code": "GUEST_QUOTA_EXCEEDED",
                        "reason": guest_check.reason,
                        "needs_login": True,
                    }, ensure_ascii=False),
                }
                yield {"event": "done", "data": "{}"}
                return

        async with _graph_semaphore:
            async for event in run_match_agent(
                movie_id_1=request.movie_id_1,
                movie_id_2=request.movie_id_2,
                user_id=user_id,
                guest_id=guest_id,
                client_ip=client_ip,
            ):
                yield event

    return EventSourceResponse(_event_generator())


# ============================================================
# POST /match/sync — 동기 JSON 엔드포인트 (디버그용)
# ============================================================

@match_router.post(
    "/match/sync",
    response_model=MovieMatchResponse,
    summary="Movie Match 동기 JSON (디버그용)",
    description=(
        "Movie Match를 동기 모드로 실행하여 JSON 결과를 반환한다.\n"
        "테스트/디버깅 전용 — 프로덕션에서는 SSE 엔드포인트를 사용한다."
    ),
    responses={
        200: {
            "description": "매칭 결과",
            "content": {
                "application/json": {
                    "example": {
                        "movie_1_title": "인셉션",
                        "movie_2_title": "라라랜드",
                        "shared_features": {
                            "common_genres": ["드라마"],
                            "common_moods": ["몰입", "감동"],
                            "common_keywords": [],
                            "common_directors": [],
                            "common_cast": [],
                            "era_range": [2010, 2016],
                            "avg_rating": 8.15,
                            "similarity_summary": "두 영화 모두 몰입감 있는 드라마...",
                        },
                        "recommendations": [],
                    }
                }
            },
        },
        400: {
            "description": "잘못된 요청 (동일 영화 선택 등)",
        },
    },
)
async def match_sync(request: MovieMatchRequest, raw_request: Request):
    """
    Movie Match 동기 엔드포인트 (디버그/테스트용).

    그래프를 동기 실행하고 최종 결과를 JSON으로 반환한다.

    Args:
        request: MovieMatchRequest (movie_id_1, movie_id_2, user_id)
        raw_request: FastAPI Request (JWT 추출용)

    Returns:
        MovieMatchResponse (JSON)
    """
    # ── 1. 동일 영화 검증 ──
    if request.movie_id_1 == request.movie_id_2:
        return JSONResponse(
            status_code=400,
            content={"detail": "movie_id_1과 movie_id_2는 다른 영화여야 합니다."},
        )

    # ── 2. user_id 결정 (JWT optional) ──
    user_id = _resolve_user_id(request.user_id, raw_request)

    logger.info(
        "match_sync_request",
        movie_id_1=request.movie_id_1,
        movie_id_2=request.movie_id_2,
        user_id=user_id or "anonymous",
    )

    # ── 3. 동기 실행 ──
    async with _graph_semaphore:
        final_state = await run_match_agent_sync(
            movie_id_1=request.movie_id_1,
            movie_id_2=request.movie_id_2,
            user_id=user_id,
        )

    # ── 4. 응답 구성 ──
    # movie_1, movie_2에서 제목 추출
    movie_1 = final_state.get("movie_1", {})
    movie_2 = final_state.get("movie_2", {})
    movie_1_title = movie_1.get("title", request.movie_id_1) if isinstance(movie_1, dict) else request.movie_id_1
    movie_2_title = movie_2.get("title", request.movie_id_2) if isinstance(movie_2, dict) else request.movie_id_2

    # shared_features 추출
    shared_features = final_state.get("shared_features")
    if shared_features is None:
        shared_features = SharedFeatures()
    elif not isinstance(shared_features, SharedFeatures):
        # dict인 경우 Pydantic 변환
        shared_features = SharedFeatures(**shared_features) if isinstance(shared_features, dict) else SharedFeatures()

    # ranked_movies 추출
    ranked_movies = final_state.get("ranked_movies", [])
    recommendations = []
    for movie in ranked_movies:
        if isinstance(movie, MatchedMovie):
            recommendations.append(movie)
        elif isinstance(movie, dict):
            try:
                recommendations.append(MatchedMovie(**movie))
            except Exception:
                pass

    return MovieMatchResponse(
        movie_1_title=movie_1_title,
        movie_2_title=movie_2_title,
        shared_features=shared_features,
        recommendations=recommendations,
    )
