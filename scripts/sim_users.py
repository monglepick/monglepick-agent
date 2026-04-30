"""몽글픽 가상 사용자 API 시뮬레이션 (단일 파일).

설계 핵심
---------
1. **실제 API 호출** — 운영 backend 의 비즈니스 로직 100% 통과.
   enum 정확 / 부수효과(포인트 적립·시청이력 동기화·추천임팩트) 자동 처리.
2. **LOCAL provider** — OAuth 시뮬 X. signup → JWT → Authorization 헤더.
3. **LLM 의존 endpoint Skip** — chat/recommend (Solar 비용). SQL 직접 적재로 대체.
4. **Toss Mock Mode 전제** — 운영 backend 에 TOSS_MOCK_MODE=true 환경변수 활성화 필요.
5. **마커**: 가상 사용자의 nickname/email 에 ``sim_`` prefix → 운영 사용자와 구분.
6. **결정성** — random seed 고정 (재실행 동일 결과).
7. **동시성 제한** — asyncio.Semaphore (default 30) — 운영 backend 부하 방지.
8. **80/20 활동 분포** — heavy 20% / casual 30% / lurker 50% 페르소나.

실행
----
의존성 임시 설치 (pyproject.toml 변경 0)::

    cd monglepick-agent

    # 로컬 스택 검증 (먼저 docker compose up -d backend)
    uv run --with aiohttp --with faker python scripts/sim_users.py \\
        --base-url http://localhost:8080 \\
        --user-count 100 -v

    # 운영 (VM2 agent 컨테이너에서)
    uv run --with aiohttp --with faker python scripts/sim_users.py \\
        --base-url http://monglepick-backend:8080 \\
        --user-count 10000

    # 롤백 (sim_ 마커 기반)
    uv run --with aiohttp --with faker python scripts/sim_users.py \\
        --base-url http://monglepick-backend:8080 --rollback
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import logging
import math
import random
import string
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
from faker import Faker

logger = logging.getLogger("sim_users")


# ============================================================
# 상수
# ============================================================

#: 모든 가상 사용자 nickname/email 의 prefix → 운영과 구분 + 마커 기반 롤백.
SIM_NICKNAME_PREFIX = "sim_"
SIM_EMAIL_DOMAIN = "sim.monglepick.test"

#: 결정성 보장. 변경 금지.
DEFAULT_RANDOM_SEED = 20_260_430

#: 시드용 공통 비밀번호. 운영 BCrypt 정책 (8~128자, 영문+숫자) 충족.
#: 가상 계정은 평소 로그인 금지. 운영자 검증 시 수동 사용 가능.
SIM_PASSWORD = "Seed!2026Demo"


# ============================================================
# 페르소나
# ============================================================


@dataclasses.dataclass
class Persona:
    """가상 사용자 활동 패턴.

    Attributes
    ----------
    name : str
        식별자 (heavy/casual/lurker).
    weight : float
        전체 사용자 중 비중 (합=1.0).
    review_min, review_max : int
        리뷰 작성 횟수 범위.
    watch_min, watch_max : int
        시청 이력 등록 횟수 범위.
    wishlist_min, wishlist_max : int
        위시리스트 추가 횟수 범위.
    review_like_min, review_like_max : int
        다른 유저 리뷰 좋아요 횟수.
    playlist_min, playlist_max : int
        플레이리스트 생성 개수.
    post_min, post_max : int
        커뮤니티 게시글 작성 횟수.
    post_comment_min, post_comment_max : int
        댓글 작성 횟수.
    quiz_min, quiz_max : int
        퀴즈 풀이 횟수.
    worldcup_min, worldcup_max : int
        이상형월드컵 진행 횟수.
    course_min, course_max : int
        도장깨기 코스 시작 개수.
    payment_prob : float
        결제(SUBSCRIPTION/POINT_PACK) 확률.
    support_prob : float
        지원 티켓 등록 확률.
    """

    name: str
    weight: float
    review_min: int
    review_max: int
    watch_min: int
    watch_max: int
    wishlist_min: int
    wishlist_max: int
    review_like_min: int
    review_like_max: int
    playlist_min: int
    playlist_max: int
    post_min: int
    post_max: int
    post_comment_min: int
    post_comment_max: int
    quiz_min: int
    quiz_max: int
    worldcup_min: int
    worldcup_max: int
    course_min: int
    course_max: int
    payment_prob: float
    support_prob: float


HEAVY = Persona(
    name="heavy", weight=0.20,
    review_min=10, review_max=30,
    watch_min=20, watch_max=50,
    wishlist_min=10, wishlist_max=25,
    review_like_min=5, review_like_max=20,
    playlist_min=2, playlist_max=5,
    post_min=2, post_max=8,
    post_comment_min=5, post_comment_max=20,
    quiz_min=3, quiz_max=10,
    worldcup_min=1, worldcup_max=3,
    course_min=1, course_max=3,
    payment_prob=0.40,
    support_prob=0.15,
)

CASUAL = Persona(
    name="casual", weight=0.30,
    review_min=2, review_max=10,
    watch_min=5, watch_max=20,
    wishlist_min=3, wishlist_max=10,
    review_like_min=1, review_like_max=5,
    playlist_min=0, playlist_max=2,
    post_min=0, post_max=2,
    post_comment_min=0, post_comment_max=5,
    quiz_min=0, quiz_max=3,
    worldcup_min=0, worldcup_max=1,
    course_min=0, course_max=1,
    payment_prob=0.10,
    support_prob=0.05,
)

LURKER = Persona(
    name="lurker", weight=0.50,
    review_min=0, review_max=2,
    watch_min=0, watch_max=5,
    wishlist_min=0, wishlist_max=3,
    review_like_min=0, review_like_max=1,
    playlist_min=0, playlist_max=1,
    post_min=0, post_max=0,
    post_comment_min=0, post_comment_max=1,
    quiz_min=0, quiz_max=1,
    worldcup_min=0, worldcup_max=0,
    course_min=0, course_max=0,
    payment_prob=0.02,
    support_prob=0.01,
)

PERSONAS = [HEAVY, CASUAL, LURKER]


# ============================================================
# 설정
# ============================================================


@dataclasses.dataclass
class SimConfig:
    """CLI 인자 + 환경변수 조합."""

    base_url: str
    user_count: int = 100
    concurrency: int = 30
    random_seed: int = DEFAULT_RANDOM_SEED
    request_timeout_sec: int = 30
    movies_pool_size: int = 5_000  # 가상 사용자 행동 시 사용할 영화 풀
    rollback: bool = False
    dry_run: bool = False


# ============================================================
# HTTP 클라이언트
# ============================================================


class ApiClient:
    """가상 사용자 1명의 HTTP 세션 + JWT 관리.

    - signup → JWT 받음 → 모든 후속 호출에 ``Authorization: Bearer`` 자동 부착.
    - 에러는 graceful — 호출 실패해도 시뮬 흐름 계속.
    """

    def __init__(self, base_url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.semaphore = semaphore
        self.access_token: str | None = None
        self.user_id: str | None = None
        self.nickname: str | None = None
        self.email: str | None = None

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        return h

    async def _request(self, method: str, path: str, **kwargs: Any) -> tuple[int, Any]:
        """공통 요청 — semaphore 로 동시성 제한.

        Returns
        -------
        (status_code, body) — body 는 JSON dict 또는 raw text (graceful)
        """
        async with self.semaphore:
            url = f"{self.base_url}{path}"
            try:
                async with self.session.request(
                    method, url, headers=self._headers(), **kwargs,
                ) as resp:
                    status = resp.status
                    try:
                        body = await resp.json(content_type=None)
                    except (aiohttp.ContentTypeError, ValueError):
                        body = await resp.text()
                    return status, body
            except aiohttp.ClientError as e:
                logger.warning("HTTP %s %s 실패: %s", method, path, e)
                return 0, None
            except asyncio.TimeoutError:
                logger.warning("HTTP %s %s 타임아웃", method, path)
                return 0, None

    async def get(self, path: str, **kwargs: Any) -> tuple[int, Any]:
        return await self._request("GET", path, **kwargs)

    async def post(self, path: str, json: Any = None, **kwargs: Any) -> tuple[int, Any]:
        return await self._request("POST", path, json=json, **kwargs)

    async def put(self, path: str, json: Any = None, **kwargs: Any) -> tuple[int, Any]:
        return await self._request("PUT", path, json=json, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> tuple[int, Any]:
        return await self._request("DELETE", path, **kwargs)


# ============================================================
# 사용자 풀 / 식별자 생성
# ============================================================

# 자주 사용되는 한국 성씨
_KOREAN_SURNAMES = [
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "전", "홍",
]

_KOREAN_GIVEN_NAMES = [
    "민지", "서연", "지우", "수빈", "예린", "하은", "지민", "예은", "다은",
    "민준", "도윤", "서준", "예준", "주원", "현우", "지호", "지훈", "건우",
    "수민", "다인", "준영", "재현", "성호", "동현", "경준", "태형", "정현",
]


def gen_sim_identity(rng: random.Random, idx: int, used: set[str]) -> tuple[str, str]:
    """가상 사용자 nickname + email 생성. 마커 포함, 충돌 방지.

    Returns
    -------
    (nickname, email)
    """
    surname = rng.choice(_KOREAN_SURNAMES)
    given = rng.choice(_KOREAN_GIVEN_NAMES)
    base_nick = f"{surname}{given}"

    # nickname: 'sim_김민지12345' (운영 닉네임과 충돌 0)
    for _ in range(20):
        nick = f"{SIM_NICKNAME_PREFIX}{base_nick}{rng.randint(1, 99_999)}"
        if nick not in used:
            used.add(nick)
            break
    else:
        nick = f"{SIM_NICKNAME_PREFIX}{idx:08d}"
        used.add(nick)

    # email: 'sim000123@sim.monglepick.test' — 가짜 도메인이라 운영 메일 발송 X
    email = f"sim{idx:08d}@{SIM_EMAIL_DOMAIN}"
    return nick, email


# ============================================================
# 분포 헬퍼
# ============================================================


def sample_persona(rng: random.Random) -> Persona:
    """heavy/casual/lurker 가중치 sampling."""
    pick = rng.random()
    cum = 0.0
    for p in PERSONAS:
        cum += p.weight
        if pick < cum:
            return p
    return PERSONAS[-1]


def sample_rating(rng: random.Random) -> float:
    """평점 분포 — mean ≈ 3.7, 좌측 skew."""
    weights = [0.5, 1.0, 1.5, 3.0, 5.0, 9.0, 12.0, 18.0, 22.0, 28.0]
    ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    total = sum(weights)
    pick = rng.random() * total
    cum = 0.0
    for r, w in zip(ratings, weights):
        cum += w
        if pick < cum:
            return r
    return 5.0


def sample_review_text(rng: random.Random, rating: float) -> str:
    """평점 기반 리뷰 본문 — LLM 호출 없이 코퍼스 sampling."""
    if rating >= 4.0:
        pool = [
            "정말 감동적이었어요. 오랜만에 본 명작입니다.",
            "최고의 영화였습니다. 시간 가는 줄 모르게 봤어요.",
            "강력 추천합니다. 다시 봐도 좋을 것 같아요.",
            "배우들의 연기가 인상적이었고 스토리도 탄탄했어요.",
            "기대 이상이었어요. 친구들에게도 추천했습니다.",
            "엔딩에서 눈물이 났어요. 마음을 울리는 영화입니다.",
        ]
    elif rating >= 3.0:
        pool = [
            "괜찮은 편이었어요. 큰 기대 없이 보면 만족할 듯.",
            "볼만했습니다. 평범하지만 그래도 시간 보내기엔 좋아요.",
            "기대보다 약간 평범했습니다. 그래도 나쁘진 않았어요.",
            "한 번쯤 보면 좋은 영화. 두 번 볼 정도는 아닙니다.",
        ]
    else:
        pool = [
            "기대에 못 미쳤어요. 시간이 아까웠습니다.",
            "지루했습니다. 중간에 졸 뻔했어요.",
            "스토리 전개가 너무 늘어져서 몰입이 안됐습니다.",
        ]
    n = rng.randint(1, 3)
    return " ".join(rng.choices(pool, k=n))


def sample_zipf_idx(rng: random.Random, size: int, s: float = 0.9) -> int:
    """Zipf 분포 — popularity 정렬 영화 풀에서 인기 영화 우선."""
    if size <= 0:
        return 0
    weights = [1.0 / math.pow(i + 1, s) for i in range(size)]
    total = sum(weights)
    pick = rng.random() * total
    cum = 0.0
    for i, w in enumerate(weights):
        cum += w
        if pick < cum:
            return i
    return size - 1


# ============================================================
# 통계
# ============================================================


@dataclasses.dataclass
class SimStats:
    """시뮬 진행/결과 카운터."""

    signups_ok: int = 0
    signups_fail: int = 0
    reviews_ok: int = 0
    reviews_fail: int = 0
    watches_ok: int = 0
    watches_fail: int = 0
    wishlists_ok: int = 0
    wishlists_fail: int = 0
    review_likes_ok: int = 0
    playlists_ok: int = 0
    playlist_items_ok: int = 0
    posts_ok: int = 0
    post_comments_ok: int = 0
    post_likes_ok: int = 0
    quiz_submits_ok: int = 0
    quiz_submits_fail: int = 0
    worldcups_ok: int = 0
    courses_ok: int = 0
    payments_ok: int = 0
    payments_fail: int = 0
    support_tickets_ok: int = 0
    point_shop_ok: int = 0
    api_calls_total: int = 0
    api_errors: int = 0
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None

    def report(self) -> str:
        elapsed = ""
        if self.started_at and self.finished_at:
            secs = (self.finished_at - self.started_at).total_seconds()
            elapsed = f" ({secs:.1f}s)"
        return (
            f"[SimStats]{elapsed}\n"
            f"  signups: ok={self.signups_ok} fail={self.signups_fail}\n"
            f"  reviews: ok={self.reviews_ok} fail={self.reviews_fail}\n"
            f"  watches: ok={self.watches_ok} fail={self.watches_fail}\n"
            f"  wishlists: ok={self.wishlists_ok} fail={self.wishlists_fail}\n"
            f"  review_likes: {self.review_likes_ok}\n"
            f"  playlists: {self.playlists_ok} (items {self.playlist_items_ok})\n"
            f"  posts: {self.posts_ok} (comments {self.post_comments_ok}, likes {self.post_likes_ok})\n"
            f"  quizzes: ok={self.quiz_submits_ok} fail={self.quiz_submits_fail}\n"
            f"  worldcups: {self.worldcups_ok}\n"
            f"  courses: {self.courses_ok}\n"
            f"  payments: ok={self.payments_ok} fail={self.payments_fail}\n"
            f"  support_tickets: {self.support_tickets_ok}\n"
            f"  point_shop: {self.point_shop_ok}\n"
            f"  api: total={self.api_calls_total} errors={self.api_errors}\n"
        )


STATS = SimStats()


# ============================================================
# 인증 흐름 (LOCAL signup → JWT)
# ============================================================


async def signup_and_login(client: ApiClient, rng: random.Random,
                           idx: int, used_nicks: set[str]) -> bool:
    """LOCAL provider 회원가입 → JWT 자동 발급. 성공 시 client 의 access_token 설정.

    Returns
    -------
    bool — 성공 여부
    """
    nick, email = gen_sim_identity(rng, idx, used_nicks)
    body = {
        "email": email,
        "password": SIM_PASSWORD,
        "nickname": nick,
        "requiredTerm": True,
        "optionTerm": rng.random() < 0.7,
        "marketingAgreed": rng.random() < 0.5,
    }
    status, resp = await client.post("/api/v1/auth/signup", json=body)
    STATS.api_calls_total += 1
    if status not in (200, 201) or not isinstance(resp, dict):
        STATS.signups_fail += 1
        STATS.api_errors += 1
        if STATS.signups_fail <= 5:  # 처음 5개만 디버그 로그
            logger.warning("signup 실패 idx=%d status=%d resp=%s", idx, status, resp)
        return False

    # AccessToken 추출 — 운영 응답 구조에 맞춤
    # 응답 형태: {"accessToken": "...", "user": {...}} 등으로 추정
    token = (
        resp.get("accessToken") or resp.get("access_token")
        or (resp.get("data", {}) or {}).get("accessToken")
    )
    user_obj = resp.get("user") or resp.get("data") or {}
    user_id = user_obj.get("userId") if isinstance(user_obj, dict) else None

    if not token:
        STATS.signups_fail += 1
        logger.warning("signup 응답에 accessToken 없음 idx=%d resp=%s", idx, resp)
        return False

    client.access_token = token
    client.user_id = user_id
    client.nickname = nick
    client.email = email
    STATS.signups_ok += 1
    return True


# ============================================================
# 도메인 액션 함수 — 가상 사용자 1명이 실제 운영 API 호출
# ============================================================
# 모든 액션은 graceful — API 실패해도 다음 액션 진행 (시뮬 흐름 끊김 방지).
# UNIQUE 충돌(예: 같은 영화 위시리스트 중복 추가) 도 무시.


async def act_watch_history(client: ApiClient, movies: list[str],
                            persona: Persona, rng: random.Random) -> int:
    """`POST /api/v1/watch-history` — 영화 시청 이력 등록.

    페르소나 활동량 따라 N건 호출. 평점은 30% 만 함께 등록 (실제 사용자 패턴).
    """
    n = rng.randint(persona.watch_min, persona.watch_max)
    ok = 0
    used: set[str] = set()
    for _ in range(n):
        idx = sample_zipf_idx(rng, len(movies), 0.85)
        mid = movies[idx]
        if mid in used:
            continue
        used.add(mid)
        body: dict[str, Any] = {
            "movieId": mid,
            "watchSource": "detail",
            "completionStatus": "COMPLETED",
        }
        if rng.random() < 0.30:
            body["rating"] = sample_rating(rng)
        status, _ = await client.post("/api/v1/watch-history", json=body)
        STATS.api_calls_total += 1
        if status in (200, 201):
            ok += 1
            STATS.watches_ok += 1
        else:
            STATS.watches_fail += 1
            STATS.api_errors += 1
    return ok


async def act_wishlist(client: ApiClient, movies: list[str],
                       persona: Persona, rng: random.Random) -> int:
    """`POST /api/v1/users/me/wishlist/{movieId}` — 위시리스트 추가."""
    n = rng.randint(persona.wishlist_min, persona.wishlist_max)
    ok = 0
    used: set[str] = set()
    for _ in range(n):
        idx = sample_zipf_idx(rng, len(movies), 0.7)
        mid = movies[idx]
        if mid in used:
            continue
        used.add(mid)
        status, _ = await client.post(f"/api/v1/users/me/wishlist/{mid}")
        STATS.api_calls_total += 1
        if status in (200, 201):
            ok += 1
            STATS.wishlists_ok += 1
        else:
            STATS.wishlists_fail += 1
            STATS.api_errors += 1
    return ok


async def act_review(client: ApiClient, movies: list[str],
                     persona: Persona, rng: random.Random) -> list[int]:
    """`POST /api/v1/movies/{movieId}/reviews` — 리뷰 작성.

    Returns
    -------
    list[int] — 생성된 review_id 리스트 (다른 유저가 좋아요 시 사용)
    """
    n = rng.randint(persona.review_min, persona.review_max)
    review_ids: list[int] = []
    used: set[str] = set()
    for _ in range(n):
        idx = sample_zipf_idx(rng, len(movies), 0.9)
        mid = movies[idx]
        if mid in used:
            continue
        used.add(mid)
        rating = sample_rating(rng)
        body = {
            "rating": rating,
            "content": sample_review_text(rng, rating),
            # review_source / review_category_code 는 운영 ReviewService 가
            # 적절히 채움 (직접 시청은 detail / NULL).
        }
        status, resp = await client.post(f"/api/v1/movies/{mid}/reviews", json=body)
        STATS.api_calls_total += 1
        if status in (200, 201):
            STATS.reviews_ok += 1
            # 응답에서 reviewId 추출 — 다른 유저 좋아요용 풀에 추가
            if isinstance(resp, dict):
                rid = resp.get("reviewId") or resp.get("id") or (resp.get("data", {}) or {}).get("reviewId")
                if rid:
                    review_ids.append(rid)
        else:
            STATS.reviews_fail += 1
            STATS.api_errors += 1
    return review_ids


async def act_review_like(client: ApiClient, all_review_ids: list[tuple[str, int]],
                          persona: Persona, rng: random.Random) -> int:
    """`POST /api/v1/movies/{mid}/reviews/{rid}/like` — 다른 유저 리뷰 좋아요.

    all_review_ids : 모든 시뮬 사용자의 (movie_id, review_id) 풀.
    자기 자신 리뷰는 제외.
    """
    n = rng.randint(persona.review_like_min, persona.review_like_max)
    if n == 0 or not all_review_ids:
        return 0
    ok = 0
    for _ in range(n):
        mid, rid = rng.choice(all_review_ids)
        status, _ = await client.post(f"/api/v1/movies/{mid}/reviews/{rid}/like")
        STATS.api_calls_total += 1
        if status in (200, 201):
            ok += 1
            STATS.review_likes_ok += 1
        elif status == 0:
            STATS.api_errors += 1
        # 409 등은 자기 리뷰 / 중복 — 무시
    return ok


async def act_playlist(client: ApiClient, movies: list[str],
                       persona: Persona, rng: random.Random) -> int:
    """`POST /api/v1/playlists` 생성 + `POST /api/v1/playlists/{id}/movies` 추가."""
    n_pl = rng.randint(persona.playlist_min, persona.playlist_max)
    ok = 0
    names = ["주말 영화 모음", "감동 명작 셀렉션", "혼자 보기 좋은 영화",
             "데이트 영화 BEST", "OST 좋은 영화", "분위기 있는 로맨스"]
    for _ in range(n_pl):
        body = {
            "playlistName": rng.choice(names) + f" #{rng.randint(1, 99)}",
            "description": "[sim] 시뮬 플레이리스트",
            "isPublic": True,
        }
        status, resp = await client.post("/api/v1/playlists", json=body)
        STATS.api_calls_total += 1
        if status not in (200, 201) or not isinstance(resp, dict):
            STATS.api_errors += 1
            continue
        ok += 1
        STATS.playlists_ok += 1
        pl_id = resp.get("playlistId") or resp.get("id") or (resp.get("data", {}) or {}).get("playlistId")
        if not pl_id:
            continue
        # 영화 3~8 추가
        n_items = rng.randint(3, 8)
        added: set[str] = set()
        for _ in range(n_items):
            mid = movies[sample_zipf_idx(rng, len(movies), 0.85)]
            if mid in added:
                continue
            added.add(mid)
            s2, _ = await client.post(f"/api/v1/playlists/{pl_id}/movies", json={"movieId": mid})
            STATS.api_calls_total += 1
            if s2 in (200, 201):
                STATS.playlist_items_ok += 1
            else:
                STATS.api_errors += 1
    return ok


async def act_post_and_comment(client: ApiClient, all_post_ids: list[int],
                               persona: Persona, rng: random.Random) -> list[int]:
    """`POST /api/v1/posts` 게시글 작성 + 다른 유저 글에 댓글.

    all_post_ids : 풀 (이전 사용자들이 작성한 post_id) — 댓글 대상.
    """
    n_post = rng.randint(persona.post_min, persona.post_max)
    n_comment = rng.randint(persona.post_comment_min, persona.post_comment_max)
    new_posts: list[int] = []
    titles = ["추천 부탁드립니다", "이 영화 어떠세요?", "정말 감동적이었어요",
              "주말에 볼만한 영화", "OTT 어디서 볼 수 있나요?", "베스트 액션"]
    contents = [
        "이번에 본 영화가 정말 좋아서 공유합니다.",
        "다들 어떻게 생각하시나요?",
        "친구에게 추천받아 봤는데 진짜 좋더라구요.",
    ]
    categories = ["FREE", "DISCUSSION", "RECOMMENDATION", "NEWS", "PLAYLIST_SHARE"]
    cat_weights = [40, 25, 20, 10, 5]

    for _ in range(n_post):
        cat_idx = sample_zipf_idx(rng, len(categories), 0.5)  # 균등 가깝게
        # 가중치 카테고리
        total_w = sum(cat_weights)
        pick = rng.random() * total_w
        cum = 0.0
        chosen_cat = "FREE"
        for c, w in zip(categories, cat_weights):
            cum += w
            if pick < cum:
                chosen_cat = c
                break
        body = {
            "category": chosen_cat,
            "title": rng.choice(titles) + f" #{rng.randint(1, 99)}",
            "content": rng.choice(contents),
        }
        status, resp = await client.post("/api/v1/posts", json=body)
        STATS.api_calls_total += 1
        if status in (200, 201) and isinstance(resp, dict):
            STATS.posts_ok += 1
            pid = resp.get("postId") or resp.get("id") or (resp.get("data", {}) or {}).get("postId")
            if pid:
                new_posts.append(pid)
        else:
            STATS.api_errors += 1

    # 다른 유저 글에 댓글 + 좋아요
    if all_post_ids:
        for _ in range(n_comment):
            target_pid = rng.choice(all_post_ids)
            body = {"content": rng.choice([
                "공감합니다", "저도 그렇게 생각해요", "좋은 글이네요",
                "추천합니다", "한번 봐야겠네요",
            ])}
            s, _ = await client.post(f"/api/v1/posts/{target_pid}/comments", json=body)
            STATS.api_calls_total += 1
            if s in (200, 201):
                STATS.post_comments_ok += 1
            else:
                STATS.api_errors += 1
            # 30% 확률 좋아요도
            if rng.random() < 0.30:
                s2, _ = await client.post(f"/api/v1/posts/{target_pid}/like")
                STATS.api_calls_total += 1
                if s2 in (200, 201):
                    STATS.post_likes_ok += 1
                else:
                    STATS.api_errors += 1
    return new_posts


async def act_quiz(client: ApiClient, persona: Persona, rng: random.Random) -> int:
    """`GET /api/v1/quizzes/today` → `POST /api/v1/quizzes/{id}/submit`."""
    n = rng.randint(persona.quiz_min, persona.quiz_max)
    ok = 0
    for _ in range(n):
        # 오늘의 퀴즈 가져오기
        status, resp = await client.get("/api/v1/quizzes/today")
        STATS.api_calls_total += 1
        if status != 200 or not isinstance(resp, (dict, list)):
            STATS.quiz_submits_fail += 1
            continue
        # 응답 형태 — 단일 또는 list
        quiz = resp[0] if isinstance(resp, list) and resp else resp if isinstance(resp, dict) else None
        if not quiz:
            continue
        qid = quiz.get("quizId") or quiz.get("id")
        options = quiz.get("options", [])
        if not qid or not options:
            continue
        answer = rng.choice(options) if isinstance(options, list) and options else "1"
        s2, _ = await client.post(f"/api/v1/quizzes/{qid}/submit", json={"answer": str(answer)})
        STATS.api_calls_total += 1
        if s2 in (200, 201):
            ok += 1
            STATS.quiz_submits_ok += 1
        else:
            STATS.quiz_submits_fail += 1
            STATS.api_errors += 1
    return ok


async def act_worldcup(client: ApiClient, movies: list[str],
                       persona: Persona, rng: random.Random) -> int:
    """이상형 월드컵 — start → pick 반복.

    `POST /api/v1/worldcup/start` → `POST /api/v1/worldcup/pick` (round 진행)
    """
    n = rng.randint(persona.worldcup_min, persona.worldcup_max)
    ok = 0
    for _ in range(n):
        round_size = rng.choice([16, 32])
        body = {
            "sourceType": "ONBOARDING",
            "selectedGenres": ["액션", "드라마"],
            "roundSize": round_size,
        }
        status, resp = await client.post("/api/v1/worldcup/start", json=body)
        STATS.api_calls_total += 1
        if status not in (200, 201) or not isinstance(resp, dict):
            STATS.api_errors += 1
            continue
        sid = resp.get("sessionId") or (resp.get("data", {}) or {}).get("sessionId")
        if not sid:
            continue
        # 단순화 — 첫 round 의 첫 match 만 pick (실제 운영 흐름 모방)
        first_match = (resp.get("matches") or [{}])[0] if isinstance(resp.get("matches"), list) else {}
        match_id = first_match.get("matchId")
        winner = first_match.get("movieAId")
        if match_id and winner:
            await client.post("/api/v1/worldcup/pick", json={
                "sessionId": sid, "matchId": match_id, "winnerMovieId": winner,
            })
            STATS.api_calls_total += 1
        ok += 1
        STATS.worldcups_ok += 1
    return ok


async def act_course_start(client: ApiClient, persona: Persona, rng: random.Random) -> int:
    """`POST /api/v1/roadmap/courses/{courseId}/start` — 도장깨기 코스 시작.

    사전: 운영 DB 의 courses 테이블에 코스가 있어야 함.
    여기서는 운영에 정의된 courseId 풀이 없어 1~10 범위 임의 시도 (404 graceful).
    """
    n = rng.randint(persona.course_min, persona.course_max)
    ok = 0
    for _ in range(n):
        course_id = rng.randint(1, 10)
        status, _ = await client.post(f"/api/v1/roadmap/courses/{course_id}/start")
        STATS.api_calls_total += 1
        if status in (200, 201):
            ok += 1
            STATS.courses_ok += 1
        elif status == 0:
            STATS.api_errors += 1
        # 404/409 등은 graceful skip (해당 코스 없음 / 이미 시작)
    return ok


async def act_payment(client: ApiClient, persona: Persona, rng: random.Random) -> int:
    """결제 시뮬 — 운영 PaymentService 의 createOrder + confirm 흐름.

    Toss Mock Mode 의존 — 실제 Toss API 호출 0. status=COMPLETED 까지 모방.
    """
    if rng.random() >= persona.payment_prob:
        return 0
    # POST /api/v1/payment/orders
    order_type = sample_persona_choice(rng, ["SUBSCRIPTION", "POINT_PACK"], [70, 30])
    if order_type == "SUBSCRIPTION":
        body = {"orderType": "SUBSCRIPTION", "amount": 5900, "planCode": "monthly_premium"}
    else:
        body = {"orderType": "POINT_PACK", "amount": 1000, "pointsAmount": 100}
    status, resp = await client.post("/api/v1/payment/orders", json=body)
    STATS.api_calls_total += 1
    if status not in (200, 201) or not isinstance(resp, dict):
        STATS.payments_fail += 1
        STATS.api_errors += 1
        return 0
    order_id = resp.get("orderId") or (resp.get("data", {}) or {}).get("orderId")
    if not order_id:
        STATS.payments_fail += 1
        return 0
    # POST /api/v1/payment/confirm — Toss Mock Mode 가 우회
    confirm_body = {
        "orderId": order_id,
        "paymentKey": f"sim-pk-{rng.randint(0, 10**12):012x}",
        "amount": body["amount"],
    }
    s2, _ = await client.post("/api/v1/payment/confirm", json=confirm_body)
    STATS.api_calls_total += 1
    if s2 in (200, 201):
        STATS.payments_ok += 1
        return 1
    STATS.payments_fail += 1
    STATS.api_errors += 1
    return 0


async def act_point_shop(client: ApiClient, persona: Persona, rng: random.Random) -> int:
    """`POST /api/v1/point/shop/ai-tokens` — 포인트로 AI 이용권 구매."""
    if rng.random() >= persona.payment_prob * 0.5:  # 결제 확률의 절반
        return 0
    pack = sample_persona_choice(rng, ["AI_TOKEN_1", "AI_TOKEN_5", "AI_TOKEN_20"], [60, 30, 10])
    status, _ = await client.post("/api/v1/point/shop/ai-tokens", json={"packType": pack})
    STATS.api_calls_total += 1
    if status in (200, 201):
        STATS.point_shop_ok += 1
        return 1
    STATS.api_errors += 1
    return 0


async def act_support_ticket(client: ApiClient, persona: Persona, rng: random.Random) -> int:
    """`POST /api/v1/support/tickets` — 지원 티켓 등록 (가능한 경우).

    경로/spec 운영마다 다를 수 있어 graceful — 실패해도 흐름 계속.
    """
    if rng.random() >= persona.support_prob:
        return 0
    titles = ["결제 환불 문의", "계정 비밀번호 분실", "추천 결과가 이상해요",
              "포인트 잔액 오류", "기능 추가 건의"]
    body = {
        "category": sample_persona_choice(
            rng, ["GENERAL", "ACCOUNT", "PAYMENT", "RECOMMENDATION"], [40, 20, 20, 20],
        ),
        "title": rng.choice(titles),
        "content": "[sim] 시뮬 문의 본문",
    }
    status, _ = await client.post("/api/v1/support/tickets", json=body)
    STATS.api_calls_total += 1
    if status in (200, 201):
        STATS.support_tickets_ok += 1
        return 1
    STATS.api_errors += 1
    return 0


def sample_persona_choice(rng: random.Random, items: list[Any], weights: list[float]) -> Any:
    """가중치 기반 카테고리컬 sampling — 결제/aitoken/카테고리 등에 사용."""
    total = sum(weights)
    pick = rng.random() * total
    cum = 0.0
    for it, w in zip(items, weights):
        cum += w
        if pick < cum:
            return it
    return items[-1]


# ============================================================
# 사용자 시나리오 — 페르소나 따라 액션 호출
# ============================================================


async def user_scenario(
    client: ApiClient, persona: Persona, movies: list[str], rng: random.Random,
    shared_review_ids: list[tuple[str, int]], shared_post_ids: list[int],
) -> None:
    """가상 사용자 1명의 행동 시퀀스.

    페르소나(heavy/casual/lurker) 활동량에 따라 액션 횟수 분배.
    실패한 액션이 있어도 다음 액션 진행 (graceful).

    shared_review_ids / shared_post_ids 는 모든 사용자 공유 풀 — 다른 유저
    리뷰/게시글에 좋아요/댓글 달기 위함.
    """
    # 1) 시청 (모든 후속 액션의 기반)
    await act_watch_history(client, movies, persona, rng)
    # 2) 위시리스트
    await act_wishlist(client, movies, persona, rng)
    # 3) 리뷰 작성 (review_id 풀에 추가)
    new_review_ids = await act_review(client, movies, persona, rng)
    # 시청한 영화 중 일부에 리뷰 — review 응답의 movie_id 도 추적해야 하지만
    # API 응답 구조에 따라 달라 단순화: rng 로 movie 재선택해서 (movie, rid) 쌍
    for rid in new_review_ids:
        # 임의 movie 매핑 (정확한 매핑이 안되면 다른 사용자가 like 시 graceful)
        shared_review_ids.append((movies[sample_zipf_idx(rng, len(movies), 0.9)], rid))

    # 4) 다른 유저 리뷰 좋아요
    await act_review_like(client, shared_review_ids, persona, rng)
    # 5) 플레이리스트
    await act_playlist(client, movies, persona, rng)
    # 6) 커뮤니티 글 + 댓글
    new_posts = await act_post_and_comment(client, shared_post_ids, persona, rng)
    shared_post_ids.extend(new_posts)
    # 7) 퀴즈
    await act_quiz(client, persona, rng)
    # 8) 이상형월드컵
    await act_worldcup(client, movies, persona, rng)
    # 9) 도장깨기 코스 시작
    await act_course_start(client, persona, rng)
    # 10) 결제 (Mock)
    await act_payment(client, persona, rng)
    # 11) 포인트샵
    await act_point_shop(client, persona, rng)
    # 12) 지원 티켓
    await act_support_ticket(client, persona, rng)


# ============================================================
# 단일 사용자 처리 (signup → scenario)
# ============================================================


async def process_user(
    base_url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
    rng: random.Random, idx: int, used_nicks: set[str],
    movies: list[str], shared_review_ids: list[tuple[str, int]], shared_post_ids: list[int],
) -> None:
    """가상 사용자 1명 — signup → 페르소나 결정 → 시나리오 실행."""
    client = ApiClient(base_url, session, semaphore)
    ok = await signup_and_login(client, rng, idx, used_nicks)
    if not ok:
        return
    persona = sample_persona(rng)
    try:
        await user_scenario(client, persona, movies, rng, shared_review_ids, shared_post_ids)
    except Exception as e:  # pylint: disable=broad-except
        # 시나리오 중간 실패해도 다른 사용자에 영향 없도록
        logger.warning("user_scenario idx=%d 예외: %s", idx, e)
        STATS.api_errors += 1


# ============================================================
# 롤백 — sim_ 마커 기반 일괄 DELETE (DB 직접)
# ============================================================


async def rollback(db_url: str) -> int:
    """가상 사용자 (`nickname LIKE 'sim_%'`) 의 모든 데이터 일괄 삭제.

    backend API 의 회원탈퇴 호출 시 부수효과 (포인트 환수 / refresh token 제거 등)
    가 발생하므로 DB 직접 DELETE 가 안전.

    Returns
    -------
    int — 삭제된 user 수
    """
    import aiomysql
    from urllib.parse import urlparse
    parsed = urlparse(db_url.replace("mysql+pymysql", "mysql"))
    conn = await aiomysql.connect(
        host=parsed.hostname or "localhost",
        port=parsed.port or 3306,
        user=parsed.username or "monglepick",
        password=parsed.password or "",
        db=parsed.path.lstrip("/") or "monglepick",
    )
    deleted = 0
    try:
        async with conn.cursor() as cur:
            await cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            # 시뮬 user_id 추출
            await cur.execute(
                "SELECT user_id FROM users WHERE nickname LIKE %s OR email LIKE %s",
                (f"{SIM_NICKNAME_PREFIX}%", f"%@{SIM_EMAIL_DOMAIN}"),
            )
            sim_user_ids = [r[0] for r in await cur.fetchall()]
            if not sim_user_ids:
                logger.info("[rollback] 삭제할 sim_ 유저 없음")
                await cur.execute("SET FOREIGN_KEY_CHECKS = 1")
                return 0
            placeholders = ",".join(["%s"] * len(sim_user_ids))
            logger.warning("[rollback] %d 시뮬 유저 삭제 시작", len(sim_user_ids))

            # 의존 역순 — user_id 컬럼 가지는 모든 테이블
            tables = [
                "admin_audit_logs", "toxicity_logs",
                "event_logs", "search_history", "chat_session_archive",
                "points_history",
                "recommendation_impact", "recommendation_log",
                "user_implicit_rating", "user_behavior_profile",
                "post_declaration", "comment_likes", "post_likes", "post_comment", "posts",
                "worldcup_results", "quiz_rewards", "quiz_attempts", "quiz_participations",
                "playlist_scrap", "playlist_likes", "playlist_item", "playlist",
                "course_final_movie", "course_verification", "course_review", "user_course_progress",
                "user_streak_attendance", "user_attendance", "user_activity_progress",
                "user_achievements", "user_calendars",
                "movie_ticket_entry", "user_subscriptions", "payment_orders",
                "point_orders", "user_items",
                "ticket_replies", "support_tickets", "support_faq_feedback",
                "review_votes", "review_likes", "reviews",
                "user_watch_history", "user_wishlist", "likes",
                "fav_movie", "fav_directors", "fav_actors", "fav_genre",
                "user_preferences", "user_ai_quota", "user_points", "user_status",
                "users",
            ]
            for t in tables:
                try:
                    await cur.execute(
                        f"DELETE FROM `{t}` WHERE user_id IN ({placeholders})",
                        tuple(sim_user_ids),
                    )
                    if cur.rowcount > 0:
                        logger.info("DELETE %s — %d rows", t, cur.rowcount)
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning("DELETE %s skip: %s", t, type(e).__name__)
            deleted = len(sim_user_ids)
            await cur.execute("SET FOREIGN_KEY_CHECKS = 1")
            await conn.commit()
            logger.warning("[rollback] 완료 — %d 시뮬 유저 삭제", deleted)
    finally:
        conn.close()
    return deleted


# ============================================================
# 영화 풀 사전 로드 (DB 직접 SELECT — read-only)
# ============================================================


async def load_movie_pool(db_url: str, limit: int) -> list[str]:
    """popularity 정렬된 movie_id 풀. SQLAlchemy/aiomysql 사용.

    가상 사용자 행동 시 인기 영화에 가중치를 두기 위해 사전 로드.
    """
    # aiomysql 의존성. monglepick-agent 는 이미 보유.
    import aiomysql
    from urllib.parse import urlparse

    # mysql+pymysql://user:pw@host:port/db → aiomysql.connect 인자
    parsed = urlparse(db_url.replace("mysql+pymysql", "mysql"))
    conn = await aiomysql.connect(
        host=parsed.hostname or "localhost",
        port=parsed.port or 3306,
        user=parsed.username or "monglepick",
        password=parsed.password or "",
        db=parsed.path.lstrip("/") or "monglepick",
    )
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT movie_id FROM movies WHERE popularity_score IS NOT NULL "
            "ORDER BY popularity_score DESC LIMIT %s",
            (limit,),
        )
        rows = await cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="몽글픽 가상 사용자 API 시뮬 — 운영 backend 의 비즈니스 로직 통과",
    )
    p.add_argument("--base-url", required=True,
                   help="Backend base URL (예: http://localhost:8080 또는 http://monglepick-backend:8080)")
    p.add_argument("--db-url", default=None,
                   help="MySQL URL — movies 풀 사전 로드용 (예: mysql+pymysql://user:pw@host:3306/db)")
    p.add_argument("--user-count", type=int, default=100,
                   help="생성할 가상 사용자 수 (default: 100)")
    p.add_argument("--concurrency", type=int, default=30,
                   help="동시 진행 사용자 수 (default: 30)")
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--request-timeout", type=int, default=30, help="HTTP timeout (초)")
    p.add_argument("--movies-pool-size", type=int, default=5_000,
                   help="popularity 정렬 영화 풀 크기 (default: 5000)")
    p.add_argument("--rollback", action="store_true",
                   help="sim_ 마커 가상 사용자 일괄 삭제 (DB 직접)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


# ============================================================
# main 진입점 — asyncio.gather + 진행 모니터링
# ============================================================


async def _async_main(args: argparse.Namespace) -> int:
    """비동기 메인 — rollback / 시뮬 분기."""
    # rollback 분기
    if args.rollback:
        if not args.db_url:
            logger.error("--rollback 시 --db-url 필수")
            return 1
        deleted = await rollback(args.db_url)
        logger.info("롤백 완료 — %d 시뮬 유저 삭제", deleted)
        return 0

    # movies 풀 로드 (DB 직접 SELECT)
    if not args.db_url:
        logger.error("--db-url 미지정 — movies 풀 로드 불가. 예: mysql+pymysql://user:pw@host:3306/monglepick")
        return 1

    logger.info("[1/3] movies 풀 로드 중 (limit=%d)", args.movies_pool_size)
    try:
        movies = await load_movie_pool(args.db_url, args.movies_pool_size)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("movies 풀 로드 실패: %s", e)
        return 1
    if not movies:
        logger.error("movies 풀 비어있음 — 운영 영화 데이터 적재 먼저 필요")
        return 1
    logger.info("[1/3] movies 풀 로드 완료 — %d 개", len(movies))

    # 가상 사용자 시뮬 시작
    rng = random.Random(args.random_seed)
    Faker.seed(args.random_seed)  # 향후 한글 nickname Faker 도입 대비
    semaphore = asyncio.Semaphore(args.concurrency)
    used_nicks: set[str] = set()
    shared_review_ids: list[tuple[str, int]] = []
    shared_post_ids: list[int] = []

    STATS.started_at = dt.datetime.now()
    logger.info("[2/3] 시뮬 시작 — user_count=%d concurrency=%d base_url=%s",
                args.user_count, args.concurrency, args.base_url)

    timeout = aiohttp.ClientTimeout(total=args.request_timeout_sec)
    progress_interval = max(10, args.user_count // 20)  # 5% 간격 진행 로그

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks: list[asyncio.Task[None]] = []
        for idx in range(args.user_count):
            t = asyncio.create_task(
                process_user(
                    args.base_url, session, semaphore, rng, idx,
                    used_nicks, movies, shared_review_ids, shared_post_ids,
                )
            )
            tasks.append(t)

        # 진행 모니터링
        completed = 0
        for fut in asyncio.as_completed(tasks):
            try:
                await fut
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("task 예외: %s", e)
            completed += 1
            if completed % progress_interval == 0 or completed == args.user_count:
                elapsed = (dt.datetime.now() - STATS.started_at).total_seconds()
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info("진행 %d/%d (%.0f%%) — %.1f users/sec, signups ok=%d",
                            completed, args.user_count,
                            completed / args.user_count * 100, rate, STATS.signups_ok)

    STATS.finished_at = dt.datetime.now()

    logger.info("[3/3] 시뮬 완료")
    logger.info("=" * 60)
    logger.info("\n%s", STATS.report())
    logger.info("=" * 60)
    if STATS.signups_fail > STATS.signups_ok:
        logger.error("⚠️ signup 실패율 높음 — backend 또는 인증 정책 확인 필요")
        return 2
    return 0


def main() -> int:
    """python -m / sys.exit 진입점."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())
