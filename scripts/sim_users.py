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
# 도메인 액션 헬퍼 (지면 절약 — 다음 응답에서 시나리오 함수 추가)
# ============================================================
#
# (다음 응답에서 추가될 함수들)
# - act_watch_history(client, movie_ids, persona, rng)
# - act_wishlist(client, movie_ids, persona, rng)
# - act_review(client, movie_ids, persona, rng)
# - act_review_like(client, persona, rng) — 다른 유저 review 풀에서
# - act_playlist(client, movie_ids, persona, rng)
# - act_post_and_comment(client, persona, rng)
# - act_quiz(client, persona, rng)
# - act_worldcup(client, movie_ids, persona, rng)
# - act_course_start(client, persona, rng)
# - act_payment(client, persona, rng) — Mock Mode 의존
# - act_point_shop(client, persona, rng)
# - act_support_ticket(client, persona, rng)


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


# main() 은 다음 응답에서 추가
def main() -> int:
    """진입점 — 다음 응답에서 시나리오 함수 + main 본체 추가."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger.warning("sim_users.py 인프라만 작성됨 — 도메인 시나리오 함수는 다음 단계에서 추가")
    logger.info("config: base_url=%s user_count=%d concurrency=%d seed=%d",
                args.base_url, args.user_count, args.concurrency, args.random_seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
