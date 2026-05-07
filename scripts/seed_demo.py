"""몽글픽 운영 DB 데모 데이터 시드 (단일 파일).

설계 핵심
---------
1. **SQLAlchemy MetaData.reflect()** 로 운영 DB 스키마를 자동 추출 — 컬럼 mismatch 위험 0.
2. **Faker(ko_KR)** 로 자연스러운 한국어 데이터.
3. **마커**: 모든 시드 user_id = ``seed_xxxxxx`` prefix → 단일 SQL 1줄로 깨끗한 롤백.
4. **분포**: Pareto(80/20) · Lognormal · Zipf · Power-law — 운영 데이터처럼.
5. **부수효과 우회**: SQL 직접 INSERT 라 Toss/임베딩/Redis writeback/메일 자동 호출 0.

실행 방법
---------
의존성 임시 설치 (pyproject.toml 변경 0)::

    cd monglepick-agent

    # 로컬 docker mysql 검증 (먼저 docker compose up -d mysql)
    uv run --with faker --with pymysql python scripts/seed_demo.py \\
        --db-url "mysql+pymysql://monglepick:monglepick_dev@localhost:3306/monglepick" \\
        --user-count 1000 --review-count 5000

    # Dry-run (실제 INSERT 없이 카운트만)
    uv run --with faker --with pymysql python scripts/seed_demo.py \\
        --db-url "..." --dry-run

    # 운영 적용 (백업 후, VM2 agent 컨테이너에서)
    uv run --with faker --with pymysql python scripts/seed_demo.py \\
        --db-url "$DB_URL" --user-count 10000 --review-count 100000

    # 롤백 (마커 기반 1 트랜잭션)
    uv run --with faker --with pymysql python scripts/seed_demo.py \\
        --db-url "..." --rollback
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import random
import string
import sys
from collections.abc import Iterable, Sequence
from typing import Any

from faker import Faker
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.engine import Engine

# ============================================================
# 상수
# ============================================================

#: 모든 시드 user_id 의 prefix. 운영 UUID 와 충돌 0 (UUID 는 16진수만 사용).
USER_ID_PREFIX = "seed_"

#: BCrypt 공통 해시. 평문 'Seed!2026' (Spring Security 호환).
SEED_PASSWORD_HASH = "$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy"

#: 결정성 보장. 변경 금지.
DEFAULT_RANDOM_SEED = 20_260_430

logger = logging.getLogger("seed_demo")


# ============================================================
# 분포 헬퍼
# ============================================================


def sample_weighted_idx(rng: random.Random, weights: Sequence[float]) -> int:
    total = sum(weights)
    pick = rng.random() * total
    cum = 0.0
    for i, w in enumerate(weights):
        cum += w
        if pick < cum:
            return i
    return len(weights) - 1


def sample_weighted(rng: random.Random, items: Sequence[Any], weights: Sequence[float]) -> Any:
    return items[sample_weighted_idx(rng, weights)]


def sample_pareto(rng: random.Random, alpha: float = 1.16, scale: float = 1.0) -> float:
    """Pareto α=1.16 — 상위 20% 가 80% 활동."""
    u = rng.random()
    return scale / math.pow(1 - u, 1.0 / alpha)


def sample_rating(rng: random.Random) -> float:
    """평점 분포 — mean ≈ 3.7, 좌측 skew."""
    weights = [0.5, 1.0, 1.5, 3.0, 5.0, 9.0, 12.0, 18.0, 22.0, 28.0]
    ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    return ratings[sample_weighted_idx(rng, weights)]


def sample_review_length(rng: random.Random) -> int:
    """Lognormal(μ=4, σ=0.5). 중앙값 ≈ 55자, 최대 500자."""
    u1, u2 = rng.random(), rng.random()
    normal = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return max(15, min(500, int(math.exp(4.0 + 0.5 * normal))))


def sample_zipf_idx(rng: random.Random, size: int, s: float = 0.9) -> int:
    """Zipf — popularity 정렬된 영화 풀에서 인기 영화 우선."""
    if size <= 0:
        return 0
    weights = [1.0 / math.pow(i + 1, s) for i in range(size)]
    return sample_weighted_idx(rng, weights)


def sample_power_law_date(rng: random.Random, max_months_ago: int = 36) -> dt.datetime:
    """가입일 power-law — 최근일수록 많이 샘플링."""
    weights = [math.pow(max_months_ago - m, 1.5) for m in range(max_months_ago)]
    months_back = max_months_ago - 1 - sample_weighted_idx(rng, weights)
    seconds_back = months_back * 30 * 86_400 + rng.randint(0, 30 * 86_400)
    return dt.datetime.now() - dt.timedelta(seconds=seconds_back)


def sample_uniform_date(rng: random.Random, start: dt.datetime, end: dt.datetime | None = None) -> dt.datetime:
    end = end or dt.datetime.now()
    if start >= end:
        return start
    seconds = int((end - start).total_seconds())
    return start + dt.timedelta(seconds=rng.randint(0, max(1, seconds)))


# ============================================================
# DB 헬퍼
# ============================================================


def reflect_db(engine: Engine) -> MetaData:
    """운영 DB 의 모든 테이블 스키마 자동 추출."""
    meta = MetaData()
    meta.reflect(bind=engine)
    logger.info("스키마 reflect 완료 — %d 테이블", len(meta.tables))
    return meta


def insert_batch(
    engine: Engine,
    table: Table,
    rows: list[dict[str, Any]],
    *,
    ignore_duplicates: bool = False,
    batch_size: int = 1_000,
    dry_run: bool = False,
) -> int:
    """Multi-row INSERT — IGNORE 옵션으로 UNIQUE 충돌 무시."""
    if not rows:
        return 0
    if dry_run:
        logger.info("[dry-run] %s ← %d rows", table.name, len(rows))
        return len(rows)

    total = 0
    with engine.begin() as conn:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            stmt = mysql_insert(table).values(chunk)
            if ignore_duplicates:
                stmt = stmt.prefix_with("IGNORE")
            result = conn.execute(stmt)
            total += result.rowcount or 0
    return total


def fill_required_defaults(table: Table, row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """NN 컬럼 중 row 에 없고 default/server_default 도 없으면 보수적 default 채움.

    reflect 로 추출한 스키마에서 NN 컬럼만 자동 보강.
    """
    out = dict(row)
    for col in table.columns:
        if col.name in out:
            continue
        if col.nullable:
            continue
        if col.default is not None or col.server_default is not None:
            continue
        if col.autoincrement is True:
            continue
        # NN + default 없음 → 타입별 보수적 default
        py_type = col.type.python_type if hasattr(col.type, "python_type") else None
        if py_type in (int, float):
            out[col.name] = 0
        elif py_type is bool:
            out[col.name] = False
        elif py_type is dt.datetime:
            out[col.name] = dt.datetime.now()
        elif py_type is dt.date:
            out[col.name] = dt.date.today()
        elif py_type is str:
            out[col.name] = ""
        else:
            out[col.name] = None
    return out


# ============================================================
# 분포 사양
# ============================================================

PROVIDERS = ["LOCAL", "NAVER", "KAKAO", "GOOGLE"]
PROVIDER_WEIGHTS = [90.0, 5.0, 3.0, 2.0]

EVENT_TYPES = ["view", "click", "play", "scroll", "search", "wishlist_add",
               "like", "review_open", "movie_detail", "trailer_play"]
EVENT_TYPE_WEIGHTS = [25, 18, 8, 12, 10, 5, 6, 5, 6, 5]

SEARCH_KEYWORDS = [
    "액션 영화", "코미디", "감동적인", "한국 영화", "로맨스", "스릴러", "공포",
    "넷플릭스", "추천", "최신 영화", "주말 영화", "데이트", "OTT", "디즈니플러스",
    "왓챠", "마블", "히어로", "범죄", "전쟁", "역사", "애니메이션", "지브리", "픽사",
]

POSITIVE_REVIEWS = [
    "정말 감동적이었어요. 오랜만에 본 명작입니다.",
    "최고의 영화였습니다. 시간 가는 줄 모르게 봤어요.",
    "강력 추천합니다. 다시 봐도 좋을 것 같아요.",
    "배우들의 연기가 인상적이었고 스토리도 탄탄했어요.",
    "기대 이상이었어요. 친구들에게도 추천했습니다.",
    "장면 하나하나가 예술 같았어요. 영상미가 압도적입니다.",
    "엔딩에서 눈물이 났어요. 마음을 울리는 영화입니다.",
    "OST가 너무 좋았어요. 배경음악이 분위기를 살려줍니다.",
]
NEUTRAL_REVIEWS = [
    "괜찮은 편이었어요. 큰 기대 없이 보면 만족할 듯.",
    "볼만했습니다. 평범하지만 그래도 시간 보내기엔 좋아요.",
    "기대보다 약간 평범했습니다. 그래도 나쁘진 않았어요.",
    "한 번쯤 보면 좋은 영화. 두 번 볼 정도는 아닙니다.",
    "킬링 타임용으로는 충분했어요.",
]
NEGATIVE_REVIEWS = [
    "기대에 못 미쳤어요. 시간이 아까웠습니다.",
    "지루했습니다. 중간에 졸 뻔했어요.",
    "스토리 전개가 너무 늘어져서 몰입이 안됐습니다.",
    "결말이 너무 허무했습니다. 무엇을 말하고 싶었는지 모르겠어요.",
]


def gen_review_text(rng: random.Random, rating: float, length: int) -> str:
    if rating >= 4.0:
        pool = POSITIVE_REVIEWS
    elif rating >= 3.0:
        pool = NEUTRAL_REVIEWS
    else:
        pool = NEGATIVE_REVIEWS
    text_ = rng.choice(pool)
    while len(text_) < length and len(text_) < 480:
        text_ += " " + rng.choice(pool)
    return text_[:length]


# ============================================================
# 도메인 시드 함수
# ============================================================


def seed_users(engine: Engine, meta: MetaData, faker: Faker, rng: random.Random,
               count: int, dry_run: bool) -> list[str]:
    """users + user_points + user_ai_quota + user_preferences 적재."""
    users_t = meta.tables.get("users")
    if users_t is None:
        raise RuntimeError("users 테이블이 없습니다 — DB 스키마 확인 필요")

    # 마지막 seed_ user_id 조회 — 재실행 시 충돌 방지
    with engine.connect() as conn:
        last = conn.execute(
            text("SELECT user_id FROM users WHERE user_id LIKE :p ORDER BY user_id DESC LIMIT 1"),
            {"p": USER_ID_PREFIX + "%"},
        ).scalar()
    start_idx = 1
    if last:
        try:
            start_idx = int(last.removeprefix(USER_ID_PREFIX)) + 1
        except (ValueError, AttributeError):
            pass

    user_ids: list[str] = []
    user_rows: list[dict] = []
    used_emails: set[str] = set()
    used_nicknames: set[str] = set()

    for i in range(count):
        user_id = f"{USER_ID_PREFIX}{start_idx + i:06d}"
        user_ids.append(user_id)

        provider = sample_weighted(rng, PROVIDERS, PROVIDER_WEIGHTS)

        # 한국어 닉네임 (Faker ko_KR)
        for _ in range(20):
            nickname = faker.name()
            if rng.random() < 0.5:
                nickname += str(rng.randint(1, 9999))
            if nickname not in used_nicknames:
                used_nicknames.add(nickname)
                break
        else:
            nickname = f"sim_{i:06d}"
            used_nicknames.add(nickname)

        # 이메일
        for _ in range(20):
            email = (
                "".join(rng.choices(string.ascii_lowercase, k=rng.randint(5, 10)))
                + str(rng.randint(1, 99_999))
                + "@" + sample_weighted(rng, ["gmail.com", "naver.com", "kakao.com"], [80, 10, 10])
            )
            if email not in used_emails:
                used_emails.add(email)
                break

        created = sample_power_law_date(rng, 36)
        updated = sample_uniform_date(rng, created)

        row = {
            "user_id": user_id,
            "email": email,
            "nickname": nickname,
            "password_hash": SEED_PASSWORD_HASH if provider == "LOCAL" else None,
            "provider": provider,
            "provider_id": None if provider == "LOCAL" else f"sim-{provider.lower()}-{i}",
            "user_role": "USER",
            "required_term": True,
            "option_term": rng.random() < 0.7,
            "marketing_agreed": rng.random() < 0.5,
            "status": "ACTIVE",
            "is_deleted": False,
            "created_at": created,
            "updated_at": updated,
            "last_login_at": sample_uniform_date(rng, created),
        }
        user_rows.append(fill_required_defaults(users_t, row, rng))

    inserted = insert_batch(engine, users_t, user_rows, ignore_duplicates=True, dry_run=dry_run)
    logger.info("users INSERT — %d/%d", inserted, count)

    # 부속 테이블
    _seed_user_points(engine, meta, rng, user_ids, dry_run)
    _seed_user_ai_quota(engine, meta, rng, user_ids, dry_run)
    _seed_user_preferences(engine, meta, rng, user_ids, dry_run)

    return user_ids


def _seed_user_points(engine: Engine, meta: MetaData, rng: random.Random,
                      user_ids: list[str], dry_run: bool) -> None:
    t = meta.tables.get("user_points")
    if t is None:
        return
    grade_balance = [
        ("NORMAL",   0,      999),
        ("BRONZE",   1_000,  3_999),
        ("SILVER",   4_000,  9_999),
        ("GOLD",     10_000, 29_999),
        ("PLATINUM", 30_000, 99_999),
        ("DIAMOND",  100_000, 500_000),
    ]
    weights = [70, 15, 8, 5, 1.8, 0.2]
    rows = []
    for uid in user_ids:
        idx = sample_weighted_idx(rng, weights)
        _, lo, hi = grade_balance[idx]
        balance = rng.randint(lo, hi)
        total_earned = int(balance * (1.5 + rng.random() * 1.5))
        total_spent = max(0, total_earned - balance)
        row = {
            "user_id": uid,
            "balance": balance,
            "total_earned": total_earned,
            "daily_earned": rng.randint(0, 100),
            "daily_reset": dt.date.today(),
            "earned_by_activity": 0,
            "daily_cap_used": 0,
            "total_spent": total_spent,
        }
        rows.append(fill_required_defaults(t, row, rng))
    insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)


def _seed_user_ai_quota(engine: Engine, meta: MetaData, rng: random.Random,
                        user_ids: list[str], dry_run: bool) -> None:
    t = meta.tables.get("user_ai_quota")
    if t is None:
        return
    today = dt.date.today()
    first_of_month = today.replace(day=1)
    rows = []
    for uid in user_ids:
        row = {
            "user_id": uid,
            "daily_ai_used": 0,
            "daily_ai_reset": today,
            "monthly_coupon_used": 0,
            "monthly_reset": first_of_month,
            "purchased_ai_tokens": rng.randint(0, 50),
            "free_daily_granted": 0,
            "last_granted_date": today,
        }
        rows.append(fill_required_defaults(t, row, rng))
    insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)


def _seed_user_preferences(engine: Engine, meta: MetaData, rng: random.Random,
                           user_ids: list[str], dry_run: bool) -> None:
    t = meta.tables.get("user_preferences")
    if t is None:
        return
    genres = ["액션", "코미디", "드라마", "로맨스", "스릴러", "공포", "SF", "판타지",
              "애니메이션", "다큐멘터리", "범죄", "전쟁", "역사"]
    moods = ["감동적", "긴장감있는", "유쾌한", "잔잔한", "낭만적", "어두운", "희망찬"]
    rows = []
    for uid in user_ids:
        row = {
            "user_id": uid,
            "preferred_genres": json.dumps(rng.sample(genres, rng.randint(2, 5)), ensure_ascii=False),
            "preferred_moods": json.dumps(rng.sample(moods, rng.randint(1, 3)), ensure_ascii=False),
            "preferred_platforms": json.dumps(rng.sample(["넷플릭스", "왓챠", "티빙"], rng.randint(1, 3)), ensure_ascii=False),
            "excluded_keywords": "[]",
            "disliked_genres": "[]",
            "favorite_movies": "[]",
        }
        rows.append(fill_required_defaults(t, row, rng))
    insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)


def seed_reviews_and_history(engine: Engine, meta: MetaData, rng: random.Random,
                             user_ids: list[str], movies: list[str], target: int,
                             dry_run: bool) -> int:
    """reviews 100K (80/20 + zipf) + user_watch_history 자동 동기화."""
    reviews_t = meta.tables.get("reviews")
    if reviews_t is None or not movies:
        return 0

    # 유저별 활동량 (Pareto)
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())

    rows = []
    base = dt.datetime.now() - dt.timedelta(days=365 * 2)
    for uid, act in activities.items():
        n = max(0, int(round(target * act / total_act)))
        n = min(n, 500, len(movies) // 2)
        used: set[str] = set()
        attempts = 0
        while len(used) < n and attempts < n * 3:
            mid = movies[sample_zipf_idx(rng, len(movies), 0.9)]
            attempts += 1
            if mid in used:
                continue
            used.add(mid)
            rating = sample_rating(rng)
            length = sample_review_length(rng)
            content = gen_review_text(rng, rating, length)
            created = sample_uniform_date(rng, base)
            row = {
                "user_id": uid,
                "movie_id": mid,
                "rating": rating,
                "contents": content,
                "is_deleted": False,
                "is_blinded": False,
                "is_spoiler": rng.random() < 0.05,
                "like_count": rng.randint(0, 50),
                "review_source": "seed",
                # review_category_code 는 운영 enum (AI_RECOMMEND 등) 외 값을 넣으면
                # MyBatis ResultMapException 으로 admin 쿼리 깨짐 → NULL 로 둔다.
                "review_category_code": None,
                "created_at": created,
                "updated_at": created,
            }
            rows.append(fill_required_defaults(reviews_t, row, rng))

    inserted = insert_batch(engine, reviews_t, rows, ignore_duplicates=True, dry_run=dry_run)
    logger.info("reviews INSERT — %d", inserted)

    # user_watch_history 자동 동기화 — 단일 SQL
    if not dry_run and meta.tables.get("user_watch_history") is not None:
        with engine.begin() as conn:
            result = conn.execute(text("""
                INSERT IGNORE INTO user_watch_history
                    (user_id, movie_id, watched_at, rating, watch_source, completion_status, created_at, updated_at)
                SELECT user_id, movie_id, created_at, rating, 'review_seed', 'COMPLETED', created_at, created_at
                FROM reviews WHERE user_id LIKE :p
            """), {"p": USER_ID_PREFIX + "%"})
            wh = result.rowcount
        logger.info("user_watch_history (reviews 동기화) — %d", wh)
        inserted += wh

    return inserted


def seed_engagement_actions(engine: Engine, meta: MetaData, rng: random.Random,
                            user_ids: list[str], movies: list[str],
                            wishlist_target: int, like_target: int,
                            dry_run: bool) -> int:
    """user_wishlist + likes — UNIQUE(user, movie)."""
    total = 0
    base = dt.datetime.now() - dt.timedelta(days=365)

    # wishlist
    t = meta.tables.get("user_wishlist")
    if t is not None and movies:
        used = set()
        rows = []
        attempts = 0
        while len(rows) < wishlist_target and attempts < wishlist_target * 3:
            uid = rng.choice(user_ids)
            mid = movies[sample_zipf_idx(rng, len(movies), 0.7)]
            if (uid, mid) in used:
                attempts += 1
                continue
            used.add((uid, mid))
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "movie_id": mid,
                "created_at": created, "updated_at": created,
            }, rng))
            attempts += 1
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # likes (영화 좋아요)
    t = meta.tables.get("likes")
    if t is not None and movies:
        used = set()
        rows = []
        attempts = 0
        while len(rows) < like_target and attempts < like_target * 3:
            uid = rng.choice(user_ids)
            mid = movies[sample_zipf_idx(rng, len(movies), 0.85)]
            if (uid, mid) in used:
                attempts += 1
                continue
            used.add((uid, mid))
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "movie_id": mid,
                "deleted_at": None,
                "created_at": created, "updated_at": created,
            }, rng))
            attempts += 1
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    logger.info("wishlist + likes — %d", total)
    return total


def seed_logs(engine: Engine, meta: MetaData, rng: random.Random,
              user_ids: list[str], movies: list[str],
              event_target: int, search_target: int,
              dry_run: bool) -> int:
    """event_logs + search_history."""
    total = 0
    base = dt.datetime.now() - dt.timedelta(days=180)
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())

    # event_logs
    t = meta.tables.get("event_logs")
    if t is not None:
        rows = []
        for uid, act in activities.items():
            n = max(0, int(round(event_target * act / total_act)))
            for _ in range(n):
                etype = sample_weighted(rng, EVENT_TYPES, EVENT_TYPE_WEIGHTS)
                mid = movies[sample_zipf_idx(rng, len(movies), 0.85)] if (movies and rng.random() < 0.85) else None
                created = sample_uniform_date(rng, base)
                rows.append(fill_required_defaults(t, {
                    "user_id": uid, "movie_id": mid, "event_type": etype,
                    "recommend_score": round(rng.uniform(0, 1), 4) if rng.random() < 0.3 else None,
                    "metadata": json.dumps({"src": "seed"}),
                    "created_at": created, "updated_at": created,
                }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # search_history
    t = meta.tables.get("search_history")
    if t is not None:
        rows = []
        for _ in range(search_target):
            uid = rng.choice(user_ids)
            kw = rng.choice(SEARCH_KEYWORDS)
            searched = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "keyword": kw, "searched_at": searched,
                "result_count": rng.randint(0, 100),
                "clicked_movie_id": movies[sample_zipf_idx(rng, len(movies), 0.9)] if (movies and rng.random() < 0.3) else None,
                "filters": "{}", "created_at": searched, "updated_at": searched,
            }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    logger.info("logs (event + search) — %d", total)
    return total


def seed_attendance(engine: Engine, meta: MetaData, rng: random.Random,
                    user_ids: list[str], target: int, dry_run: bool) -> int:
    t = meta.tables.get("user_attendance")
    if t is None:
        return 0
    today = dt.date.today()
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())
    rows, used = [], set()
    for uid, act in activities.items():
        n = max(0, int(round(target * act / total_act)))
        n = min(n, 60)
        streak = 0
        for i in range(n):
            check = today - dt.timedelta(days=i)
            if (uid, check) in used:
                continue
            used.add((uid, check))
            streak += 1
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "check_date": check,
                "streak_count": streak, "reward_point": 10,
                "created_at": dt.datetime.now(), "updated_at": dt.datetime.now(),
            }, rng))
    inserted = insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)
    logger.info("user_attendance — %d", inserted)
    return inserted


# ============================================================
# 행동 흐름 시뮬 (Phase 3 — 시스템 기준)
# ============================================================
# 운영 backend 의 비즈니스 로직 흐름을 SQL 로 시뮬레이션한다.
# 각 흐름이 도메인 간 정합성 (FK·출처 추적·enum 정확) 을 보장한다.


def flow_recommend_to_review(
    engine: Engine, meta: MetaData, rng: random.Random,
    user_ids: list[str], movies: list[str],
    target_recommends: int, target_reviews: int,
    dry_run: bool,
) -> tuple[int, int, int, int]:
    """AI 추천 → 클릭 → 시청 → 평점 funnel.

    funnel:
        - recommendation_log INSERT (target_recommends 건)
        - 30% clicked → recommendation_impact.clicked = true
        - 70% detail_viewed → impact.detail_viewed = true
        - 50% watched → user_watch_history INSERT
        - 30% rated → reviews INSERT
            review_source = "rec_log_<recommendation_log_id>"
            review_category_code = "AI_RECOMMEND"

    target_reviews 까지 도달하면 reviews INSERT 종료. 다른 funnel 단계는 계속.

    Returns: (rec_log_count, impact_count, watch_count, review_count)
    """
    rec_log_t = meta.tables.get("recommendation_log")
    impact_t = meta.tables.get("recommendation_impact")
    watch_t = meta.tables.get("user_watch_history")
    review_t = meta.tables.get("reviews")
    if rec_log_t is None or not movies or not user_ids:
        logger.warning("flow_recommend_to_review skip — 필수 테이블/풀 없음")
        return (0, 0, 0, 0)

    base_date = dt.datetime.now() - dt.timedelta(days=180)

    # 1) Pareto 활동량 분배
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())

    # 2) recommendation_log row 빌드 (각 row 의 메타 추적 — INSERT 후 ID 매칭용)
    rec_rows: list[dict[str, Any]] = []
    rec_meta: list[dict[str, Any]] = []
    for uid, act in activities.items():
        n = max(0, int(round(target_recommends * act / total_act)))
        for _ in range(n):
            mid = movies[sample_zipf_idx(rng, len(movies), 0.9)]
            sid = f"seed-rec-{rng.randint(0, 10**12):012x}"[:36]
            cf = round(rng.uniform(0, 1), 4)
            cbf = round(rng.uniform(0, 1), 4)
            score = round((cf + cbf) / 2, 4)
            created = sample_uniform_date(rng, base_date)
            rec_rows.append(fill_required_defaults(rec_log_t, {
                "user_id": uid, "movie_id": mid, "session_id": sid,
                "reason": "[seed] AI 추천 사유",
                "score": score, "cf_score": cf, "cbf_score": cbf, "hybrid_score": score,
                "genre_match": round(rng.uniform(0, 1), 4),
                "mood_match": round(rng.uniform(0, 1), 4),
                "rank_position": rng.randint(1, 10),
                "user_intent": "[seed] 사용자 의도",
                "response_time_ms": rng.randint(800, 5000),
                "model_version": "seed-v1",
                "clicked": False,  # 이후 funnel 단계에서 결정
                "source_type": sample_weighted(rng, ["GRADE_FREE", "SUB_BONUS", "PURCHASED"], [70, 20, 10]),
                "created_at": created, "updated_at": created,
            }, rng))
            rec_meta.append({"uid": uid, "mid": mid, "sid": sid, "created": created})

    if dry_run:
        logger.info("[dry-run] flow_recommend_to_review — log:%d (impact/watch/review 시뮬 생략)", len(rec_rows))
        return (len(rec_rows), 0, 0, 0)

    if not rec_rows:
        return (0, 0, 0, 0)

    # 3) recommendation_log INSERT 청크 — auto_increment ID 추적
    all_ids: list[int] = []
    chunk_size = 500
    with engine.begin() as conn:
        for start in range(0, len(rec_rows), chunk_size):
            chunk = rec_rows[start:start + chunk_size]
            stmt = mysql_insert(rec_log_t).values(chunk)
            result = conn.execute(stmt)
            first_id = result.lastrowid or 0
            for i in range(len(chunk)):
                all_ids.append(first_id + i)
    logger.info("flow_recommend_to_review — recommendation_log INSERT %d, ID range %d..%d",
                len(rec_rows), all_ids[0] if all_ids else 0, all_ids[-1] if all_ids else 0)

    # 4) funnel 진행 → impact / watch / review row 빌드
    impact_rows: list[dict[str, Any]] = []
    watch_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    review_count = 0
    used_user_movie_review: set[tuple[str, str]] = set()
    used_user_movie_watch: set[tuple[str, str]] = set()

    for rec_log_id, m_ in zip(all_ids, rec_meta):
        clicked = rng.random() < 0.30
        detail = clicked and rng.random() < 0.70
        watched = detail and rng.random() < 0.50
        rated = watched and review_count < target_reviews and rng.random() < 0.30
        wishlisted = clicked and rng.random() < 0.10
        ttc = rng.randint(1, 60) if clicked else None
        dismissed = (not clicked) and rng.random() < 0.10

        if impact_t is not None:
            impact_rows.append(fill_required_defaults(impact_t, {
                "user_id": m_["uid"], "movie_id": m_["mid"],
                "recommendation_position": rng.randint(1, 10),
                "clicked": clicked, "detail_viewed": detail,
                "wishlisted": wishlisted, "watched": watched, "rated": rated,
                "time_to_click_seconds": ttc, "dismissed": dismissed,
                "recommendation_log_id": rec_log_id,
                "created_at": m_["created"], "updated_at": m_["created"],
            }, rng))

        if watched and watch_t is not None:
            key = (m_["uid"], m_["mid"])
            if key not in used_user_movie_watch:
                used_user_movie_watch.add(key)
                w_at = m_["created"] + dt.timedelta(minutes=rng.randint(1, 60))
                watch_rows.append(fill_required_defaults(watch_t, {
                    "user_id": m_["uid"], "movie_id": m_["mid"],
                    "watched_at": w_at,
                    "rating": None, "watch_source": "ai_recommend",
                    "completion_status": "COMPLETED",
                    "created_at": w_at, "updated_at": w_at,
                }, rng))

        if rated and review_t is not None:
            key = (m_["uid"], m_["mid"])
            if key not in used_user_movie_review:
                used_user_movie_review.add(key)
                rating = sample_rating(rng)
                length = sample_review_length(rng)
                r_at = m_["created"] + dt.timedelta(hours=rng.randint(1, 48))
                review_rows.append(fill_required_defaults(review_t, {
                    "user_id": m_["uid"], "movie_id": m_["mid"],
                    "rating": rating,
                    "contents": gen_review_text(rng, rating, length),
                    "is_deleted": False, "is_blinded": False,
                    "is_spoiler": rng.random() < 0.05,
                    "like_count": 0,
                    "review_source": f"rec_log_{rec_log_id}",
                    "review_category_code": "AI_RECOMMEND",
                    "created_at": r_at, "updated_at": r_at,
                }, rng))
                review_count += 1

    # 5) 일괄 INSERT
    impact_inserted = insert_batch(engine, impact_t, impact_rows, dry_run=dry_run) if impact_t is not None else 0
    watch_inserted = insert_batch(engine, watch_t, watch_rows, ignore_duplicates=True, dry_run=dry_run) if watch_t is not None else 0
    review_inserted = insert_batch(engine, review_t, review_rows, ignore_duplicates=True, dry_run=dry_run) if review_t is not None else 0

    # watch_at 기준으로 reviews.user_watch_history 자동 매칭은 운영 ReviewService 가 하지만,
    # 여기서는 review INSERT 시 review_source/category 로 출처 추적이 이미 확보됐으므로 충분.
    logger.info("flow_recommend_to_review 완료 — log:%d, impact:%d, watch:%d, review:%d (AI_RECOMMEND)",
                len(rec_rows), impact_inserted, watch_inserted, review_inserted)
    return (len(rec_rows), impact_inserted, watch_inserted, review_inserted)


def flow_chat_to_review(
    engine: Engine, meta: MetaData, rng: random.Random,
    user_ids: list[str], movies: list[str],
    target_sessions: int, target_reviews: int,
    dry_run: bool,
) -> tuple[int, int, int]:
    """챗봇 추천 → 시청 → 리뷰.

    funnel:
        - chat_session_archive INSERT (target_sessions, UNIQUE session_id)
        - 세션당 평균 2~5 영화 추천
        - 그 중 30% 시청 → user_watch_history INSERT
        - 시청의 30% 평점 → reviews INSERT
            review_source = "chat_<session_uuid>"
            review_category_code = "AI_RECOMMEND"

    Returns: (chat_count, watch_count, review_count)
    """
    chat_t = meta.tables.get("chat_session_archive")
    watch_t = meta.tables.get("user_watch_history")
    review_t = meta.tables.get("reviews")
    if chat_t is None or not movies or not user_ids:
        logger.warning("flow_chat_to_review skip")
        return (0, 0, 0)

    base_date = dt.datetime.now() - dt.timedelta(days=180)
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())

    chat_rows: list[dict[str, Any]] = []
    watch_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    review_count = 0
    used_user_movie_watch: set[tuple[str, str]] = set()
    used_user_movie_review: set[tuple[str, str]] = set()

    for uid, act in activities.items():
        n = max(0, int(round(target_sessions * act / total_act)))
        for _ in range(n):
            sid = f"seed-chat-{rng.randint(0, 10**18):018x}"[:36]
            started = sample_uniform_date(rng, base_date)
            ended = started + dt.timedelta(minutes=rng.randint(2, 30))
            turn_count = rng.randint(2, 12)
            # 세션에서 추천된 영화 풀 (2~5)
            session_movies: list[str] = []
            n_recs = rng.randint(2, 5)
            for _ in range(n_recs):
                mid = movies[sample_zipf_idx(rng, len(movies), 0.9)]
                if mid not in session_movies:
                    session_movies.append(mid)
            messages = json.dumps([
                {"role": "user", "content": "추천 영화 알려줘"},
                {"role": "assistant", "content": f"이번에 인기 있는 작품 {len(session_movies)}편을 추천드릴게요."},
            ], ensure_ascii=False)
            chat_rows.append(fill_required_defaults(chat_t, {
                "user_id": uid, "session_id": sid,
                "messages": messages,
                "turn_count": turn_count,
                "session_state": json.dumps({"recommended_movies": session_movies}, ensure_ascii=False),
                "intent_summary": json.dumps({"intent": "recommend"}, ensure_ascii=False),
                "started_at": started, "ended_at": ended,
                "title": "AI 추천 채팅", "last_message_at": ended,
                "is_active": False, "is_deleted": False,
                "recommended_movie_count": len(session_movies),
                "created_at": started, "updated_at": ended,
            }, rng))

            # 세션의 영화별 funnel
            for mid in session_movies:
                watched = rng.random() < 0.30
                rated = watched and review_count < target_reviews and rng.random() < 0.30
                if watched and watch_t is not None:
                    key = (uid, mid)
                    if key not in used_user_movie_watch:
                        used_user_movie_watch.add(key)
                        w_at = ended + dt.timedelta(hours=rng.randint(1, 72))
                        watch_rows.append(fill_required_defaults(watch_t, {
                            "user_id": uid, "movie_id": mid,
                            "watched_at": w_at, "rating": None,
                            "watch_source": f"chat_{sid}",
                            "completion_status": "COMPLETED",
                            "created_at": w_at, "updated_at": w_at,
                        }, rng))
                if rated and review_t is not None:
                    key = (uid, mid)
                    if key not in used_user_movie_review:
                        used_user_movie_review.add(key)
                        rating = sample_rating(rng)
                        length = sample_review_length(rng)
                        r_at = ended + dt.timedelta(hours=rng.randint(2, 96))
                        review_rows.append(fill_required_defaults(review_t, {
                            "user_id": uid, "movie_id": mid,
                            "rating": rating,
                            "contents": gen_review_text(rng, rating, length),
                            "is_deleted": False, "is_blinded": False,
                            "is_spoiler": rng.random() < 0.05,
                            "like_count": 0,
                            "review_source": f"chat_{sid}",
                            "review_category_code": "AI_RECOMMEND",
                            "created_at": r_at, "updated_at": r_at,
                        }, rng))
                        review_count += 1

    chat_inserted = insert_batch(engine, chat_t, chat_rows, ignore_duplicates=True, dry_run=dry_run)
    watch_inserted = insert_batch(engine, watch_t, watch_rows, ignore_duplicates=True, dry_run=dry_run) if watch_t is not None else 0
    review_inserted = insert_batch(engine, review_t, review_rows, ignore_duplicates=True, dry_run=dry_run) if review_t is not None else 0

    logger.info("flow_chat_to_review 완료 — chat:%d, watch:%d, review:%d (AI_RECOMMEND via chat)",
                chat_inserted, watch_inserted, review_inserted)
    return (chat_inserted, watch_inserted, review_inserted)


def flow_browse_to_review(
    engine: Engine, meta: MetaData, rng: random.Random,
    user_ids: list[str], movies: list[str],
    target_watches: int, target_reviews: int,
    dry_run: bool,
) -> tuple[int, int]:
    """영화 상세 페이지 직접 시청 → 리뷰 (검색/메인 캐러셀에서 진입).

    funnel:
        - user_watch_history INSERT (target_watches 건, watch_source="detail")
        - 30% 평점 → reviews INSERT
            review_source = "detail"
            review_category_code = NULL (운영도 직접 시청 시 카테고리 없음)

    Returns: (watch_count, review_count)
    """
    watch_t = meta.tables.get("user_watch_history")
    review_t = meta.tables.get("reviews")
    if watch_t is None or not movies or not user_ids:
        return (0, 0)

    base_date = dt.datetime.now() - dt.timedelta(days=180)
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())

    watch_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    review_count = 0
    used_watch: set[tuple[str, str]] = set()
    used_review: set[tuple[str, str]] = set()

    for uid, act in activities.items():
        n = max(0, int(round(target_watches * act / total_act)))
        for _ in range(n):
            mid = movies[sample_zipf_idx(rng, len(movies), 0.85)]
            key = (uid, mid)
            if key in used_watch:
                continue
            used_watch.add(key)
            w_at = sample_uniform_date(rng, base_date)
            watch_rows.append(fill_required_defaults(watch_t, {
                "user_id": uid, "movie_id": mid,
                "watched_at": w_at, "rating": None,
                "watch_source": "detail",
                "completion_status": "COMPLETED",
                "created_at": w_at, "updated_at": w_at,
            }, rng))

            # 30% 평점
            if review_t is not None and review_count < target_reviews and rng.random() < 0.30:
                if key not in used_review:
                    used_review.add(key)
                    rating = sample_rating(rng)
                    length = sample_review_length(rng)
                    r_at = w_at + dt.timedelta(hours=rng.randint(1, 48))
                    review_rows.append(fill_required_defaults(review_t, {
                        "user_id": uid, "movie_id": mid,
                        "rating": rating,
                        "contents": gen_review_text(rng, rating, length),
                        "is_deleted": False, "is_blinded": False,
                        "is_spoiler": rng.random() < 0.05,
                        "like_count": 0,
                        "review_source": "detail",
                        "review_category_code": None,
                        "created_at": r_at, "updated_at": r_at,
                    }, rng))
                    review_count += 1

    watch_inserted = insert_batch(engine, watch_t, watch_rows, ignore_duplicates=True, dry_run=dry_run)
    review_inserted = insert_batch(engine, review_t, review_rows, ignore_duplicates=True, dry_run=dry_run) if review_t is not None else 0

    logger.info("flow_browse_to_review 완료 — watch:%d, review:%d (detail / NULL)",
                watch_inserted, review_inserted)
    return (watch_inserted, review_inserted)


def flow_course_to_review(
    engine: Engine, meta: MetaData, rng: random.Random,
    user_ids: list[str], movies: list[str],
    target_courses: int, target_reviews: int,
    dry_run: bool,
) -> tuple[int, int, int]:
    """도장깨기 코스 → 영화별 시청/리뷰.

    funnel:
        - user_course_progress INSERT (target_courses 건)
        - 코스당 5~15 영화 — 그 중 verified_movies 만큼 user_watch_history + course_review
        - course_review 의 일부 → reviews 도 INSERT
            review_source = "course_<course_id>"
            review_category_code = "COURSE"

    Returns: (course_count, course_review_count, review_count)
    """
    progress_t = meta.tables.get("user_course_progress")
    course_review_t = meta.tables.get("course_review")
    watch_t = meta.tables.get("user_watch_history")
    review_t = meta.tables.get("reviews")
    if progress_t is None or not movies or not user_ids:
        return (0, 0, 0)

    base_date = dt.datetime.now() - dt.timedelta(days=180)

    # 일부 사용자만 코스 참여 (활동량 기반)
    course_users = rng.sample(user_ids, min(target_courses, len(user_ids)))

    progress_rows: list[dict[str, Any]] = []
    course_review_rows: list[dict[str, Any]] = []
    watch_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    review_count = 0
    used_watch: set[tuple[str, str]] = set()
    used_review: set[tuple[str, str]] = set()

    for uid in course_users:
        course_id = f"seed-course-{rng.randint(0, 10**12):012x}"[:50]
        total_movies = rng.randint(5, 12)
        verified = rng.randint(0, total_movies)
        progress = round(verified / total_movies * 100, 2) if total_movies else 0
        status = "COMPLETED" if verified == total_movies else "IN_PROGRESS"
        started = sample_uniform_date(rng, base_date)
        completed = started + dt.timedelta(days=rng.randint(7, 60)) if status == "COMPLETED" else None

        progress_rows.append(fill_required_defaults(progress_t, {
            "user_id": uid, "course_id": course_id,
            "total_movies": total_movies, "verified_movies": verified,
            "progress_percent": progress, "status": status,
            "started_at": started, "completed_at": completed,
            "reward_granted": status == "COMPLETED",
            "deadline_at": started + dt.timedelta(days=90),
            "created_at": started, "updated_at": completed or started,
        }, rng))

        # 코스의 영화 풀
        course_movies: list[str] = []
        used_in_course: set[str] = set()
        while len(course_movies) < total_movies:
            mid = movies[sample_zipf_idx(rng, len(movies), 0.7)]
            if mid not in used_in_course:
                used_in_course.add(mid)
                course_movies.append(mid)

        # verified 영화만큼 watch + course_review
        for i in range(verified):
            mid = course_movies[i]
            w_at = started + dt.timedelta(days=rng.randint(0, 60))
            if watch_t is not None:
                key = (uid, mid)
                if key not in used_watch:
                    used_watch.add(key)
                    watch_rows.append(fill_required_defaults(watch_t, {
                        "user_id": uid, "movie_id": mid,
                        "watched_at": w_at, "rating": None,
                        "watch_source": f"course_{course_id[:20]}",
                        "completion_status": "COMPLETED",
                        "created_at": w_at, "updated_at": w_at,
                    }, rng))

            # course_review (verified 영화별 1건)
            if course_review_t is not None:
                course_review_rows.append(fill_required_defaults(course_review_t, {
                    "course_id": course_id[:50], "movie_id": mid, "user_id": uid,
                    "review_text": "[seed] 도장깨기 한 영화 후기",
                    "verified_count": 1, "award_point": 50,
                    "created_at": w_at, "updated_at": w_at,
                }, rng))

            # 일부 → reviews (전체 reviews 의 일부)
            if review_t is not None and review_count < target_reviews and rng.random() < 0.40:
                key = (uid, mid)
                if key not in used_review:
                    used_review.add(key)
                    rating = sample_rating(rng)
                    length = sample_review_length(rng)
                    r_at = w_at + dt.timedelta(hours=rng.randint(2, 72))
                    review_rows.append(fill_required_defaults(review_t, {
                        "user_id": uid, "movie_id": mid,
                        "rating": rating,
                        "contents": gen_review_text(rng, rating, length),
                        "is_deleted": False, "is_blinded": False,
                        "is_spoiler": rng.random() < 0.05,
                        "like_count": 0,
                        "review_source": f"course_{course_id[:20]}",
                        "review_category_code": "COURSE",
                        "created_at": r_at, "updated_at": r_at,
                    }, rng))
                    review_count += 1

    progress_inserted = insert_batch(engine, progress_t, progress_rows, dry_run=dry_run)
    course_review_inserted = insert_batch(engine, course_review_t, course_review_rows, dry_run=dry_run) if course_review_t is not None else 0
    watch_inserted = insert_batch(engine, watch_t, watch_rows, ignore_duplicates=True, dry_run=dry_run) if watch_t is not None else 0
    review_inserted = insert_batch(engine, review_t, review_rows, ignore_duplicates=True, dry_run=dry_run) if review_t is not None else 0

    logger.info("flow_course_to_review 완료 — progress:%d, course_review:%d, watch:%d, review:%d (COURSE)",
                progress_inserted, course_review_inserted, watch_inserted, review_inserted)
    return (progress_inserted, course_review_inserted, review_inserted)


def flow_wishlist_to_review(
    engine: Engine, meta: MetaData, rng: random.Random,
    user_ids: list[str], movies: list[str],
    target_wishes: int, target_reviews: int,
    dry_run: bool,
) -> tuple[int, int, int]:
    """위시리스트 추가 → 시청 → 리뷰.

    funnel:
        - user_wishlist INSERT (target_wishes 건, UNIQUE(user, movie))
        - 20% 시청 → user_watch_history INSERT
        - 시청의 30% 평점 → reviews INSERT
            review_source = "wishlist"
            review_category_code = "WISHLIST"

    Returns: (wishlist_count, watch_count, review_count)
    """
    wish_t = meta.tables.get("user_wishlist")
    watch_t = meta.tables.get("user_watch_history")
    review_t = meta.tables.get("reviews")
    if wish_t is None or not movies or not user_ids:
        return (0, 0, 0)

    base_date = dt.datetime.now() - dt.timedelta(days=180)
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())

    wish_rows: list[dict[str, Any]] = []
    watch_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    review_count = 0
    used_pair: set[tuple[str, str]] = set()
    used_watch: set[tuple[str, str]] = set()
    used_review: set[tuple[str, str]] = set()

    for uid, act in activities.items():
        n = max(0, int(round(target_wishes * act / total_act)))
        for _ in range(n):
            mid = movies[sample_zipf_idx(rng, len(movies), 0.7)]
            key = (uid, mid)
            if key in used_pair:
                continue
            used_pair.add(key)
            w_at = sample_uniform_date(rng, base_date)
            wish_rows.append(fill_required_defaults(wish_t, {
                "user_id": uid, "movie_id": mid,
                "created_at": w_at, "updated_at": w_at,
            }, rng))

            # 20% 시청
            if watch_t is not None and rng.random() < 0.20:
                if key not in used_watch:
                    used_watch.add(key)
                    watched_at = w_at + dt.timedelta(days=rng.randint(1, 30))
                    watch_rows.append(fill_required_defaults(watch_t, {
                        "user_id": uid, "movie_id": mid,
                        "watched_at": watched_at, "rating": None,
                        "watch_source": "wishlist",
                        "completion_status": "COMPLETED",
                        "created_at": watched_at, "updated_at": watched_at,
                    }, rng))

                    # 시청의 30% 평점
                    if review_t is not None and review_count < target_reviews and rng.random() < 0.30:
                        if key not in used_review:
                            used_review.add(key)
                            rating = sample_rating(rng)
                            length = sample_review_length(rng)
                            r_at = watched_at + dt.timedelta(hours=rng.randint(1, 48))
                            review_rows.append(fill_required_defaults(review_t, {
                                "user_id": uid, "movie_id": mid,
                                "rating": rating,
                                "contents": gen_review_text(rng, rating, length),
                                "is_deleted": False, "is_blinded": False,
                                "is_spoiler": rng.random() < 0.05,
                                "like_count": 0,
                                "review_source": "wishlist",
                                "review_category_code": "WISHLIST",
                                "created_at": r_at, "updated_at": r_at,
                            }, rng))
                            review_count += 1

    wish_inserted = insert_batch(engine, wish_t, wish_rows, ignore_duplicates=True, dry_run=dry_run)
    watch_inserted = insert_batch(engine, watch_t, watch_rows, ignore_duplicates=True, dry_run=dry_run) if watch_t is not None else 0
    review_inserted = insert_batch(engine, review_t, review_rows, ignore_duplicates=True, dry_run=dry_run) if review_t is not None else 0

    logger.info("flow_wishlist_to_review 완료 — wishlist:%d, watch:%d, review:%d (WISHLIST)",
                wish_inserted, watch_inserted, review_inserted)
    return (wish_inserted, watch_inserted, review_inserted)


def flow_worldcup_to_review(
    engine: Engine, meta: MetaData, rng: random.Random,
    user_ids: list[str], movies: list[str],
    target_worldcups: int, target_reviews: int,
    dry_run: bool,
) -> tuple[int, int, int]:
    """이상형 월드컵 → 우승 영화 시청 → 리뷰.

    funnel:
        - worldcup_results INSERT (target_worldcups 건)
        - 우승 영화 50% 시청 → user_watch_history INSERT
        - 시청의 40% 평점 → reviews INSERT
            review_source = "worldcup"
            review_category_code = "WORLDCUP"

    Returns: (worldcup_count, watch_count, review_count)
    """
    wc_t = meta.tables.get("worldcup_results")
    watch_t = meta.tables.get("user_watch_history")
    review_t = meta.tables.get("reviews")
    if wc_t is None or not movies or not user_ids:
        return (0, 0, 0)

    base_date = dt.datetime.now() - dt.timedelta(days=180)
    wc_rows: list[dict[str, Any]] = []
    watch_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    review_count = 0
    used_watch: set[tuple[str, str]] = set()
    used_review: set[tuple[str, str]] = set()

    for _ in range(target_worldcups):
        uid = rng.choice(user_ids)
        round_size = rng.choice([16, 32, 64])
        winner = movies[sample_zipf_idx(rng, len(movies), 0.7)]
        runner_up = movies[sample_zipf_idx(rng, len(movies), 0.7)]
        if winner == runner_up:
            continue
        semi = ",".join([movies[rng.randint(0, len(movies) - 1)] for _ in range(4)])
        created = sample_uniform_date(rng, base_date)
        wc_rows.append(fill_required_defaults(wc_t, {
            "user_id": uid, "round_size": round_size,
            "semi_final_movie_ids": semi, "selection_log": "[]", "genre_preferences": "[]",
            "onboarding_completed": True, "reward_granted": True,
            "winner_movie_id": winner, "runner_up_movie_id": runner_up,
            "total_matches": round_size - 1,
            "created_at": created, "updated_at": created,
        }, rng))

        # 우승 영화 50% 시청
        if watch_t is not None and rng.random() < 0.50:
            key = (uid, winner)
            if key not in used_watch:
                used_watch.add(key)
                w_at = created + dt.timedelta(hours=rng.randint(1, 48))
                watch_rows.append(fill_required_defaults(watch_t, {
                    "user_id": uid, "movie_id": winner,
                    "watched_at": w_at, "rating": None,
                    "watch_source": "worldcup",
                    "completion_status": "COMPLETED",
                    "created_at": w_at, "updated_at": w_at,
                }, rng))

                # 시청의 40% 평점
                if review_t is not None and review_count < target_reviews and rng.random() < 0.40:
                    if key not in used_review:
                        used_review.add(key)
                        rating = sample_rating(rng)
                        length = sample_review_length(rng)
                        r_at = w_at + dt.timedelta(hours=rng.randint(1, 48))
                        review_rows.append(fill_required_defaults(review_t, {
                            "user_id": uid, "movie_id": winner,
                            "rating": rating,
                            "contents": gen_review_text(rng, rating, length),
                            "is_deleted": False, "is_blinded": False,
                            "is_spoiler": rng.random() < 0.05,
                            "like_count": 0,
                            "review_source": "worldcup",
                            "review_category_code": "WORLDCUP",
                            "created_at": r_at, "updated_at": r_at,
                        }, rng))
                        review_count += 1

    wc_inserted = insert_batch(engine, wc_t, wc_rows, dry_run=dry_run)
    watch_inserted = insert_batch(engine, watch_t, watch_rows, ignore_duplicates=True, dry_run=dry_run) if watch_t is not None else 0
    review_inserted = insert_batch(engine, review_t, review_rows, ignore_duplicates=True, dry_run=dry_run) if review_t is not None else 0

    logger.info("flow_worldcup_to_review 완료 — worldcup:%d, watch:%d, review:%d (WORLDCUP)",
                wc_inserted, watch_inserted, review_inserted)
    return (wc_inserted, watch_inserted, review_inserted)


# ============================================================
# 추가 도메인 시드 (Phase 2)
# ============================================================


def seed_favorites(engine: Engine, meta: MetaData, rng: random.Random,
                   user_ids: list[str], movies: list[str], dry_run: bool) -> int:
    """fav_genre / fav_actors / fav_directors / fav_movie."""
    total = 0
    # genre_master 풀
    genre_ids = []
    if "genre_master" in meta.tables:
        with engine.connect() as conn:
            genre_ids = [r[0] for r in conn.execute(text("SELECT genre_id FROM genre_master")).fetchall()]

    actors_pool = ["송강호", "최민식", "이정재", "황정민", "김혜수", "전도연", "박서준", "공유",
                   "이병헌", "하정우", "정우성", "마동석", "유아인", "김다미", "박소담", "김태리"]
    directors_pool = ["봉준호", "박찬욱", "김지운", "윤제균", "최동훈", "이창동", "류승완", "나홍진"]

    # fav_genre
    if genre_ids and "fav_genre" in meta.tables:
        t = meta.tables["fav_genre"]
        rows = []
        for uid in user_ids:
            sampled = rng.sample(genre_ids, min(rng.randint(1, 5), len(genre_ids)))
            for prio, gid in enumerate(sampled):
                rows.append(fill_required_defaults(t, {"user_id": uid, "genre_id": gid, "priority": prio}, rng))
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # fav_actors
    if "fav_actors" in meta.tables:
        t = meta.tables["fav_actors"]
        rows = []
        for uid in user_ids:
            for actor in rng.sample(actors_pool, rng.randint(0, 5)):
                rows.append(fill_required_defaults(t, {"user_id": uid, "actor_name": actor}, rng))
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # fav_directors
    if "fav_directors" in meta.tables:
        t = meta.tables["fav_directors"]
        rows = []
        for uid in user_ids:
            for d in rng.sample(directors_pool, rng.randint(0, 2)):
                rows.append(fill_required_defaults(t, {"user_id": uid, "director_name": d}, rng))
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # fav_movie
    if movies and "fav_movie" in meta.tables:
        t = meta.tables["fav_movie"]
        rows = []
        for uid in user_ids:
            n = rng.randint(0, 5)
            used = set()
            for _ in range(n):
                mid = movies[sample_zipf_idx(rng, len(movies), 0.8)]
                if mid in used:
                    continue
                used.add(mid)
                rows.append(fill_required_defaults(t, {"user_id": uid, "movie_id": mid, "priority": len(used)-1}, rng))
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    logger.info("favorites — %d", total)
    return total


def seed_review_engagement(engine: Engine, meta: MetaData, rng: random.Random,
                           user_ids: list[str], dry_run: bool) -> int:
    """review_likes / review_votes."""
    review_ids = []
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT review_id FROM reviews WHERE user_id LIKE :p ORDER BY RAND() LIMIT 50000"
        ), {"p": USER_ID_PREFIX + "%"}).fetchall()
        review_ids = [r[0] for r in rows]
    if not review_ids:
        return 0

    total = 0
    base = dt.datetime.now() - dt.timedelta(days=180)

    # review_likes — 50K 목표 (UNIQUE review_id, user_id)
    t = meta.tables.get("review_likes")
    if t is not None:
        rows, used = [], set()
        target = min(50_000, len(review_ids) * len(user_ids) // 100)
        attempts = 0
        while len(rows) < target and attempts < target * 3:
            rid = rng.choice(review_ids)
            uid = rng.choice(user_ids)
            if (rid, uid) in used:
                attempts += 1
                continue
            used.add((rid, uid))
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {"review_id": rid, "user_id": uid, "created_at": created, "updated_at": created}, rng))
            attempts += 1
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # review_votes — 30K
    t = meta.tables.get("review_votes")
    if t is not None:
        rows, used = [], set()
        target = min(30_000, len(review_ids) * len(user_ids) // 200)
        attempts = 0
        while len(rows) < target and attempts < target * 3:
            rid = rng.choice(review_ids)
            uid = rng.choice(user_ids)
            if (rid, uid) in used:
                attempts += 1
                continue
            used.add((rid, uid))
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "review_id": rid, "user_id": uid, "helpful": rng.random() < 0.7,
                "created_at": created, "updated_at": created,
            }, rng))
            attempts += 1
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    logger.info("review_engagement — %d", total)
    return total


def seed_recommendation(engine: Engine, meta: MetaData, rng: random.Random,
                        user_ids: list[str], movies: list[str], scale: int, dry_run: bool) -> int:
    """recommendation_log / recommendation_impact / user_implicit_rating / user_behavior_profile."""
    total = 0
    base = dt.datetime.now() - dt.timedelta(days=180)
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())

    # recommendation_log
    t = meta.tables.get("recommendation_log")
    if t is not None and movies:
        rows = []
        target = scale * 3  # 10K user → 30K
        for _ in range(target):
            uid = rng.choice(user_ids)
            mid = movies[sample_zipf_idx(rng, len(movies), 0.9)]
            sid = f"seed-{rng.randint(0, 10**12):012x}"[:36]
            cf = round(rng.uniform(0, 1), 4)
            cbf = round(rng.uniform(0, 1), 4)
            score = round((cf + cbf) / 2, 4)
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "movie_id": mid, "session_id": sid,
                "reason": "[seed] 데모 추천 사유",
                "score": score, "cf_score": cf, "cbf_score": cbf, "hybrid_score": score,
                "genre_match": round(rng.uniform(0, 1), 4), "mood_match": round(rng.uniform(0, 1), 4),
                "rank_position": rng.randint(1, 10),
                "user_intent": "[seed] 사용자 의도",
                "response_time_ms": rng.randint(800, 5000), "model_version": "seed-v1",
                "clicked": rng.random() < 0.25,
                "source_type": sample_weighted(rng, ["GRADE_FREE", "SUB_BONUS", "PURCHASED"], [70, 20, 10]),
                "created_at": created, "updated_at": created,
            }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # recommendation_impact
    t = meta.tables.get("recommendation_impact")
    if t is not None and movies:
        rows = []
        target = scale * 3
        for _ in range(target):
            uid = rng.choice(user_ids)
            mid = movies[sample_zipf_idx(rng, len(movies), 0.85)]
            clicked = rng.random() < 0.3
            detail = clicked and rng.random() < 0.7
            watched = detail and rng.random() < 0.4
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "movie_id": mid,
                "recommendation_position": rng.randint(1, 10),
                "clicked": clicked, "detail_viewed": detail,
                "wishlisted": clicked and rng.random() < 0.2,
                "watched": watched,
                "rated": watched and rng.random() < 0.5,
                "time_to_click_seconds": rng.randint(1, 60) if clicked else None,
                "dismissed": (not clicked) and rng.random() < 0.1,
                "created_at": created, "updated_at": created,
            }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # user_implicit_rating (UNIQUE user, movie)
    t = meta.tables.get("user_implicit_rating")
    if t is not None and movies:
        rows, used = [], set()
        target = scale * 5
        attempts = 0
        while len(rows) < target and attempts < target * 3:
            uid = rng.choice(user_ids)
            mid = movies[sample_zipf_idx(rng, len(movies), 0.85)]
            if (uid, mid) in used:
                attempts += 1
                continue
            used.add((uid, mid))
            last = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "movie_id": mid,
                "implicit_score": round(rng.uniform(0.1, 1.0), 4),
                "contributing_actions": json.dumps({"view": rng.randint(0, 5), "click": rng.randint(0, 3)}),
                "last_action_at": last, "created_at": last, "updated_at": last,
            }, rng))
            attempts += 1
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # user_behavior_profile (PK=user_id, 1:1)
    t = meta.tables.get("user_behavior_profile")
    if t is not None:
        rows = []
        now = dt.datetime.now()
        for uid in user_ids:
            rows.append(fill_required_defaults(t, {
                "user_id": uid,
                "genre_affinity": json.dumps({"액션": round(rng.random(), 3), "드라마": round(rng.random(), 3)}, ensure_ascii=False),
                "mood_affinity": json.dumps({"감동적": round(rng.random(), 3)}, ensure_ascii=False),
                "director_affinity": json.dumps({}),
                "taste_consistency": round(rng.uniform(0.3, 1.0), 4),
                "recommendation_acceptance_rate": round(rng.uniform(0.1, 0.9), 4),
                "avg_exploration_depth": round(rng.uniform(1, 10), 2),
                "activity_level": sample_weighted(rng, ["LOW", "MEDIUM", "HIGH"], [40, 40, 20]),
                "profile_updated_at": now,
            }, rng))
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    logger.info("recommendation domain — %d", total)
    return total


def seed_points_history(engine: Engine, meta: MetaData, rng: random.Random,
                        user_ids: list[str], scale: int, dry_run: bool) -> int:
    """points_history 100K (action 분포)."""
    t = meta.tables.get("points_history")
    if t is None:
        return 0
    base = dt.datetime.now() - dt.timedelta(days=365)
    actions = [
        ("REVIEW_CREATE", 30, 50),
        ("ATTENDANCE", 25, 10),
        ("RECOMMEND_RECEIVED", 15, 20),
        ("ITEM_PURCHASE", 10, -100),
        ("ACHIEVEMENT", 5, 100),
        ("FIRST_REVIEW", 5, 200),
        ("QUIZ_REWARD", 5, 50),
        ("ETC", 5, 30),
    ]
    weights = [a[1] for a in actions]
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())
    target = scale * 10  # 10K user → 100K
    rows = []
    for uid, act in activities.items():
        n = max(0, int(round(target * act / total_act)))
        running = rng.randint(0, 500)
        for _ in range(n):
            action_idx = sample_weighted_idx(rng, weights)
            atype, _, base_amt = actions[action_idx]
            change = base_amt + rng.randint(-5, 5)
            running = max(0, running + change)
            ptype = "EARN" if change >= 0 else "SPEND"
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "point_change": change, "point_after": running,
                "point_type": ptype, "description": f"[seed] {atype}",
                "action_type": atype, "base_amount": abs(base_amt),
                "applied_multiplier": 1.0,
                "created_at": created, "created_by": "seed_py",
            }, rng))
    inserted = insert_batch(engine, t, rows, dry_run=dry_run)
    logger.info("points_history — %d", inserted)
    return inserted


def seed_calendars(engine: Engine, meta: MetaData, rng: random.Random,
                   user_ids: list[str], scale: int, dry_run: bool) -> int:
    """user_calendars."""
    t = meta.tables.get("user_calendars")
    if t is None:
        return 0
    titles = ["주말 영화의 밤", "친구와 만나기", "가족 시간", "데이트", "혼자 영화 감상"]
    target = scale // 2  # 10K → 5K
    base = dt.datetime.now() - dt.timedelta(days=30)
    rows = []
    for _ in range(target):
        uid = rng.choice(user_ids)
        start = base + dt.timedelta(days=rng.randint(0, 60), hours=rng.randint(8, 22))
        end = start + dt.timedelta(hours=rng.randint(1, 4))
        rows.append(fill_required_defaults(t, {
            "user_id": uid, "schedule_title": rng.choice(titles),
            "schedule_description": "[seed] 일정 메모",
            "start_time": start, "end_time": end,
        }, rng))
    inserted = insert_batch(engine, t, rows, dry_run=dry_run)
    logger.info("user_calendars — %d", inserted)
    return inserted


def seed_streak_attendance(engine: Engine, meta: MetaData, rng: random.Random,
                           user_ids: list[str], dry_run: bool) -> int:
    """user_streak_attendance — 유저당 1건 (PK 유저별 streak)."""
    t = meta.tables.get("user_streak_attendance")
    if t is None:
        return 0
    rows = []
    for uid in user_ids:
        rows.append(fill_required_defaults(t, {
            "user_id": uid,
            "streak_check_date": rng.randint(1, 30),
            "reward_point": rng.randint(0, 100),
        }, rng))
    inserted = insert_batch(engine, t, rows, dry_run=dry_run)
    logger.info("user_streak_attendance — %d", inserted)
    return inserted


def seed_user_achievements(engine: Engine, meta: MetaData, rng: random.Random,
                           user_ids: list[str], dry_run: bool) -> int:
    """user_achievements — achievement_type 테이블이 없으면 임의 키."""
    t = meta.tables.get("user_achievements")
    if t is None:
        return 0
    keys = [
        ("REVIEW", "first_review"), ("REVIEW", "review_10"), ("REVIEW", "review_50"),
        ("WATCH", "watch_10"), ("WATCH", "watch_100"),
        ("ATTENDANCE", "streak_7"), ("ATTENDANCE", "streak_30"),
        ("PLAYLIST", "create_playlist"), ("COMMUNITY", "first_post"),
        ("QUIZ", "quiz_correct_10"), ("RECOMMEND", "recommend_10"),
    ]
    base = dt.datetime.now() - dt.timedelta(days=180)
    rows = []
    used = set()
    for uid in user_ids:
        n = rng.randint(0, 5)
        for ach_type, ach_key in rng.sample(keys, min(n, len(keys))):
            if (uid, ach_key) in used:
                continue
            used.add((uid, ach_key))
            achieved = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "achievement_type": ach_type, "achievement_key": ach_key,
                "achievement_type_id": rng.randint(1, 30),
                "achieved_at": achieved, "metadata": json.dumps({}),
                "created_at": achieved, "updated_at": achieved,
            }, rng))
    inserted = insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)
    logger.info("user_achievements — %d", inserted)
    return inserted


def seed_playlists(engine: Engine, meta: MetaData, rng: random.Random,
                   user_ids: list[str], movies: list[str], scale: int, dry_run: bool) -> int:
    """playlist + items + likes + scrap."""
    total = 0
    pl_t = meta.tables.get("playlist")
    if pl_t is None:
        return 0
    base = dt.datetime.now() - dt.timedelta(days=365)
    names = ["주말 영화 모음", "감동 명작", "혼자 보기 좋은", "데이트 영화 BEST",
             "역대 최고 액션", "심야 공포 영화", "분위기 있는 로맨스", "OST 좋은 영화"]

    # playlist (5K)
    target = scale // 2
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())
    rows = []
    for uid, act in activities.items():
        n = min(5, max(0, int(round(target * act / total_act))))
        for _ in range(n):
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(pl_t, {
                "user_id": uid,
                "playlist_name": rng.choice(names) + f" #{rng.randint(1, 99)}",
                "description": "[seed] 시드 플레이리스트",
                "is_public": True, "like_count": 0,
                "is_imported": False, "is_deleted": False,
                "created_at": created, "updated_at": created,
            }, rng))
    total += insert_batch(engine, pl_t, rows, dry_run=dry_run)

    if dry_run:
        logger.info("playlists (dry-run) — %d", total)
        return total

    # playlist_id 회수
    with engine.connect() as conn:
        pl_ids = [r[0] for r in conn.execute(text(
            "SELECT playlist_id FROM playlist WHERE user_id LIKE :p"
        ), {"p": USER_ID_PREFIX + "%"}).fetchall()]

    # playlist_item — 평균 6 items / playlist (UNIQUE 가능)
    t = meta.tables.get("playlist_item")
    if t is not None and pl_ids and movies:
        rows = []
        for pid in pl_ids:
            n = rng.randint(3, 12)
            used = set()
            for so in range(n):
                mid = movies[sample_zipf_idx(rng, len(movies), 0.85)]
                if mid in used:
                    continue
                used.add(mid)
                rows.append(fill_required_defaults(t, {
                    "playlist_id": pid, "movie_id": mid, "sort_order": so,
                    "added_at": dt.datetime.now(),
                }, rng))
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # playlist_likes (UNIQUE)
    t = meta.tables.get("playlist_likes")
    if t is not None and pl_ids:
        rows, used = [], set()
        target_l = min(scale, len(pl_ids) * 5)
        attempts = 0
        while len(rows) < target_l and attempts < target_l * 3:
            pid = rng.choice(pl_ids)
            uid = rng.choice(user_ids)
            if (pid, uid) in used:
                attempts += 1
                continue
            used.add((pid, uid))
            rows.append(fill_required_defaults(t, {"playlist_id": pid, "user_id": uid}, rng))
            attempts += 1
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # playlist_scrap (UNIQUE)
    t = meta.tables.get("playlist_scrap")
    if t is not None and pl_ids:
        rows, used = [], set()
        target_s = min(scale // 2, len(pl_ids) * 3)
        attempts = 0
        while len(rows) < target_s and attempts < target_s * 3:
            pid = rng.choice(pl_ids)
            uid = rng.choice(user_ids)
            if (pid, uid) in used:
                attempts += 1
                continue
            used.add((pid, uid))
            rows.append(fill_required_defaults(t, {"user_id": uid, "playlist_id": pid}, rng))
            attempts += 1
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    logger.info("playlists domain — %d", total)
    return total


def seed_courses(engine: Engine, meta: MetaData, rng: random.Random,
                 user_ids: list[str], movies: list[str], scale: int, dry_run: bool) -> int:
    """user_course_progress + course_review + course_verification + course_final_movie.
    user_courses 테이블이 없어 course_id 는 임의 문자열로."""
    total = 0
    if not movies:
        return 0
    base = dt.datetime.now() - dt.timedelta(days=180)

    # 임의 course_id 풀 — 시드 유저별 1~3 개
    user_courses: dict[str, list[str]] = {}
    for uid in user_ids:
        n = rng.randint(0, 3)
        user_courses[uid] = [f"seed-course-{uid}-{i}" for i in range(n)]

    # user_course_progress
    t = meta.tables.get("user_course_progress")
    if t is not None:
        rows = []
        for uid, course_ids in user_courses.items():
            for cid in course_ids:
                total_movies = rng.randint(5, 15)
                verified = rng.randint(0, total_movies)
                progress = round(verified / total_movies * 100, 2) if total_movies else 0
                status = "COMPLETED" if verified == total_movies else "IN_PROGRESS"
                started = sample_uniform_date(rng, base)
                completed = started + dt.timedelta(days=rng.randint(7, 60)) if status == "COMPLETED" else None
                rows.append(fill_required_defaults(t, {
                    "user_id": uid, "course_id": cid,
                    "total_movies": total_movies, "verified_movies": verified,
                    "progress_percent": progress, "status": status,
                    "started_at": started, "completed_at": completed,
                    "reward_granted": status == "COMPLETED",
                    "deadline_at": started + dt.timedelta(days=90),
                    "created_at": started, "updated_at": started,
                }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # course_review (verified_movies 만큼 리뷰 1개씩)
    t = meta.tables.get("course_review")
    if t is not None:
        rows = []
        for uid, course_ids in user_courses.items():
            for cid in course_ids:
                n_reviews = rng.randint(0, 5)
                for _ in range(n_reviews):
                    mid = movies[sample_zipf_idx(rng, len(movies), 0.85)]
                    created = sample_uniform_date(rng, base)
                    rows.append(fill_required_defaults(t, {
                        "course_id": cid[:50], "movie_id": mid, "user_id": uid,
                        "review_text": "[seed] 도장깨기 리뷰",
                        "verified_count": 1, "award_point": 50,
                        "created_at": created, "updated_at": created,
                    }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # course_verification (이미 적재된 course_review 와 1:1)
    if not dry_run:
        t = meta.tables.get("course_verification")
        if t is not None:
            with engine.connect() as conn:
                cr_rows = conn.execute(text("""
                    SELECT course_id, movie_id, user_id FROM course_review WHERE user_id LIKE :p LIMIT 30000
                """), {"p": USER_ID_PREFIX + "%"}).fetchall()
            rows = []
            for cid, mid, uid in cr_rows:
                rows.append(fill_required_defaults(t, {
                    "course_id": cid, "movie_id": mid, "user_id": uid,
                    "verification_type": "AI",
                    "ai_confidence": round(rng.uniform(0.5, 1.0), 4),
                    "is_verified": rng.random() < 0.85,
                    "verified_at": dt.datetime.now(),
                    "similarity_score": round(rng.uniform(0.3, 0.95), 4),
                    "review_status": "AUTO_VERIFIED",
                }, rng))
            total += insert_batch(engine, t, rows, dry_run=dry_run)

    # course_final_movie
    t = meta.tables.get("course_final_movie")
    if t is not None:
        rows = []
        for uid, course_ids in user_courses.items():
            for cid in course_ids:
                if rng.random() > 0.3:
                    continue
                rows.append(fill_required_defaults(t, {
                    "user_id": uid, "course_id": cid[:50],
                    "is_completed": rng.random() < 0.7,
                    "final_review_text": "[seed] 마지막 한 편 후기",
                    "complete_at": dt.datetime.now(),
                }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    logger.info("courses domain — %d", total)
    return total


def seed_quizzes(engine: Engine, meta: MetaData, rng: random.Random,
                 user_ids: list[str], movies: list[str], dry_run: bool) -> int:
    """quizzes(글로벌 100) + quiz_participations + quiz_attempts + quiz_rewards."""
    total = 0
    qz_t = meta.tables.get("quizzes")
    if qz_t is None:
        return 0
    today = dt.date.today()

    # 기존 [seed] 퀴즈 수 확인
    with engine.connect() as conn:
        existing = conn.execute(text(
            "SELECT COUNT(*) FROM quizzes WHERE question LIKE '[seed]%'"
        )).scalar()

    if existing < 100 and not dry_run:
        rows = []
        for i in range(100 - existing):
            mid = movies[sample_zipf_idx(rng, len(movies), 0.7)] if (movies and rng.random() < 0.6) else None
            rows.append(fill_required_defaults(qz_t, {
                "movie_id": mid,
                "question": f"[seed] 시드 퀴즈 #{existing + i + 1}",
                "explanation": "[seed] 시드 해설",
                "correct_answer": "보기 2",
                "options": json.dumps(["보기 1", "보기 2", "보기 3", "보기 4"], ensure_ascii=False),
                "reward_point": 100,
                "status": "PUBLISHED",
                "quiz_date": today - dt.timedelta(days=rng.randint(0, 30)),
                "quiz_type": "DAILY",
            }, rng))
        total += insert_batch(engine, qz_t, rows, dry_run=dry_run)

    # quiz_id 회수
    with engine.connect() as conn:
        quiz_ids = [r[0] for r in conn.execute(text(
            "SELECT quiz_id FROM quizzes WHERE question LIKE '[seed]%' LIMIT 100"
        )).fetchall()]
    if not quiz_ids:
        return total

    # quiz_participations (UNIQUE user, quiz)
    t = meta.tables.get("quiz_participations")
    if t is not None:
        rows, used = [], set()
        target = 5_000
        base = dt.datetime.now() - dt.timedelta(days=90)
        attempts = 0
        while len(rows) < target and attempts < target * 3:
            uid = rng.choice(user_ids)
            qid = rng.choice(quiz_ids)
            if (uid, qid) in used:
                attempts += 1
                continue
            used.add((uid, qid))
            is_correct = rng.random() < 0.6
            sub = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "quiz_id": qid, "user_id": uid,
                "selected_option": "보기 2" if is_correct else f"보기 {rng.randint(1, 4)}",
                "is_correct": is_correct,
                "submitted_at": sub, "created_at": sub, "updated_at": sub,
            }, rng))
            attempts += 1
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # quiz_rewards
    t = meta.tables.get("quiz_rewards")
    if t is not None:
        rows = []
        target = 3_000
        base = dt.datetime.now() - dt.timedelta(days=90)
        for _ in range(target):
            uid = rng.choice(user_ids)
            qid = rng.choice(quiz_ids)
            r = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "quiz_id": qid, "user_id": uid,
                "reward_points": rng.randint(50, 200),
                "rewarded_at": r, "created_at": r, "updated_at": r,
            }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    logger.info("quizzes domain — %d", total)
    return total


def seed_worldcup(engine: Engine, meta: MetaData, rng: random.Random,
                  user_ids: list[str], movies: list[str], scale: int, dry_run: bool) -> int:
    """worldcup_results — sessions/matches 테이블 없음."""
    t = meta.tables.get("worldcup_results")
    if t is None or not movies:
        return 0
    target = scale // 2
    base = dt.datetime.now() - dt.timedelta(days=180)
    rows = []
    for _ in range(target):
        uid = rng.choice(user_ids)
        round_size = rng.choice([16, 32, 64])
        winner = movies[sample_zipf_idx(rng, len(movies), 0.7)]
        runner_up = movies[sample_zipf_idx(rng, len(movies), 0.7)]
        if winner == runner_up:
            continue
        semi = ",".join([movies[rng.randint(0, len(movies)-1)] for _ in range(4)])
        created = sample_uniform_date(rng, base)
        rows.append(fill_required_defaults(t, {
            "user_id": uid, "round_size": round_size,
            "semi_final_movie_ids": semi, "selection_log": "[]", "genre_preferences": "[]",
            "onboarding_completed": True, "reward_granted": True,
            "winner_movie_id": winner, "runner_up_movie_id": runner_up,
            "total_matches": round_size - 1,
            "created_at": created, "updated_at": created,
        }, rng))
    inserted = insert_batch(engine, t, rows, dry_run=dry_run)
    logger.info("worldcup_results — %d", inserted)
    return inserted


def seed_community(engine: Engine, meta: MetaData, rng: random.Random,
                   user_ids: list[str], scale: int, dry_run: bool) -> int:
    """posts + post_comment + post_likes + post_images + comment_likes + post_declaration."""
    total = 0
    posts_t = meta.tables.get("posts")
    if posts_t is None:
        return 0
    base = dt.datetime.now() - dt.timedelta(days=365)

    # posts (5K)
    target = scale // 2
    titles = ["추천 부탁드립니다", "이 영화 어떠세요?", "정말 감동적이었어요",
              "주말에 볼만한 영화", "OTT 어디서 볼 수 있나요?", "베스트 액션 영화 추천"]
    activities = {uid: sample_pareto(rng, 1.16, 1.0) for uid in user_ids}
    total_act = sum(activities.values())
    rows = []
    for uid, act in activities.items():
        n = min(5, max(0, int(round(target * act / total_act))))
        for _ in range(n):
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(posts_t, {
                "user_id": uid,
                "title": rng.choice(titles) + f" #{rng.randint(1, 99)}",
                "content": "[seed] " + rng.choice(["좋은 영화 공유합니다.", "다들 어떻게 생각하시나요?", "추천 부탁드려요!"]),
                "category": sample_weighted(rng, ["FREE", "DISCUSSION", "RECOMMENDATION", "NEWS", "PLAYLIST_SHARE"], [40, 25, 20, 10, 5]),
                "view_count": rng.randint(0, 500),
                "status": "PUBLISHED", "like_count": 0, "comment_count": 0,
                "is_deleted": False, "is_blinded": False,
                "created_at": created, "updated_at": created,
            }, rng))
    total += insert_batch(engine, posts_t, rows, dry_run=dry_run)

    if dry_run:
        logger.info("community (dry-run) — %d", total)
        return total

    # post_id 회수
    with engine.connect() as conn:
        post_ids = [r[0] for r in conn.execute(text(
            "SELECT post_id FROM posts WHERE user_id LIKE :p"
        ), {"p": USER_ID_PREFIX + "%"}).fetchall()]
    if not post_ids:
        return total

    # post_comment
    t = meta.tables.get("post_comment")
    if t is not None:
        rows = []
        for _ in range(scale + scale // 2):  # 15K
            pid = rng.choice(post_ids)
            uid = rng.choice(user_ids)
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "post_id": pid, "user_id": uid,
                "content": "[seed] " + rng.choice(["공감합니다", "저도 그렇게 생각해요", "좋은 글이네요"]),
                "is_deleted": False, "like_count": 0,
                "created_at": created, "updated_at": created,
            }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # post_likes (UNIQUE post, user)
    t = meta.tables.get("post_likes")
    if t is not None:
        rows, used = [], set()
        target_l = scale * 3  # 30K
        attempts = 0
        while len(rows) < target_l and attempts < target_l * 3:
            pid = rng.choice(post_ids)
            uid = rng.choice(user_ids)
            if (pid, uid) in used:
                attempts += 1
                continue
            used.add((pid, uid))
            created = sample_uniform_date(rng, base)
            rows.append(fill_required_defaults(t, {
                "post_id": pid, "user_id": uid,
                "created_at": created, "updated_at": created,
            }, rng))
            attempts += 1
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # post_images (post 30% 정도 이미지 보유)
    t = meta.tables.get("post_images")
    if t is not None:
        rows = []
        for pid in post_ids:
            if rng.random() > 0.3:
                continue
            n = rng.randint(1, 3)
            for so in range(n):
                rows.append(fill_required_defaults(t, {
                    "post_id": pid, "image_url": f"https://placehold.co/600x400?seed={pid}_{so}",
                    "sort_order": so,
                }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # comment_likes — comment_id 회수
    t = meta.tables.get("comment_likes")
    if t is not None:
        with engine.connect() as conn:
            comment_ids = [r[0] for r in conn.execute(text(
                "SELECT post_comment_id FROM post_comment WHERE user_id LIKE :p ORDER BY RAND() LIMIT 5000"
            ), {"p": USER_ID_PREFIX + "%"}).fetchall()]
        if comment_ids:
            rows, used = [], set()
            target_c = min(10_000, len(comment_ids) * 3)
            attempts = 0
            while len(rows) < target_c and attempts < target_c * 3:
                cid = rng.choice(comment_ids)
                uid = rng.choice(user_ids)
                if (cid, uid) in used:
                    attempts += 1
                    continue
                used.add((cid, uid))
                rows.append(fill_required_defaults(t, {"comment_id": cid, "user_id": uid}, rng))
                attempts += 1
            total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # post_declaration (신고)
    t = meta.tables.get("post_declaration")
    if t is not None:
        rows = []
        for _ in range(min(500, scale // 20)):
            pid = rng.choice(post_ids)
            uid = rng.choice(user_ids)
            reported = rng.choice(user_ids)
            rows.append(fill_required_defaults(t, {
                "post_id": pid, "user_id": uid, "reported_user_id": reported,
                "declaration_content": "[seed] 시드 신고",
                "status": "PENDING", "target_type": "POST",
                "toxicity_score": round(rng.uniform(0.3, 0.95), 4),
            }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    logger.info("community domain — %d", total)
    return total


def seed_commerce(engine: Engine, meta: MetaData, rng: random.Random,
                  user_ids: list[str], scale: int, dry_run: bool) -> int:
    """payment_orders + user_subscriptions + point_orders + user_items + ticket lottery + entry."""
    total = 0
    base = dt.datetime.now() - dt.timedelta(days=365)

    # 마스터 풀
    plans = []
    items = []
    with engine.connect() as conn:
        try:
            plans = conn.execute(text("SELECT subscription_plan_id, price FROM subscription_plans WHERE is_active=1")).fetchall()
        except Exception:
            pass
        try:
            items = conn.execute(text("SELECT point_item_id, item_price FROM point_items WHERE is_active=1 LIMIT 50")).fetchall()
        except Exception:
            pass

    # payment_orders (10K)
    t = meta.tables.get("payment_orders")
    if t is not None:
        rows = []
        target = scale  # 10K
        for i in range(target):
            uid = rng.choice(user_ids)
            order_type = sample_weighted(rng, ["SUBSCRIPTION", "POINT_PACK"], [70, 30])
            if order_type == "SUBSCRIPTION" and plans:
                pid = rng.choice(plans)
                plan_id, amount = pid[0], pid[1]
            elif items:
                it = rng.choice(items)
                plan_id, amount = None, it[1]
            else:
                plan_id, amount = None, 5900
            status = sample_weighted(rng, ["COMPLETED", "FAILED", "REFUNDED", "PENDING"], [80, 5, 5, 10])
            order_id = f"seed-{rng.randint(0, 10**18):018x}"[:50]
            created = sample_uniform_date(rng, base)
            completed = created + dt.timedelta(seconds=rng.randint(1, 60)) if status == "COMPLETED" else None
            rows.append(fill_required_defaults(t, {
                "payment_order_id": order_id, "user_id": uid,
                "order_type": order_type, "amount": amount,
                "points_amount": amount if order_type == "POINT_PACK" else None,
                "idempotency_key": f"seed-idem-{order_id}"[:100],
                "status": status,
                "pg_transaction_id": f"seed-pg-{rng.randint(0, 10**9)}" if status == "COMPLETED" else None,
                "pg_provider": "TOSS_PAYMENTS",
                "completed_at": completed,
                "subscription_plan_id": plan_id,
                "created_at": created, "updated_at": created,
            }, rng))
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    # user_subscriptions
    t = meta.tables.get("user_subscriptions")
    if t is not None and plans:
        target = scale // 3  # 3K
        sampled = rng.sample(user_ids, min(target, len(user_ids)))
        rows = []
        for uid in sampled:
            plan = rng.choice(plans)
            status = sample_weighted(rng, ["ACTIVE", "CANCELLED", "EXPIRED"], [60, 20, 20])
            started = sample_uniform_date(rng, base)
            expires = started + dt.timedelta(days=rng.choice([30, 365]))
            cancelled = started + dt.timedelta(days=rng.randint(1, 25)) if status == "CANCELLED" else None
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "subscription_plan_id": plan[0],
                "status": status, "started_at": started, "expires_at": expires,
                "cancelled_at": cancelled,
                "auto_renew": status == "ACTIVE",
                "next_billing_date": expires.date() if status == "ACTIVE" else None,
                "remaining_ai_bonus": rng.randint(0, 60),
            }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # point_orders
    t = meta.tables.get("point_orders")
    if t is not None and items:
        rows = []
        target = scale // 2  # 5K
        for _ in range(target):
            uid = rng.choice(user_ids)
            it = rng.choice(items)
            cnt = rng.randint(1, 5)
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "point_item_id": it[0],
                "item_count": cnt, "used_point": it[1] * cnt,
                "status": "COMPLETED",
            }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # user_items
    t = meta.tables.get("user_items")
    if t is not None and items:
        rows = []
        target = (scale * 4) // 5  # 8K
        for _ in range(target):
            uid = rng.choice(user_ids)
            it = rng.choice(items)
            acquired = sample_uniform_date(rng, base)
            status = sample_weighted(rng, ["ACTIVE", "USED", "EXPIRED"], [60, 30, 10])
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "point_item_id": it[0],
                "acquired_at": acquired,
                "expires_at": acquired + dt.timedelta(days=rng.choice([30, 90, 365])),
                "used_at": acquired + dt.timedelta(days=rng.randint(1, 30)) if status == "USED" else None,
                "status": status, "source": "POINT_EXCHANGE",
                "remaining_quantity": rng.randint(1, 5),
            }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # movie_ticket_lottery (10 cycles)
    t = meta.tables.get("movie_ticket_lottery")
    lottery_ids = []
    if t is not None and not dry_run:
        rows = []
        today = dt.date.today()
        for i in range(10):
            ym = (today - dt.timedelta(days=30 * i)).strftime("%Y-%m")
            rows.append(fill_required_defaults(t, {
                "cycle_year_month": ym,
                "status": "COMPLETED" if i > 0 else "PENDING",
                "winner_count": 5,
                "drawn_at": dt.datetime.now() if i > 0 else None,
            }, rng))
        total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

        with engine.connect() as conn:
            lottery_ids = [r[0] for r in conn.execute(text(
                "SELECT lottery_id FROM movie_ticket_lottery ORDER BY lottery_id DESC LIMIT 10"
            )).fetchall()]

    # movie_ticket_entry (응모) — user_item_id 필요
    t = meta.tables.get("movie_ticket_entry")
    if t is not None and lottery_ids and not dry_run:
        with engine.connect() as conn:
            user_item_ids = [r[0] for r in conn.execute(text(
                "SELECT user_item_id FROM user_items WHERE user_id LIKE :p ORDER BY RAND() LIMIT 3000"
            ), {"p": USER_ID_PREFIX + "%"}).fetchall()]
        if user_item_ids:
            rows = []
            for uiid in user_item_ids:
                lid = rng.choice(lottery_ids)
                # user_id 회수
                with engine.connect() as conn:
                    uid_row = conn.execute(text(
                        "SELECT user_id FROM user_items WHERE user_item_id = :i"
                    ), {"i": uiid}).scalar()
                if not uid_row:
                    continue
                rows.append(fill_required_defaults(t, {
                    "user_id": uid_row, "lottery_id": lid, "user_item_id": uiid,
                    "status": sample_weighted(rng, ["PENDING", "WON", "LOST"], [40, 5, 55]),
                }, rng))
            # 처음 1K 만 — 더 효율적인 방법 (위 1:1 SELECT 가 느림)
            total += insert_batch(engine, t, rows[:1000], dry_run=dry_run)

    logger.info("commerce domain — %d", total)
    return total


def seed_support(engine: Engine, meta: MetaData, rng: random.Random,
                 user_ids: list[str], scale: int, dry_run: bool) -> int:
    """support_tickets + ticket_replies + support_faq_feedback."""
    total = 0
    base = dt.datetime.now() - dt.timedelta(days=180)

    # support_tickets (3K)
    t = meta.tables.get("support_tickets")
    ticket_ids = []
    if t is not None:
        target = scale // 3  # 3K
        titles = ["결제 환불 문의", "계정 비밀번호 분실", "추천 결과가 이상해요",
                  "포인트 잔액 오류", "앱이 자꾸 꺼져요", "기능 추가 건의"]
        rows = []
        for _ in range(target):
            uid = rng.choice(user_ids)
            cat = sample_weighted(rng, ["GENERAL", "ACCOUNT", "PAYMENT", "RECOMMENDATION", "CHAT", "COMMUNITY"], [30, 20, 20, 10, 10, 10])
            status = sample_weighted(rng, ["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"], [20, 20, 40, 20])
            created = sample_uniform_date(rng, base)
            resolved = created + dt.timedelta(hours=rng.randint(1, 72)) if status in ("RESOLVED", "CLOSED") else None
            rows.append(fill_required_defaults(t, {
                "user_id": uid, "category": cat, "title": rng.choice(titles),
                "content": "[seed] 시드 문의 본문",
                "status": status, "priority": sample_weighted(rng, ["LOW", "MEDIUM", "HIGH"], [50, 35, 15]),
                "resolved_at": resolved,
                "closed_at": resolved if status == "CLOSED" else None,
                "created_at": created, "updated_at": created,
            }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

        if not dry_run:
            with engine.connect() as conn:
                ticket_ids = [(r[0], r[1]) for r in conn.execute(text(
                    "SELECT ticket_id, user_id FROM support_tickets WHERE user_id LIKE :p"
                ), {"p": USER_ID_PREFIX + "%"}).fetchall()]

    # ticket_replies (각 티켓 1~2 답변)
    t = meta.tables.get("ticket_replies")
    if t is not None and ticket_ids:
        rows = []
        for tid, uid in ticket_ids:
            n = rng.randint(0, 2)
            for i in range(n):
                is_admin = i == 0
                rows.append(fill_required_defaults(t, {
                    "ticket_id": tid,
                    "author_id": "admin" if is_admin else uid,
                    "author_type": "ADMIN" if is_admin else "USER",
                    "content": "[seed] 답변 내용",
                }, rng))
        total += insert_batch(engine, t, rows, dry_run=dry_run)

    # support_faq_feedback (UNIQUE faq, user)
    t = meta.tables.get("support_faq_feedback")
    if t is not None:
        with engine.connect() as conn:
            faqs = []
            try:
                faqs = [r[0] for r in conn.execute(text("SELECT faq_id FROM support_faqs LIMIT 50")).fetchall()]
            except Exception:
                pass
        if faqs:
            rows, used = [], set()
            target = scale // 3  # 3K
            attempts = 0
            while len(rows) < target and attempts < target * 3:
                fid = rng.choice(faqs)
                uid = rng.choice(user_ids)
                if (fid, uid) in used:
                    attempts += 1
                    continue
                used.add((fid, uid))
                rows.append(fill_required_defaults(t, {
                    "faq_id": fid, "user_id": uid, "helpful": rng.random() < 0.7,
                }, rng))
                attempts += 1
            total += insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)

    logger.info("support domain — %d", total)
    return total


def seed_chat_archive(engine: Engine, meta: MetaData, rng: random.Random,
                      user_ids: list[str], scale: int, dry_run: bool) -> int:
    """chat_session_archive (UNIQUE session_id)."""
    t = meta.tables.get("chat_session_archive")
    if t is None:
        return 0
    target = scale * 2  # 20K
    base = dt.datetime.now() - dt.timedelta(days=180)
    rows = []
    for i in range(target):
        uid = rng.choice(user_ids)
        sid = f"seed-chat-{i:08d}-{rng.randint(0, 10**6):06d}"[:36]
        started = sample_uniform_date(rng, base)
        ended = started + dt.timedelta(minutes=rng.randint(1, 30))
        rows.append(fill_required_defaults(t, {
            "user_id": uid, "session_id": sid,
            "messages": json.dumps([
                {"role": "user", "content": "추천 영화 알려줘"},
                {"role": "assistant", "content": "이번 주 인기작을 추천드릴게요."},
            ], ensure_ascii=False),
            "turn_count": rng.randint(1, 10),
            "session_state": json.dumps({}), "intent_summary": json.dumps({}),
            "started_at": started, "ended_at": ended,
            "title": "시드 채팅 세션", "last_message_at": ended,
            "is_active": False, "is_deleted": False,
            "recommended_movie_count": rng.randint(0, 10),
            "created_at": started, "updated_at": ended,
        }, rng))
    inserted = insert_batch(engine, t, rows, ignore_duplicates=True, dry_run=dry_run)
    logger.info("chat_session_archive — %d", inserted)
    return inserted


def seed_admin_audit(engine: Engine, meta: MetaData, rng: random.Random,
                     user_ids: list[str], dry_run: bool) -> int:
    """admin_audit_logs."""
    t = meta.tables.get("admin_audit_logs")
    if t is None:
        return 0
    actions = ["USER_GRANT_POINT", "USER_SUSPEND", "TICKET_REPLY", "FAQ_PUBLISH",
               "NOTICE_CREATE", "BANNER_UPDATE", "QUIZ_PUBLISH"]
    base = dt.datetime.now() - dt.timedelta(days=90)
    rows = []
    for _ in range(1_000):
        action = rng.choice(actions)
        target_id = rng.choice(user_ids)
        created = sample_uniform_date(rng, base)
        rows.append(fill_required_defaults(t, {
            "admin_id": None, "action_type": action,
            "target_type": "USER", "target_id": target_id,
            "description": f"[seed] {action}",
            "ip_address": "127.0.0.1",
            "before_data": json.dumps({}), "after_data": json.dumps({}),
            "created_at": created, "updated_at": created,
            "created_by": "seed_py",
        }, rng))
    inserted = insert_batch(engine, t, rows, dry_run=dry_run)
    logger.info("admin_audit_logs — %d", inserted)
    return inserted


# ============================================================
# 마스터 로드 (movies / 유저 풀)
# ============================================================


def load_movie_ids(engine: Engine, limit: int) -> list[str]:
    """popularity_score 정렬된 movie_id 풀 (NULL 제외)."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT movie_id FROM movies WHERE popularity_score IS NOT NULL "
            "ORDER BY popularity_score DESC LIMIT :n"
        ), {"n": limit}).fetchall()
    return [r[0] for r in rows]


def load_seed_user_ids(engine: Engine) -> list[str]:
    """이미 적재된 시드 유저 ID 회수."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT user_id FROM users WHERE user_id LIKE :p ORDER BY user_id"
        ), {"p": USER_ID_PREFIX + "%"}).fetchall()
    return [r[0] for r in rows]


# ============================================================
# 보정 SQL (finalize)
# ============================================================


def finalize_corrections(engine: Engine, dry_run: bool) -> int:
    """user_points.balance ↔ points_history SUM 동기화 등 정합성 보정."""
    if dry_run:
        return 0
    total = 0
    pattern = USER_ID_PREFIX + "%"

    # reviews.like_count 동기화
    with engine.begin() as conn:
        try:
            r = conn.execute(text("""
                UPDATE reviews r
                LEFT JOIN (SELECT review_id, COUNT(*) cnt FROM review_likes GROUP BY review_id) rl
                  ON r.review_id = rl.review_id
                SET r.like_count = COALESCE(rl.cnt, r.like_count)
                WHERE r.user_id LIKE :p
            """), {"p": pattern})
            total += r.rowcount or 0
        except Exception as e:
            logger.warning("reviews.like_count 동기화 skip: %s", e)

    logger.info("finalize 보정 — %d rows", total)
    return total


# ============================================================
# 롤백 — user_id 마커 기반
# ============================================================

#: 의존 역순으로 삭제. user_id 컬럼 있는 테이블만 대상.
ROLLBACK_TABLES_BY_USER_ID = [
    "admin_audit_logs",  # admin_id FK 일 수 있음 — graceful skip
    "toxicity_logs",
    "event_logs", "search_history", "chat_session_archive",
    "points_history",
    "recommendation_impact", "recommendation_feedback", "recommendation_log",
    "user_implicit_rating", "user_behavior_profile",
    "post_declaration", "comment_likes", "post_like", "post_comment", "posts",
    "worldcup_results", "worldcup_sessions",
    "quiz_rewards", "quiz_attempts", "quiz_participations",
    "playlist_scrap", "playlist_likes", "playlist",
    "user_courses",
    "user_streak_attendance", "user_attendance", "user_activity_progress",
    "user_achievements", "user_calendars",
    "movie_ticket_entry", "user_subscriptions", "payment_orders",
    "point_orders", "user_items",
    "support_tickets", "support_faq_feedback",
    "review_votes", "review_likes", "reviews",
    "user_watch_history", "user_wishlist", "likes",
    "fav_movie", "fav_directors", "fav_actors", "fav_genre",
    "user_preferences", "user_ai_quota", "user_points", "user_status",
    "users",
]


def rollback(engine: Engine) -> int:
    """user_id LIKE 'seed_%' 마커 기반 일괄 DELETE."""
    logger.warning("=" * 60)
    logger.warning("ROLLBACK 시작")
    logger.warning("=" * 60)

    with engine.connect() as conn:
        cnt = conn.execute(text(
            "SELECT COUNT(*) FROM users WHERE user_id LIKE :p"
        ), {"p": USER_ID_PREFIX + "%"}).scalar() or 0
    logger.warning("삭제 대상 시드 유저 수: %d", cnt)
    if cnt == 0:
        logger.warning("삭제할 시드 데이터 없음")
        return 0

    total = 0
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        try:
            for tbl in ROLLBACK_TABLES_BY_USER_ID:
                try:
                    r = conn.execute(text(
                        f"DELETE FROM `{tbl}` WHERE user_id LIKE :p"
                    ), {"p": USER_ID_PREFIX + "%"})
                    if r.rowcount > 0:
                        logger.info("DELETE %s — %d", tbl, r.rowcount)
                    total += r.rowcount or 0
                except Exception as e:
                    logger.warning("DELETE %s skip: %s", tbl, type(e).__name__)
        finally:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

    logger.warning("=" * 60)
    logger.warning("ROLLBACK 완료 — 총 %d rows 삭제", total)
    logger.warning("=" * 60)
    return total


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="몽글픽 운영 DB 데모 데이터 시드")
    p.add_argument("--db-url", required=True,
                   help="SQLAlchemy URL (예: mysql+pymysql://user:pw@host:3306/db)")
    p.add_argument("--user-count", type=int, default=10_000)
    p.add_argument("--review-count", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=1_000)
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--dry-run", action="store_true", help="실제 INSERT 없이 카운트만")
    p.add_argument("--rollback", action="store_true", help="시드 데이터 롤백")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    engine = create_engine(args.db_url, pool_pre_ping=True)
    rng = random.Random(args.random_seed)
    Faker.seed(args.random_seed)
    faker = Faker("ko_KR")

    if args.rollback:
        rollback(engine)
        return 0

    logger.info("===== 몽글픽 데모 시드 =====")
    logger.info("user_count=%d review_count=%d dry_run=%s", args.user_count, args.review_count, args.dry_run)

    meta = reflect_db(engine)

    movies = load_movie_ids(engine, limit=10_000)
    if not movies:
        logger.error("movies 테이블이 비어있습니다 — 영화 적재 먼저 필요")
        return 1
    logger.info("movies 풀 — %d 개", len(movies))

    # 1) users + 부속 — user_count=0 이면 기존 시드 유저 재사용 (추가 도메인 적재용)
    if args.user_count > 0:
        user_ids = seed_users(engine, meta, faker, rng, args.user_count, args.dry_run)
    else:
        user_ids = load_seed_user_ids(engine)
        logger.info("기존 시드 유저 재사용 — %d 명", len(user_ids))

    if not user_ids:
        # dry-run 에서 첫 호출이거나 시드 유저 없는 경우
        logger.warning("user_ids 비어있음 — 추가 도메인 시뮬레이션만 가능")
    elif args.user_count == 0:
        # 기존 유저 재사용 시 분포 scale 기준 자동 설정
        args.user_count = len(user_ids)
        logger.info("user_count=0 → 기존 시드 유저 수 %d 로 분포 scale 기준 설정", args.user_count)

    # 2) reviews + watch_history
    seed_reviews_and_history(engine, meta, rng, user_ids, movies, args.review_count, args.dry_run)

    # 3) wishlist + likes
    seed_engagement_actions(engine, meta, rng, user_ids, movies,
                            wishlist_target=args.user_count * 5,
                            like_target=args.user_count * 10,
                            dry_run=args.dry_run)

    # 4) 로그성 (event_logs / search_history)
    seed_logs(engine, meta, rng, user_ids, movies,
              event_target=args.user_count * 10,
              search_target=args.user_count * 3,
              dry_run=args.dry_run)

    # 5) 출석
    seed_attendance(engine, meta, rng, user_ids, target=args.user_count * 3, dry_run=args.dry_run)

    # === Phase 2: 전 도메인 ===
    # 6) 즐겨찾기 (fav_*)
    seed_favorites(engine, meta, rng, user_ids, movies, args.dry_run)
    # 7) 리뷰 좋아요/투표
    seed_review_engagement(engine, meta, rng, user_ids, args.dry_run)
    # 8) 추천 도메인 (log/impact/implicit/behavior)
    seed_recommendation(engine, meta, rng, user_ids, movies, args.user_count, args.dry_run)
    # 9) 포인트 이력 (action 분포)
    seed_points_history(engine, meta, rng, user_ids, args.user_count, args.dry_run)
    # 10) 캘린더
    seed_calendars(engine, meta, rng, user_ids, args.user_count, args.dry_run)
    # 11) 출석 streak
    seed_streak_attendance(engine, meta, rng, user_ids, args.dry_run)
    # 12) 업적
    seed_user_achievements(engine, meta, rng, user_ids, args.dry_run)
    # 13) 플레이리스트
    seed_playlists(engine, meta, rng, user_ids, movies, args.user_count, args.dry_run)
    # 14) 도장깨기 코스
    seed_courses(engine, meta, rng, user_ids, movies, args.user_count, args.dry_run)
    # 15) 퀴즈
    seed_quizzes(engine, meta, rng, user_ids, movies, args.dry_run)
    # 16) 이상형월드컵 (results 만)
    seed_worldcup(engine, meta, rng, user_ids, movies, args.user_count, args.dry_run)
    # 17) 커뮤니티
    seed_community(engine, meta, rng, user_ids, args.user_count, args.dry_run)
    # 18) 결제/구독/아이템/추첨
    seed_commerce(engine, meta, rng, user_ids, args.user_count, args.dry_run)
    # 19) 지원
    seed_support(engine, meta, rng, user_ids, args.user_count, args.dry_run)
    # 20) 채팅 아카이브
    seed_chat_archive(engine, meta, rng, user_ids, args.user_count, args.dry_run)
    # 21) 관리자 감사 로그
    seed_admin_audit(engine, meta, rng, user_ids, args.dry_run)

    # 99) 보정
    finalize_corrections(engine, args.dry_run)

    logger.info("===== 시드 완료 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
