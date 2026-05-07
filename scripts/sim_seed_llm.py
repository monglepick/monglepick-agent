"""몽글픽 LLM/Payment 도메인 SQL 시드 — 가상 사용자(sim_*) 대상 직접 INSERT 스크립트.

이 스크립트는 sim_users.py (API 시뮬) 가 채울 수 없는 LLM/Payment 도메인 행을 직접
SQL INSERT 한다. Solar/Ollama 호출 비용을 회피하면서도 운영 화면(추천 통계, 챗봇
대시보드, CF/CBF 신호) 의 실제 데이터 흐름을 모방한다.

채우는 테이블:
  1. recommendation_log         — AI 추천 1건당 N개 영화 추천 흔적
  2. recommendation_impact      — 추천 → click/wishlist/watch/rate 전환 추적
  3. chat_session_archive       — 챗봇 세션 + JSON messages
  4. user_implicit_rating       — CF/CBF 신호 (시청/리뷰/위시 종합)
  5. user_behavior_profile      — 사용자 페르소나 프로필
  6. reviews 사후 보정          — review_source / review_category_code 분포 보정
                                  (60% rec_log / 15% chat / 15% detail / 5% course /
                                   3% wishlist / 2% worldcup)

설계 결정:
  - 대상: sim_users (`nickname LIKE 'sim_%'`) 만. 운영 사용자 데이터는 절대 건드리지 않음.
  - 결정성: random_seed 고정 (default 20260507).
  - 시간 분포: 각 sim_user 의 created_at 부터 NOW() 사이 균등 분포.
  - 인과 순서: recommendation_log.created_at < recommendation_impact.created_at <
              해당 review.created_at (review_source=rec_log_<id> 인 경우).

사용:
  python sim_seed_llm.py --db-url "mysql+pymysql://user:pw@host:3306/db" \
                          --random-seed 20260507 [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sys
import uuid
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

import pymysql

logger = logging.getLogger("sim_seed_llm")

# ============================================================
# 분포·상수
# ============================================================

# 페르소나 별 평균 추천 로그 수 (운영 1주일 사용량 모방)
PERSONA_RECLOGS = {"heavy": (40, 120), "casual": (10, 40), "lurker": (0, 8)}
# 페르소나 별 챗봇 세션 수
PERSONA_CHATS = {"heavy": (3, 10), "casual": (1, 4), "lurker": (0, 1)}
# Impact 발생률 (recommendation_log 1행 당)
IMPACT_RATE = 0.30
# Impact 액션별 확률 (cumulative — 더 강한 액션이 약한 액션 포함)
P_CLICKED = 0.85
P_DETAIL = 0.60
P_WISHLISTED = 0.20
P_WATCHED = 0.15
P_RATED = 0.10
P_DISMISSED = 0.05

# 영화 추천 reason 풀 (운영 explanation_generator 노드 출력 모방)
REASON_TEMPLATES = [
    "{genre} 장르를 즐겨보신 이력과 취향 중심 키워드 ‘{kw}’ 가 유사도 0.{sim:02d} 으로 일치합니다.",
    "최근 시청하신 작품들과 분위기 ‘{mood}’ 가 잘 어울려 추천드립니다 (CBF 점수 0.{sim:02d}).",
    "비슷한 취향의 사용자들이 평점 4점 이상으로 평가한 작품으로, 매칭 점수 0.{sim:02d} 입니다.",
    "선호 감독·배우 가중치(genre={genre}) 와 무드(‘{mood}’) 매칭 결과 상위 추천 작품입니다.",
]
GENRE_POOL = ["액션", "드라마", "코미디", "로맨스", "스릴러", "SF", "애니메이션", "공포", "다큐"]
MOOD_POOL = ["몽글몽글", "긴장감 넘치는", "잔잔한", "유쾌한", "감동적인", "신비로운", "강렬한"]
KEYWORD_POOL = ["복수", "성장", "우정", "사랑", "가족", "음악", "여행", "전쟁", "미스터리"]

# 운영 review_category_code enum (백엔드 ReviewCategoryCode.java)
REVIEW_CATEGORY_CODES = [
    "AI_RECOMMEND", "COURSE", "PLAYLIST", "THEATER_RECEIPT", "WISHLIST", "WORLDCUP",
]
# review_source 분포 (운영 흐름 모방)
REVIEW_SOURCE_DIST = [
    ("rec_log", 60),     # AI 추천 → 시청 → 리뷰 (60%)
    ("chat", 15),        # 챗봇 추천 → 시청 → 리뷰
    ("detail", 15),      # 영화 상세 직접 진입 → 리뷰 (review_source=NULL/'detail')
    ("course", 5),       # 도장깨기 코스 → 리뷰
    ("wishlist", 3),     # 위시리스트 → 리뷰
    ("worldcup", 2),     # 이상형월드컵 → 리뷰
]


# ============================================================
# DB 헬퍼
# ============================================================


def parse_db_url(db_url: str) -> dict[str, Any]:
    """SQLAlchemy 형식 URL 을 pymysql 인자로 변환."""
    parsed = urlparse(db_url.replace("mysql+pymysql", "mysql"))
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 3306,
        "user": parsed.username or "monglepick",
        "password": parsed.password or "",
        "database": parsed.path.lstrip("/") or "monglepick",
        "charset": "utf8mb4",
    }


def fetch_sim_users(conn: pymysql.connections.Connection) -> list[tuple[str, dt.datetime]]:
    """sim_* nickname 사용자의 (user_id, created_at) 풀."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, created_at FROM users "
            "WHERE nickname LIKE 'sim_%' ORDER BY created_at"
        )
        return list(cur.fetchall())


def fetch_movie_pool(conn: pymysql.connections.Connection, limit: int = 5000) -> list[str]:
    """popularity 기반 movie_id 풀 — 추천 후보."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT movie_id FROM movies WHERE popularity_score IS NOT NULL "
            "ORDER BY popularity_score DESC LIMIT %s",
            (limit,),
        )
        return [r[0] for r in cur.fetchall()]


def fetch_existing_rec_log_ids(
    conn: pymysql.connections.Connection,
) -> dict[str, list[int]]:
    """기존 system_seed rec_log 의 user_id → [recommendation_log_id, ...] 매핑.

    --skip-rec-log 와 함께 호출 시 user_to_log_ids 풀을 DB 에서 복원해
    [6] reviews 사후 보정 단계에서 review_source=rec_log_<id> 흐름이 정상 작동하게 한다.
    """
    user_to_log_ids: dict[str, list[int]] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, recommendation_log_id FROM recommendation_log "
            "WHERE created_by = 'system_seed' ORDER BY user_id, recommendation_log_id"
        )
        for user_id, log_id in cur.fetchall():
            user_to_log_ids.setdefault(user_id, []).append(log_id)
    return user_to_log_ids


def fetch_sim_review_pairs(conn: pymysql.connections.Connection) -> list[tuple[int, str, str]]:
    """sim_users 의 (review_id, user_id, movie_id) 풀 — review_source 보정용."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.review_id, r.user_id, r.movie_id "
            "FROM reviews r JOIN users u ON u.user_id = r.user_id "
            "WHERE u.nickname LIKE 'sim_%' AND r.is_deleted = b'0' "
            "ORDER BY r.review_id"
        )
        return list(cur.fetchall())


def random_dt_between(rng: random.Random, start: dt.datetime, end: dt.datetime) -> dt.datetime:
    """start ~ end 사이 균등분포 datetime."""
    if end <= start:
        return start
    delta = (end - start).total_seconds()
    return start + dt.timedelta(seconds=rng.uniform(0, delta))


def sample_persona(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.20:
        return "heavy"
    if r < 0.50:
        return "casual"
    return "lurker"


def sample_distribution(rng: random.Random, dist: list[tuple[Any, int]]) -> Any:
    """가중치 분포에서 1개 선택."""
    total = sum(w for _, w in dist)
    r = rng.random() * total
    acc = 0
    for v, w in dist:
        acc += w
        if r < acc:
            return v
    return dist[-1][0]


# ============================================================
# 1. recommendation_log
# ============================================================


def insert_recommendation_logs(
    conn: pymysql.connections.Connection, sim_users: list[tuple[str, dt.datetime]],
    movies: list[str], rng: random.Random, now: dt.datetime, dry_run: bool = False,
) -> dict[str, list[int]]:
    """페르소나 분포 기반으로 sim_user 별 recommendation_log INSERT.

    Returns
    -------
    {user_id: [recommendation_log_id, ...]} — 후속 review_source 보정용.
    """
    user_to_log_ids: dict[str, list[int]] = {}
    insert_sql = (
        "INSERT INTO recommendation_log "
        "(created_at, updated_at, created_by, updated_by, "
        " cbf_score, cf_score, clicked, genre_match, hybrid_score, model_version, "
        " mood_match, rank_position, reason, response_time_ms, score, session_id, "
        " user_id, user_intent, movie_id, source_type) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
        "        %s, %s, %s, %s)"
    )
    with conn.cursor() as cur:
        for user_id, user_created_at in sim_users:
            persona = sample_persona(rng)
            lo, hi = PERSONA_RECLOGS[persona]
            n = rng.randint(lo, hi)
            if n == 0:
                user_to_log_ids[user_id] = []
                continue

            # 추천 1세션당 5~10개 영화 → 세션 수는 n / avg(7)
            session_count = max(1, n // 7)
            log_ids: list[int] = []
            for _ in range(session_count):
                # 세션 내 모든 추천 로그는 같은 session_id, created_at 공유
                session_id = str(uuid.uuid4())
                ts = random_dt_between(rng, user_created_at, now)
                rank_count = rng.randint(5, 10)
                for rank in range(rank_count):
                    movie_id = movies[rng.randint(0, len(movies) - 1)]
                    cf = round(rng.uniform(0.20, 0.95), 4)
                    cbf = round(rng.uniform(0.30, 0.95), 4)
                    genre_match = round(rng.uniform(0.4, 1.0), 4)
                    mood_match = round(rng.uniform(0.3, 1.0), 4)
                    hybrid = round(0.6 * cf + 0.4 * cbf, 4)
                    score = round(hybrid * (1.0 - rank * 0.05), 4)
                    clicked = 1 if rng.random() < (0.5 - rank * 0.04) else 0
                    template = rng.choice(REASON_TEMPLATES)
                    reason = template.format(
                        genre=rng.choice(GENRE_POOL),
                        kw=rng.choice(KEYWORD_POOL),
                        mood=rng.choice(MOOD_POOL),
                        sim=rng.randint(40, 95),
                    )
                    user_intent = rng.choice([
                        "최근 시청 이력 기반 추천",
                        "선호 장르 중심 추천",
                        "비슷한 취향 사용자 추천",
                        "기분 전환용 가벼운 영화",
                        "감동적인 작품 추천",
                    ])
                    args = (
                        ts, ts, "system_seed", "system_seed",
                        cbf, cf, clicked, genre_match, hybrid, "v3.4",
                        mood_match, rank + 1, reason, rng.randint(120, 800),
                        score, session_id, user_id, user_intent,
                        movie_id, "AI_RECOMMEND",
                    )
                    if not dry_run:
                        cur.execute(insert_sql, args)
                        log_ids.append(cur.lastrowid)
            user_to_log_ids[user_id] = log_ids
    if not dry_run:
        conn.commit()
    return user_to_log_ids


# ============================================================
# 2. recommendation_impact
# ============================================================


def insert_recommendation_impacts(
    conn: pymysql.connections.Connection, user_to_log_ids: dict[str, list[int]],
    rng: random.Random, dry_run: bool = False,
) -> int:
    """recommendation_log 의 IMPACT_RATE 비율로 impact INSERT."""
    insert_sql = (
        "INSERT INTO recommendation_impact "
        "(created_at, updated_at, created_by, updated_by, "
        " clicked, detail_viewed, movie_id, rated, recommendation_position, "
        " time_to_click_seconds, user_id, watched, wishlisted, recommendation_log_id, "
        " dismissed) "
        "SELECT NOW(6), NOW(6), 'system_seed', 'system_seed', "
        "       %s, %s, l.movie_id, %s, l.rank_position, %s, l.user_id, %s, %s, "
        "       l.recommendation_log_id, %s "
        "FROM recommendation_log l WHERE l.recommendation_log_id = %s"
    )
    inserted = 0
    with conn.cursor() as cur:
        for user_id, log_ids in user_to_log_ids.items():
            for log_id in log_ids:
                if rng.random() >= IMPACT_RATE:
                    continue
                clicked = 1 if rng.random() < P_CLICKED else 0
                detail_viewed = 1 if clicked and rng.random() < P_DETAIL else 0
                wishlisted = 1 if detail_viewed and rng.random() < P_WISHLISTED else 0
                watched = 1 if detail_viewed and rng.random() < P_WATCHED else 0
                rated = 1 if watched and rng.random() < P_RATED else 0
                dismissed = 1 if not clicked and rng.random() < P_DISMISSED else 0
                ttc = rng.randint(2, 60) if clicked else None
                args = (
                    clicked, detail_viewed, rated, ttc, watched, wishlisted, dismissed, log_id,
                )
                if not dry_run:
                    cur.execute(insert_sql, args)
                    inserted += 1
    if not dry_run:
        conn.commit()
    return inserted


# ============================================================
# 3. chat_session_archive
# ============================================================


def _generate_chat_messages(rng: random.Random, persona: str) -> tuple[list[dict], int]:
    """챗봇 세션 1건의 messages JSON + turn_count 생성."""
    turn_count = rng.randint(2, 8) if persona != "lurker" else rng.randint(1, 3)
    messages = []
    user_questions = [
        "오늘 볼만한 영화 추천해줘", "잔잔한 분위기 영화 뭐가 좋을까?",
        "주말에 가족이랑 볼 영화 추천해줘", "최근에 본 영화랑 비슷한 거 추천",
        "스릴러 좋아하는데 추천해줘", "이번 주 신작 알려줘",
        "감독별로 영화 추천 가능해?", "넷플릭스에서 인기 있는 거",
    ]
    bot_responses = [
        "취향에 맞는 영화 5편을 추천해드릴게요!",
        "최근 시청 이력 기반으로 분위기가 비슷한 작품들이에요.",
        "이런 작품들은 어떠신가요?",
        "추가로 보고 싶은 분위기가 있으신가요?",
        "원하시는 감독이나 배우가 있으면 알려주세요.",
    ]
    for _ in range(turn_count):
        messages.append({"role": "user", "content": rng.choice(user_questions),
                         "ts": dt.datetime.now().isoformat()})
        messages.append({"role": "assistant", "content": rng.choice(bot_responses),
                         "ts": dt.datetime.now().isoformat()})
    return messages, turn_count


def insert_chat_archives(
    conn: pymysql.connections.Connection, sim_users: list[tuple[str, dt.datetime]],
    rng: random.Random, now: dt.datetime, dry_run: bool = False,
) -> int:
    """sim_users 별 chat_session_archive INSERT."""
    insert_sql = (
        "INSERT INTO chat_session_archive "
        "(created_at, updated_at, created_by, updated_by, "
        " ended_at, intent_summary, is_active, is_deleted, last_message_at, "
        " messages, recommended_movie_count, session_id, session_state, "
        " started_at, title, turn_count, user_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    inserted = 0
    with conn.cursor() as cur:
        for user_id, user_created_at in sim_users:
            persona = sample_persona(rng)
            lo, hi = PERSONA_CHATS[persona]
            n = rng.randint(lo, hi)
            for _ in range(n):
                started_at = random_dt_between(rng, user_created_at, now)
                # 세션 길이 5분 ~ 1시간
                ended_at = started_at + dt.timedelta(minutes=rng.randint(5, 60))
                if ended_at > now:
                    ended_at = now
                messages, turn_count = _generate_chat_messages(rng, persona)
                rec_count = rng.randint(0, 8)
                intent_summary = {
                    "primary_intent": rng.choice(["recommend", "search", "general"]),
                    "topics": rng.sample(KEYWORD_POOL, k=min(3, len(KEYWORD_POOL))),
                    "confidence": round(rng.uniform(0.6, 0.95), 3),
                }
                session_state = {
                    "current_turn": turn_count,
                    "last_intent": intent_summary["primary_intent"],
                    "recommended_movies": rec_count,
                }
                title = rng.choice([
                    "오늘의 영화 추천", "주말 영화 고르기", "분위기별 영화 탐색",
                    "최근 시청작과 비슷한 영화", "가족과 볼 영화", "스릴러 추천 받기",
                ])
                args = (
                    started_at, ended_at, "system_seed", "system_seed",
                    ended_at, json.dumps(intent_summary, ensure_ascii=False),
                    1, 0, ended_at,  # is_active, is_deleted, last_message_at
                    json.dumps(messages, ensure_ascii=False), rec_count,
                    str(uuid.uuid4()), json.dumps(session_state, ensure_ascii=False),
                    started_at, title, turn_count, user_id,
                )
                if not dry_run:
                    cur.execute(insert_sql, args)
                    inserted += 1
    if not dry_run:
        conn.commit()
    return inserted


# ============================================================
# 4. user_implicit_rating
# ============================================================


def insert_implicit_ratings(
    conn: pymysql.connections.Connection, dry_run: bool = False,
) -> int:
    """sim_user × movie 시청/리뷰/wishlist 종합 implicit_score INSERT.

    공식 (운영 RecommendService 와 정합):
      - reviews 평점 → score += rating (0.5~5.0)
      - user_watch_history → score += 1.0
      - user_wishlist → score += 0.7
      - recommendation_impact clicked → score += 0.5

    같은 movie 에 여러 액션 있으면 score 합산 (max 5.0 클램프).

    성능 최적화: cross-join (users × movies) 대신 signal 테이블 union 으로
    실제 액션이 있는 (user_id, movie_id) 페어만 후보로 만들어 LEFT JOIN.
    """
    aggregate_sql = """
        INSERT INTO user_implicit_rating
            (created_at, updated_at, created_by, updated_by,
             contributing_actions, implicit_score, last_action_at, movie_id, user_id)
        SELECT
            NOW(6), NOW(6), 'system_seed', 'system_seed',
            JSON_OBJECT(
                'review',   COALESCE(MAX(r.rating), 0),
                'watched',  CASE WHEN MAX(w.user_id) IS NOT NULL THEN 1 ELSE 0 END,
                'wished',   CASE WHEN MAX(wl.user_id) IS NOT NULL THEN 1 ELSE 0 END,
                'clicked',  CASE WHEN MAX(i.recommendation_log_id) IS NOT NULL THEN 1 ELSE 0 END
            ) AS contributing_actions,
            LEAST(5.0,
                COALESCE(MAX(r.rating), 0) +
                CASE WHEN MAX(w.user_id) IS NOT NULL THEN 1.0 ELSE 0 END +
                CASE WHEN MAX(wl.user_id) IS NOT NULL THEN 0.7 ELSE 0 END +
                CASE WHEN MAX(i.recommendation_log_id) IS NOT NULL THEN 0.5 ELSE 0 END
            ) AS implicit_score,
            GREATEST(
                COALESCE(MAX(r.created_at), '1970-01-01'),
                COALESCE(MAX(w.created_at), '1970-01-01'),
                COALESCE(MAX(wl.created_at), '1970-01-01')
            ) AS last_action_at,
            s.movie_id,
            s.user_id
        FROM (
            -- sim 사용자가 실제 액션을 한 (user_id, movie_id) 페어만 후보화
            SELECT r.user_id, r.movie_id FROM reviews r
              JOIN users u ON u.user_id = r.user_id
             WHERE u.nickname LIKE 'sim_%%' AND r.is_deleted = b'0'
            UNION
            SELECT w.user_id, w.movie_id FROM user_watch_history w
              JOIN users u ON u.user_id = w.user_id
             WHERE u.nickname LIKE 'sim_%%'
            UNION
            SELECT wl.user_id, wl.movie_id FROM user_wishlist wl
              JOIN users u ON u.user_id = wl.user_id
             WHERE u.nickname LIKE 'sim_%%'
            UNION
            SELECT i.user_id, i.movie_id FROM recommendation_impact i
              JOIN users u ON u.user_id = i.user_id
             WHERE u.nickname LIKE 'sim_%%' AND i.clicked = b'1'
        ) s
        LEFT JOIN reviews r              ON r.user_id = s.user_id  AND r.movie_id = s.movie_id  AND r.is_deleted = b'0'
        LEFT JOIN user_watch_history w   ON w.user_id = s.user_id  AND w.movie_id = s.movie_id
        LEFT JOIN user_wishlist wl       ON wl.user_id = s.user_id AND wl.movie_id = s.movie_id
        LEFT JOIN recommendation_impact i ON i.user_id = s.user_id  AND i.movie_id = s.movie_id  AND i.clicked = b'1'
        GROUP BY s.user_id, s.movie_id
    """
    if dry_run:
        return 0
    with conn.cursor() as cur:
        cur.execute(aggregate_sql)
        affected = cur.rowcount
    conn.commit()
    return affected


# ============================================================
# 5. user_behavior_profile
# ============================================================


def insert_behavior_profiles(
    conn: pymysql.connections.Connection, sim_users: list[tuple[str, dt.datetime]],
    rng: random.Random, dry_run: bool = False,
) -> int:
    """sim_users 별 user_behavior_profile 1행 INSERT."""
    insert_sql = (
        "INSERT INTO user_behavior_profile "
        "(user_id, created_at, updated_at, created_by, updated_by, "
        " activity_level, avg_exploration_depth, director_affinity, genre_affinity, "
        " mood_affinity, profile_updated_at, recommendation_acceptance_rate, taste_consistency) "
        "VALUES (%s, NOW(6), NOW(6), 'system_seed', 'system_seed', "
        "        %s, %s, %s, %s, %s, NOW(6), %s, %s)"
    )
    inserted = 0
    with conn.cursor() as cur:
        for user_id, _ in sim_users:
            persona = sample_persona(rng)
            level_map = {"heavy": "HEAVY", "casual": "CASUAL", "lurker": "LURKER"}
            activity = level_map[persona]
            depth_map = {"heavy": (3.5, 8.0), "casual": (1.5, 4.5), "lurker": (0.5, 2.0)}
            depth = round(rng.uniform(*depth_map[persona]), 3)
            # genre_affinity: 운영 RecommendService 가 사용 — top genre 3종에 weight
            genres = rng.sample(GENRE_POOL, k=3)
            genre_aff = {g: round(rng.uniform(0.3, 1.0), 3) for g in genres}
            moods = rng.sample(MOOD_POOL, k=2)
            mood_aff = {m: round(rng.uniform(0.3, 1.0), 3) for m in moods}
            director_aff: dict[str, float] = {}  # 운영에 director 풀 별도 — 비어둠
            accept_map = {"heavy": (0.4, 0.7), "casual": (0.2, 0.5), "lurker": (0.05, 0.20)}
            accept = round(rng.uniform(*accept_map[persona]), 3)
            consistency = round(rng.uniform(0.4, 0.9), 3)
            args = (
                user_id, activity, depth,
                json.dumps(director_aff, ensure_ascii=False),
                json.dumps(genre_aff, ensure_ascii=False),
                json.dumps(mood_aff, ensure_ascii=False),
                accept, consistency,
            )
            if not dry_run:
                cur.execute(insert_sql, args)
                inserted += 1
    if not dry_run:
        conn.commit()
    return inserted


# ============================================================
# 6. reviews 사후 보정 — review_source / review_category_code
# ============================================================


def update_review_sources(
    conn: pymysql.connections.Connection, sim_review_pairs: list[tuple[int, str, str]],
    user_to_log_ids: dict[str, list[int]], rng: random.Random, dry_run: bool = False,
) -> dict[str, int]:
    """sim_user 의 reviews 분포 보정 — source/category 필드 UPDATE.

    분포: rec_log 60% / chat 15% / detail 15% / course 5% / wishlist 3% / worldcup 2%

    Returns
    -------
    {flow: count} — 각 흐름별 UPDATE 행 수.
    """
    update_sql = (
        "UPDATE reviews SET review_source = %s, review_category_code = %s, "
        "                   updated_at = NOW(6), updated_by = 'system_seed' "
        "WHERE review_id = %s"
    )
    flow_counts: dict[str, int] = {f: 0 for f, _ in REVIEW_SOURCE_DIST}
    with conn.cursor() as cur:
        for review_id, user_id, _movie_id in sim_review_pairs:
            flow = sample_distribution(rng, REVIEW_SOURCE_DIST)
            flow_counts[flow] += 1
            log_ids = user_to_log_ids.get(user_id, [])

            if flow == "rec_log" and log_ids:
                source = f"rec_log_{rng.choice(log_ids)}"
                category = "AI_RECOMMEND"
            elif flow == "rec_log":
                # 페르소나=lurker 등 rec_log 비어있는 경우 — detail 로 폴백
                source, category = None, None
                flow_counts["rec_log"] -= 1
                flow_counts["detail"] += 1
            elif flow == "chat":
                source = f"chat_{uuid.uuid4()}"
                category = "AI_RECOMMEND"
            elif flow == "detail":
                source, category = None, None
            elif flow == "course":
                source = f"course_{rng.randint(1, 6)}"
                category = "COURSE"
            elif flow == "wishlist":
                source = "wishlist"
                category = "WISHLIST"
            elif flow == "worldcup":
                source = f"cup_mch_{rng.randint(1, 100)}"
                category = "WORLDCUP"
            else:
                source, category = None, None
            if not dry_run:
                cur.execute(update_sql, (source, category, review_id))
    if not dry_run:
        conn.commit()
    return flow_counts


# ============================================================
# Main
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM/Payment 도메인 SQL 시드")
    parser.add_argument("--db-url", required=True,
                        help="MySQL URL (예: mysql+pymysql://user:pw@host:3306/db)")
    parser.add_argument("--random-seed", type=int, default=20260507)
    parser.add_argument("--dry-run", action="store_true",
                        help="INSERT/UPDATE 없이 시뮬레이션만")
    parser.add_argument("--skip-rec-log", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--skip-implicit", action="store_true")
    parser.add_argument("--skip-behavior", action="store_true")
    parser.add_argument("--skip-review-update", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    rng = random.Random(args.random_seed)
    now = dt.datetime.now()

    conn_kw = parse_db_url(args.db_url)
    conn = pymysql.connect(**conn_kw)
    try:
        sim_users = fetch_sim_users(conn)
        logger.info("sim_users 풀 — %d 명", len(sim_users))
        if not sim_users:
            logger.error("sim_user 0 — sim_users.py 먼저 실행 필요")
            return 1

        movies = fetch_movie_pool(conn)
        logger.info("movies 풀 — %d 개", len(movies))

        user_to_log_ids: dict[str, list[int]] = {u: [] for u, _ in sim_users}

        # 1. recommendation_log
        if not args.skip_rec_log:
            logger.info("[1/6] recommendation_log INSERT 중…")
            user_to_log_ids = insert_recommendation_logs(
                conn, sim_users, movies, rng, now, args.dry_run,
            )
            total = sum(len(v) for v in user_to_log_ids.values())
            logger.info("[1/6] recommendation_log — %d 행 INSERT", total)

            # 2. recommendation_impact
            logger.info("[2/6] recommendation_impact INSERT 중…")
            impact_n = insert_recommendation_impacts(conn, user_to_log_ids, rng, args.dry_run)
            logger.info("[2/6] recommendation_impact — %d 행 INSERT", impact_n)
        else:
            logger.info("[1-2/6] rec_log/impact skip — DB 에서 기존 system_seed 풀 로드")
            user_to_log_ids = fetch_existing_rec_log_ids(conn)
            total = sum(len(v) for v in user_to_log_ids.values())
            logger.info("[1-2/6] 기존 풀 — %d users / %d log_ids",
                        len(user_to_log_ids), total)

        # 3. chat_session_archive
        if not args.skip_chat:
            logger.info("[3/6] chat_session_archive INSERT 중…")
            chat_n = insert_chat_archives(conn, sim_users, rng, now, args.dry_run)
            logger.info("[3/6] chat_session_archive — %d 행 INSERT", chat_n)
        else:
            logger.info("[3/6] chat_session_archive skip")

        # 4. user_implicit_rating (집계 INSERT — 다른 도메인 적재 후 실행)
        if not args.skip_implicit:
            logger.info("[4/6] user_implicit_rating 집계 INSERT 중…")
            implicit_n = insert_implicit_ratings(conn, args.dry_run)
            logger.info("[4/6] user_implicit_rating — %d 행 INSERT", implicit_n)
        else:
            logger.info("[4/6] user_implicit_rating skip")

        # 5. user_behavior_profile
        if not args.skip_behavior:
            logger.info("[5/6] user_behavior_profile INSERT 중…")
            profile_n = insert_behavior_profiles(conn, sim_users, rng, args.dry_run)
            logger.info("[5/6] user_behavior_profile — %d 행 INSERT", profile_n)
        else:
            logger.info("[5/6] user_behavior_profile skip")

        # 6. reviews 사후 보정 (review_source / category)
        if not args.skip_review_update:
            logger.info("[6/6] reviews review_source/category UPDATE 중…")
            sim_reviews = fetch_sim_review_pairs(conn)
            logger.info("[6/6] sim_user reviews — %d 건", len(sim_reviews))
            flow_counts = update_review_sources(conn, sim_reviews, user_to_log_ids, rng, args.dry_run)
            logger.info("[6/6] reviews 분포: %s", flow_counts)
        else:
            logger.info("[6/6] reviews UPDATE skip")

        logger.info("✅ SQL 시드 완료 (dry_run=%s)", args.dry_run)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
