"""
MySQL movies 테이블 동기화 스크립트 (Qdrant → MySQL).

Qdrant에 적재된 영화 데이터를 MySQL movies 테이블에 동기화한다.
임베딩은 불필요하며, Qdrant payload를 MySQL 컬럼에 매핑하여 upsert한다.

배경:
    run_full_reload.py는 Qdrant/Neo4j/ES 3개 DB만 적재하고 MySQL은 포함하지 않는다.
    이 스크립트로 Qdrant의 최신 데이터를 MySQL movies 테이블에 반영한다.

사용법:
    # 기본 실행 (Qdrant 전체 → MySQL upsert)
    PYTHONPATH=src uv run python scripts/run_mysql_sync.py

    # 배치 크기 조정
    PYTHONPATH=src uv run python scripts/run_mysql_sync.py --batch-size 1000

    # 현재 MySQL 건수 확인만
    PYTHONPATH=src uv run python scripts/run_mysql_sync.py --status

    # 특정 source만 동기화
    PYTHONPATH=src uv run python scripts/run_mysql_sync.py --source tmdb

소요 시간 추정:
    - Qdrant 조회: ~10분 (806K건 scroll)
    - MySQL upsert: ~30분 (배치 1000건, 806K건)
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog  # noqa: E402

from monglepick.db.clients import init_all_clients, close_all_clients, get_mysql  # noqa: E402
from monglepick.config import settings  # noqa: E402

logger = structlog.get_logger()

DEFAULT_BATCH_SIZE = 1000


# ============================================================
# Qdrant에서 전체 payload 스트리밍 조회
# ============================================================

def _scroll_qdrant_payloads(
    source_filter: str | None = None,
) -> "Generator[list[tuple[str, dict]], None, None]":
    """
    Qdrant에서 1000건씩 payload를 스트리밍 조회한다.

    Args:
        source_filter: 특정 소스만 필터링 (tmdb, kaggle, kobis, kmdb)

    Yields:
        [(point_id, payload), ...] 리스트 (1000건 단위)
    """
    from qdrant_client import QdrantClient, models as qmodels

    client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)

    # 필터 설정
    scroll_filter = None
    if source_filter:
        scroll_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="source",
                match=qmodels.MatchValue(value=source_filter),
            )]
        )

    # MySQL에 필요한 필드만 조회 (payload 전체 대신 필요 필드 지정)
    payload_fields = [
        "title", "title_en", "poster_path", "backdrop_path",
        "release_year", "runtime", "rating", "vote_count",
        "popularity_score", "genres", "director", "cast",
        "certification", "trailer_url", "overview", "tagline",
        "imdb_id", "original_language", "collection_name",
        "kobis_movie_cd", "sales_acc", "audience_count", "screen_count",
        "kobis_watch_grade", "kobis_open_dt", "kmdb_id", "awards",
        "filming_location", "source",
    ]

    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            scroll_filter=scroll_filter,
            with_vectors=False,
            with_payload=payload_fields,
        )
        if not points:
            break

        batch = []
        for p in points:
            batch.append((str(p.id), p.payload or {}))
        yield batch

        if next_offset is None:
            break
        offset = next_offset

    client.close()


# ============================================================
# payload → MySQL INSERT 값 변환
# ============================================================

def _payload_to_mysql_row(point_id: str, payload: dict) -> tuple:
    """
    Qdrant payload를 MySQL movies 테이블 INSERT 값으로 변환한다.

    JSON 컬럼(genres, cast)은 json.dumps()로 직렬화한다.
    NULL 가능 필드는 빈 값을 None으로 변환한다.

    Returns:
        MySQL INSERT에 사용할 값 튜플 (33개 컬럼 순서)
    """
    # cast 필드 정규화 (list[dict] → list[str])
    cast_raw = payload.get("cast", [])
    if cast_raw and isinstance(cast_raw[0], dict):
        cast_list = [c.get("name", "") for c in cast_raw if isinstance(c, dict)]
    elif cast_raw and isinstance(cast_raw[0], str):
        cast_list = cast_raw
    else:
        cast_list = []

    # movie_id: Qdrant point ID 사용
    movie_id = str(payload.get("id", point_id))

    return (
        movie_id,                                                    # movie_id
        payload.get("title", "") or None,                            # title
        payload.get("title_en", "") or None,                         # title_en
        payload.get("poster_path", "") or None,                      # poster_path
        payload.get("backdrop_path", "") or None,                    # backdrop_path
        payload.get("release_year") or None,                         # release_year
        payload.get("runtime") or None,                              # runtime
        payload.get("rating") or None,                               # rating
        payload.get("vote_count") or None,                           # vote_count
        payload.get("popularity_score") or None,                     # popularity_score
        json.dumps(payload.get("genres", []), ensure_ascii=False) if payload.get("genres") else None,  # genres (JSON)
        payload.get("director", "") or None,                         # director
        json.dumps(cast_list, ensure_ascii=False) if cast_list else None,  # cast (JSON)
        payload.get("certification", "") or None,                    # certification
        payload.get("trailer_url", "") or None,                      # trailer_url
        (payload.get("overview", "") or "")[:65535] or None,         # overview (TEXT, 길이 제한)
        payload.get("tagline", "") or None,                          # tagline
        payload.get("imdb_id", "") or None,                          # imdb_id
        payload.get("original_language", "") or None,                # original_language
        payload.get("collection_name", "") or None,                  # collection_name
        payload.get("kobis_movie_cd", "") or None,                   # kobis_movie_cd
        payload.get("sales_acc") or None,                            # sales_acc
        payload.get("audience_count") or None,                       # audience_count
        payload.get("screen_count") or None,                         # screen_count
        payload.get("kobis_watch_grade", "") or None,                # kobis_watch_grade
        payload.get("kobis_open_dt", "") or None,                    # kobis_open_dt
        payload.get("kmdb_id", "") or None,                          # kmdb_id
        payload.get("awards", "") or None,                           # awards
        payload.get("filming_location", "") or None,                 # filming_location
        payload.get("source", "tmdb") or "tmdb",                     # source
    )


# ============================================================
# MySQL 배치 upsert
# ============================================================

# movies 테이블의 INSERT SQL (ON DUPLICATE KEY UPDATE)
UPSERT_SQL = """
INSERT INTO movies (
    movie_id, title, title_en, poster_path, backdrop_path,
    release_year, runtime, rating, vote_count, popularity_score,
    genres, director, cast, certification, trailer_url,
    overview, tagline, imdb_id, original_language, collection_name,
    kobis_movie_cd, sales_acc, audience_count, screen_count,
    kobis_watch_grade, kobis_open_dt, kmdb_id, awards,
    filming_location, source
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s
) ON DUPLICATE KEY UPDATE
    title = VALUES(title),
    title_en = VALUES(title_en),
    poster_path = VALUES(poster_path),
    backdrop_path = VALUES(backdrop_path),
    release_year = VALUES(release_year),
    runtime = VALUES(runtime),
    rating = VALUES(rating),
    vote_count = VALUES(vote_count),
    popularity_score = VALUES(popularity_score),
    genres = VALUES(genres),
    director = VALUES(director),
    cast = VALUES(cast),
    certification = VALUES(certification),
    trailer_url = VALUES(trailer_url),
    overview = VALUES(overview),
    tagline = VALUES(tagline),
    imdb_id = VALUES(imdb_id),
    original_language = VALUES(original_language),
    collection_name = VALUES(collection_name),
    kobis_movie_cd = VALUES(kobis_movie_cd),
    sales_acc = VALUES(sales_acc),
    audience_count = VALUES(audience_count),
    screen_count = VALUES(screen_count),
    kobis_watch_grade = VALUES(kobis_watch_grade),
    kobis_open_dt = VALUES(kobis_open_dt),
    kmdb_id = VALUES(kmdb_id),
    awards = VALUES(awards),
    filming_location = VALUES(filming_location),
    source = VALUES(source)
"""


async def _upsert_batch(rows: list[tuple]) -> int:
    """
    MySQL movies 테이블에 배치 upsert를 실행한다.

    ON DUPLICATE KEY UPDATE로 기존 레코드는 갱신, 신규는 삽입한다.

    Args:
        rows: _payload_to_mysql_row()로 변환된 값 튜플 리스트

    Returns:
        처리된 행 수
    """
    pool = await get_mysql()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                await cursor.executemany(UPSERT_SQL, rows)
                await conn.commit()
                return len(rows)
            except Exception:
                await conn.rollback()
                raise


# ============================================================
# 메인 파이프라인
# ============================================================

async def run_mysql_sync(
    batch_size: int = DEFAULT_BATCH_SIZE,
    source_filter: str | None = None,
) -> None:
    """
    Qdrant → MySQL movies 테이블 동기화.

    1. Qdrant에서 payload를 1000건씩 스트리밍 조회
    2. MySQL INSERT 값으로 변환
    3. 배치 upsert (ON DUPLICATE KEY UPDATE)

    Args:
        batch_size: MySQL upsert 배치 크기
        source_filter: 특정 소스만 동기화 (tmdb, kaggle, kobis, kmdb)
    """
    pipeline_start = time.time()

    # ── Step 0: DB 초기화 ──
    await init_all_clients()

    try:
        # ── Step 1: Qdrant → MySQL 동기화 ──
        filter_msg = f" (source: {source_filter})" if source_filter else ""
        print(f"[Step 1] Qdrant → MySQL 동기화{filter_msg}")

        total_processed = 0
        total_upserted = 0
        total_errors = 0
        batch_count = 0

        # Qdrant 스트리밍 조회 (1000건씩)
        for qdrant_batch in _scroll_qdrant_payloads(source_filter):
            # payload → MySQL 행 변환
            rows: list[tuple] = []
            for pid, payload in qdrant_batch:
                try:
                    row = _payload_to_mysql_row(pid, payload)
                    # title이 없는 행은 스킵
                    if row[1]:  # title 필드 (인덱스 1)
                        rows.append(row)
                except Exception as e:
                    total_errors += 1
                    if total_errors <= 10:
                        logger.warning("row_convert_failed", id=pid, error=str(e))

            # 배치 upsert (batch_size 단위로 분할)
            for i in range(0, len(rows), batch_size):
                sub_batch = rows[i:i + batch_size]
                try:
                    affected = await _upsert_batch(sub_batch)
                    total_upserted += len(sub_batch)
                except Exception as e:
                    total_errors += len(sub_batch)
                    logger.error("mysql_upsert_failed", batch_size=len(sub_batch), error=str(e))

            total_processed += len(qdrant_batch)
            batch_count += 1

            # 10배치마다 진행률 출력
            if batch_count % 10 == 0:
                elapsed = time.time() - pipeline_start
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(
                    f"  진행: {total_processed:>10,}건 | "
                    f"upsert: {total_upserted:>10,}건 | "
                    f"에러: {total_errors:>5,}건 | "
                    f"속도: {rate:,.0f}건/s"
                )

        # ── 완료 ──
        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[MySQL 동기화 완료]")
        print(f"  Qdrant 조회: {total_processed:>10,}건")
        print(f"  MySQL upsert: {total_upserted:>10,}건")
        print(f"  에러:         {total_errors:>10,}건")
        print(f"  소요:         {total_elapsed / 60:>10.1f}분")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


# ============================================================
# 상태 조회
# ============================================================

async def show_status() -> None:
    """현재 MySQL movies 테이블 건수를 출력한다."""
    await init_all_clients()

    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # 전체 건수
                await cursor.execute("SELECT COUNT(*) FROM movies")
                (total,) = await cursor.fetchone()

                # source별 건수
                await cursor.execute(
                    "SELECT source, COUNT(*) FROM movies GROUP BY source ORDER BY COUNT(*) DESC"
                )
                sources = await cursor.fetchall()

        print("=" * 60)
        print("  MySQL movies 테이블 현황")
        print("=" * 60)
        print(f"  전체: {total:>10,}건")
        for src, cnt in sources:
            print(f"  {src or 'NULL':>10}: {cnt:>10,}건")
        print("=" * 60)

    finally:
        await close_all_clients()


# ============================================================
# 진입점
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MySQL movies 테이블 동기화 (Qdrant → MySQL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 동기화
  PYTHONPATH=src uv run python scripts/run_mysql_sync.py

  # TMDB 소스만 동기화
  PYTHONPATH=src uv run python scripts/run_mysql_sync.py --source tmdb

  # 현재 MySQL 건수 확인
  PYTHONPATH=src uv run python scripts/run_mysql_sync.py --status
        """,
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"MySQL upsert 배치 크기 (기본: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        choices=["tmdb", "kaggle", "kobis", "kmdb"],
        help="특정 소스만 동기화",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="현재 MySQL movies 건수 확인만",
    )
    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status())
    else:
        asyncio.run(
            run_mysql_sync(
                batch_size=args.batch_size,
                source_filter=args.source,
            )
        )
