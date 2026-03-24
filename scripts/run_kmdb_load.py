"""
KMDb (한국영화데이터베이스) 데이터 수집 + 적재 스크립트.

KMDb API에서 한국영화를 수집하고, 기존 DB 영화와 매칭하여:
  - 매칭 성공: 기존 영화에 KMDb 고유 데이터 보강 (수상내역, 관객수, 촬영장소 등)
  - 매칭 실패: 신규 MovieDocument로 생성하여 3DB 적재

사용법:
    # 기본 실행 (2000~현재년도 수집)
    PYTHONPATH=src uv run python scripts/run_kmdb_load.py

    # 연도 범위 지정
    PYTHONPATH=src uv run python scripts/run_kmdb_load.py --start-year 1960 --end-year 2025

    # 캐시 사용 (이전 수집 결과 재활용)
    PYTHONPATH=src uv run python scripts/run_kmdb_load.py --use-cache

    # 적재 배치 크기 조정
    PYTHONPATH=src uv run python scripts/run_kmdb_load.py --batch-size 1000

    # 현재 진행 상태 확인
    PYTHONPATH=src uv run python scripts/run_kmdb_load.py --status

소요 시간 추정:
    - 수집: ~1시간 (43K건, KMDb API 일일 1,000건 제한)
    - 매칭: ~5분 (타이틀 인덱스 구축 + 매칭)
    - 임베딩: ~10분 (신규 ~36K건, Upstage 100 RPM)
    - 적재: ~10분 (3DB 병렬)
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog  # noqa: E402

from monglepick.data_pipeline.embedder import embed_texts  # noqa: E402
from monglepick.data_pipeline.es_loader import load_to_elasticsearch  # noqa: E402
from monglepick.data_pipeline.kmdb_collector import KMDbCollector  # noqa: E402
from monglepick.data_pipeline.kmdb_enricher import (  # noqa: E402
    build_title_index,
    process_kmdb_batch,
)
from monglepick.data_pipeline.neo4j_loader import load_to_neo4j  # noqa: E402
from monglepick.data_pipeline.qdrant_loader import load_to_qdrant  # noqa: E402
from monglepick.db.clients import init_all_clients, close_all_clients  # noqa: E402
from monglepick.config import settings  # noqa: E402

logger = structlog.get_logger()

# ── 경로 및 상수 ──
CACHE_DIR = Path("data")
KMDB_CACHE_PATH = CACHE_DIR / "kmdb_movies_cache.json"
CHECKPOINT_PATH = CACHE_DIR / "kmdb_load_checkpoint.json"
DEFAULT_BATCH_SIZE = 2000
DEFAULT_EMBED_BATCH = 50


# ============================================================
# 체크포인트 관리
# ============================================================

def _new_checkpoint() -> dict:
    """새 체크포인트를 생성한다."""
    return {
        "phase": "",               # collect / match / enrich / embed / load / done
        "total_collected": 0,      # KMDb 수집 총 건수
        "total_matched": 0,        # 기존 영화 매칭 건수 (보강 대상)
        "total_new": 0,            # 신규 영화 건수 (적재 대상)
        "total_enriched": 0,       # 보강 완료 건수
        "total_loaded": 0,         # 신규 적재 완료 건수
        "batch_offset": 0,         # 현재 적재 배치 오프셋
        "failed_ids": [],
        "start_time": datetime.now().isoformat(),
        "last_updated": "",
    }


def _load_checkpoint() -> dict:
    """체크포인트 파일을 로드한다."""
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return _new_checkpoint()


def _save_checkpoint(state: dict) -> None:
    """체크포인트 파일에 진행 상태를 저장한다."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))


# ============================================================
# Qdrant에서 기존 영화 목록 조회 (매칭용)
# ============================================================

def _get_existing_movies_from_qdrant() -> list[dict]:
    """
    Qdrant에서 기존 영화 payload를 조회한다.

    KMDb 매칭에 필요한 필드: id, title, title_en, release_year
    build_title_index()에 전달하기 위한 형식으로 반환한다.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
    db_movies: list[dict] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_vectors=False,
            with_payload=["title", "title_en", "release_year", "source"],
        )
        if not points:
            break

        for p in points:
            payload = p.payload or {}
            db_movies.append({
                "id": str(p.id),
                "title": payload.get("title", ""),
                "title_en": payload.get("title_en", ""),
                "release_year": payload.get("release_year", 0),
                "source": payload.get("source", "tmdb"),
            })

        if next_offset is None:
            break
        offset = next_offset

    client.close()
    logger.info("existing_movies_loaded", count=len(db_movies))
    return db_movies


# ============================================================
# Qdrant payload 보강 (기존 영화에 KMDb 데이터 추가)
# ============================================================

async def _apply_enrichments(enrichments: list[dict]) -> int:
    """
    기존 Qdrant 포인트에 KMDb 보강 데이터를 추가한다.

    set_payload로 기존 payload를 덮어쓰지 않고 필드 단위로 갱신한다.
    빈 문자열이나 0인 값은 갱신하지 않는다.
    동기 QdrantClient를 executor에서 실행하여 이벤트 루프 블로킹을 방지한다.

    Args:
        enrichments: [{"existing_id": str, "data": {...}}, ...]

    Returns:
        보강 완료 건수
    """
    def _sync_apply() -> int:
        """동기 환경에서 set_payload를 배치 실행한다."""
        from qdrant_client import QdrantClient

        sync_client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
        enriched = 0

        for item in enrichments:
            existing_id = item["existing_id"]
            data = item["data"]

            # 빈 값 필터링 (빈 문자열, 0, 빈 리스트는 갱신하지 않음)
            payload_update = {}
            for key, value in data.items():
                if value and value != 0:
                    payload_update[key] = value

            if not payload_update:
                continue

            try:
                # Qdrant ID 타입 결정 (숫자면 int, 아니면 그대로)
                point_id: int | str
                try:
                    point_id = int(existing_id)
                except (ValueError, TypeError):
                    point_id = existing_id

                sync_client.set_payload(
                    collection_name=settings.QDRANT_COLLECTION,
                    payload=payload_update,
                    points=[point_id],
                )
                enriched += 1

            except Exception as e:
                logger.warning("enrich_failed", id=existing_id, error=str(e))

            # 1000건마다 진행률 로깅
            if enriched > 0 and enriched % 1000 == 0:
                logger.info("enrich_progress", enriched=enriched, total=len(enrichments))

        sync_client.close()
        return enriched

    # executor에서 동기 함수 실행 (이벤트 루프 블로킹 방지)
    loop = asyncio.get_event_loop()
    enriched = await loop.run_in_executor(None, _sync_apply)

    logger.info("enrichments_applied", count=enriched, total=len(enrichments))
    return enriched


# ============================================================
# KMDb 캐시 저장/로드
# ============================================================

def _save_kmdb_cache(movies: list, cache_path: Path) -> None:
    """KMDb 수집 결과를 JSON 캐시로 저장한다."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = [m.model_dump() if hasattr(m, "model_dump") else m for m in movies]
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("kmdb_cache_saved", count=len(data), path=str(cache_path))


def _load_kmdb_cache(cache_path: Path):
    """캐시된 KMDb 데이터를 로드한다."""
    from monglepick.data_pipeline.models import KMDbRawMovie

    if not cache_path.exists():
        return None

    data = json.loads(cache_path.read_text())
    movies = []
    for item in data:
        try:
            movies.append(KMDbRawMovie(**item))
        except Exception as e:
            logger.warning("cache_parse_error", error=str(e))
    logger.info("kmdb_cache_loaded", count=len(movies))
    return movies


# ============================================================
# 메인 파이프라인
# ============================================================

async def run_kmdb_load(
    start_year: int = 2000,
    end_year: int | None = None,
    use_cache: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embed_batch_size: int = DEFAULT_EMBED_BATCH,
) -> None:
    """
    KMDb 데이터 수집 → 매칭 → 보강/신규 적재.

    흐름:
    1. KMDb API 수집 (또는 캐시 로드)
    2. Qdrant에서 기존 영화 로드 → 타이틀 인덱스 구축
    3. KMDb ↔ 기존 영화 매칭 (process_kmdb_batch)
       - 매칭 성공 → enrichments (Qdrant payload 갱신)
       - 매칭 실패 → new_documents (신규 적재)
    4. 보강 적용 (Qdrant set_payload)
    5. 신규 영화: 임베딩 → 3DB 적재

    Args:
        start_year: 수집 시작 연도 (기본: 2000)
        end_year: 수집 종료 연도 (기본: 현재 연도)
        use_cache: True이면 기존 캐시 사용
        batch_size: 적재 배치 크기
        embed_batch_size: Upstage 임베딩 배치 크기
    """
    from monglepick.data_pipeline.models import KMDbRawMovie

    if end_year is None:
        end_year = datetime.now().year

    pipeline_start = time.time()
    checkpoint = _load_checkpoint()

    # ── Step 0: DB 클라이언트 초기화 ──
    await init_all_clients()

    try:
        # ── Step 1: KMDb 수집 ──
        print(f"[Step 1] KMDb 영화 수집 ({start_year}~{end_year})")

        kmdb_movies: list[KMDbRawMovie] = []

        if use_cache:
            cached = _load_kmdb_cache(KMDB_CACHE_PATH)
            if cached:
                kmdb_movies = cached
                print(f"  캐시 로드: {len(kmdb_movies):,}건")

        if not kmdb_movies:
            if not settings.KMDB_API_KEY:
                print("  [오류] KMDB_API_KEY가 .env에 설정되지 않았습니다.")
                return

            async with KMDbCollector() as collector:
                kmdb_movies = await collector.collect_all_movies(
                    start_year=start_year,
                    end_year=end_year,
                )
                # KMDbCollector에 public 프로퍼티가 없으므로 내부 카운터 직접 참조
                print(f"  API 수집: {len(kmdb_movies):,}건 (호출: {collector._request_count}회)")

                # 캐시 저장
                _save_kmdb_cache(kmdb_movies, KMDB_CACHE_PATH)

        checkpoint["total_collected"] = len(kmdb_movies)
        checkpoint["phase"] = "collect"
        _save_checkpoint(checkpoint)

        if not kmdb_movies:
            print("  수집된 영화가 없습니다.")
            return

        # ── Step 2: 기존 영화 로드 + 타이틀 인덱스 구축 ──
        print("\n[Step 2] 기존 영화 로드 + 매칭 인덱스 구축")

        db_movies = _get_existing_movies_from_qdrant()
        title_index = build_title_index(db_movies)

        print(f"  기존 영화: {len(db_movies):,}건")
        print(f"  타이틀 인덱스: {len(title_index):,}개 키")

        # ── Step 3: KMDb ↔ 기존 영화 매칭 ──
        print("\n[Step 3] KMDb ↔ 기존 영화 매칭")

        enrichments, new_documents = process_kmdb_batch(kmdb_movies, title_index)

        checkpoint["total_matched"] = len(enrichments)
        checkpoint["total_new"] = len(new_documents)
        checkpoint["phase"] = "match"
        _save_checkpoint(checkpoint)

        print(f"  매칭 성공 (보강 대상): {len(enrichments):,}건")
        print(f"  매칭 실패 (신규 적재): {len(new_documents):,}건")
        print(f"  총 처리:               {len(enrichments) + len(new_documents):,} / {len(kmdb_movies):,}건")

        # ── Step 4: 기존 영화 보강 (Qdrant payload 갱신) ──
        if enrichments:
            print(f"\n[Step 4] 기존 영화 보강 ({len(enrichments):,}건)")

            enriched_count = await _apply_enrichments(enrichments)

            checkpoint["total_enriched"] = enriched_count
            checkpoint["phase"] = "enrich"
            _save_checkpoint(checkpoint)

            print(f"  보강 완료: {enriched_count:,}건")
        else:
            print("\n[Step 4] 보강 대상 없음")

        # ── Step 5 & 6: 신규 영화 임베딩 + 적재 ──
        if new_documents:
            print(f"\n[Step 5-6] 신규 영화 임베딩 + 적재 ({len(new_documents):,}건)")

            total_loaded = 0
            start_offset = checkpoint.get("batch_offset", 0)
            total_batches = (len(new_documents) - start_offset + batch_size - 1) // batch_size

            for batch_idx, batch_start in enumerate(
                range(start_offset, len(new_documents), batch_size)
            ):
                batch_end = min(batch_start + batch_size, len(new_documents))
                batch = new_documents[batch_start:batch_end]
                batch_start_time = time.time()

                # 임베딩 (executor로 이벤트 루프 블로킹 방지)
                texts = [doc.embedding_text for doc in batch]
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, embed_texts, texts, embed_batch_size
                )

                # 3DB 적재
                qdrant_count = await load_to_qdrant(batch, embeddings)
                await load_to_neo4j(batch)
                es_count = await load_to_elasticsearch(batch)

                total_loaded += len(batch)
                batch_elapsed = time.time() - batch_start_time

                # 체크포인트 저장
                checkpoint["batch_offset"] = batch_end
                checkpoint["total_loaded"] = total_loaded
                checkpoint["phase"] = "load"
                _save_checkpoint(checkpoint)

                remaining = total_batches - batch_idx - 1
                eta = batch_elapsed * remaining if remaining > 0 else 0
                print(
                    f"  [Batch {batch_idx + 1:>3}/{total_batches}] "
                    f"{len(batch):,}건 | "
                    f"Qdrant: {qdrant_count} | ES: {es_count} | "
                    f"소요: {batch_elapsed:.1f}s | ETA: {eta / 60:.1f}m"
                )
        else:
            print("\n[Step 5-6] 신규 적재 대상 없음")

        # ── 완료 ──
        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[KMDb 적재 완료]")
        print(f"  수집:      {checkpoint['total_collected']:>10,}건")
        print(f"  보강:      {checkpoint.get('total_enriched', 0):>10,}건")
        print(f"  신규 적재: {checkpoint.get('total_loaded', 0):>10,}건")
        print(f"  소요:      {total_elapsed / 60:>10.1f}분")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


# ============================================================
# 상태 조회
# ============================================================

def show_status() -> None:
    """현재 체크포인트 상태를 출력한다."""
    checkpoint = _load_checkpoint()

    print("=" * 60)
    print("  KMDb 적재 파이프라인 상태")
    print("=" * 60)
    print(f"  현재 단계:     {checkpoint.get('phase', '미시작')}")
    print(f"  수집:          {checkpoint.get('total_collected', 0):>10,}건")
    print(f"  매칭 (보강):   {checkpoint.get('total_matched', 0):>10,}건")
    print(f"  신규:          {checkpoint.get('total_new', 0):>10,}건")
    print(f"  보강 완료:     {checkpoint.get('total_enriched', 0):>10,}건")
    print(f"  신규 적재:     {checkpoint.get('total_loaded', 0):>10,}건")
    print(f"  마지막 업데이트: {checkpoint.get('last_updated', '-')}")
    print("=" * 60)


# ============================================================
# 진입점
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KMDb 데이터 수집 + 적재 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (2000~현재)
  PYTHONPATH=src uv run python scripts/run_kmdb_load.py

  # 연도 범위 지정 + 캐시 사용
  PYTHONPATH=src uv run python scripts/run_kmdb_load.py --start-year 1960 --end-year 2025 --use-cache
        """,
    )
    parser.add_argument(
        "--start-year", type=int, default=2000,
        help="수집 시작 연도 (기본: 2000)",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="수집 종료 연도 (기본: 현재 연도)",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="이전 수집 캐시 사용",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"적재 배치 크기 (기본: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH,
        help=f"Upstage 임베딩 배치 크기 (기본: {DEFAULT_EMBED_BATCH})",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="현재 진행 상태만 확인",
    )
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        asyncio.run(
            run_kmdb_load(
                start_year=args.start_year,
                end_year=args.end_year,
                use_cache=args.use_cache,
                batch_size=args.batch_size,
                embed_batch_size=args.embed_batch_size,
            )
        )
