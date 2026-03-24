"""
KOBIS (영화진흥위원회) 데이터 수집 + 적재 스크립트.

KOBIS API에서 영화 목록/상세/박스오피스를 수집하고,
기존 DB(TMDB/Kaggle)와 중복 제거 후 Qdrant/Neo4j/ES에 적재한다.

사용법:
    # 기본 실행 (캐시 없으면 API 수집, 있으면 캐시 사용)
    PYTHONPATH=src uv run python scripts/run_kobis_load.py

    # 캐시 무시하고 API 재수집
    PYTHONPATH=src uv run python scripts/run_kobis_load.py --no-cache

    # 상세정보 수집 제한 (일일 API 한도 고려)
    PYTHONPATH=src uv run python scripts/run_kobis_load.py --detail-limit 2500

    # 박스오피스 히스토리 수집 (최근 N일)
    PYTHONPATH=src uv run python scripts/run_kobis_load.py --boxoffice-days 365

    # 적재 배치 크기 조정
    PYTHONPATH=src uv run python scripts/run_kobis_load.py --batch-size 1000

    # 현재 진행 상태 확인
    PYTHONPATH=src uv run python scripts/run_kobis_load.py --status

소요 시간 추정:
    - 목록 수집: ~10분 (117K건, 1,170 페이지)
    - 상세 수집: ~20분/2,500건 (일일 한도)
    - 임베딩: ~15분 (77K건, Upstage 100 RPM)
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
from monglepick.data_pipeline.kobis_collector import (  # noqa: E402
    KOBISCollector,
    save_kobis_cache,
    load_kobis_cache,
)
from monglepick.data_pipeline.kobis_movie_converter import (  # noqa: E402
    convert_kobis_movies,
    dedup_kobis_movies,
)
from monglepick.data_pipeline.neo4j_loader import load_to_neo4j  # noqa: E402
from monglepick.data_pipeline.qdrant_loader import load_to_qdrant  # noqa: E402
from monglepick.db.clients import init_all_clients, close_all_clients  # noqa: E402
from monglepick.config import settings  # noqa: E402

logger = structlog.get_logger()

# ── 경로 및 상수 ──
CACHE_DIR = Path("data")
KOBIS_CACHE_PATH = CACHE_DIR / "kobis_movies_cache.json"
CHECKPOINT_PATH = CACHE_DIR / "kobis_load_checkpoint.json"
DEFAULT_BATCH_SIZE = 2000
DEFAULT_EMBED_BATCH = 50
DEFAULT_DETAIL_LIMIT = 2500  # KOBIS 일일 API 한도 고려


# ============================================================
# 체크포인트 관리
# ============================================================

def _new_checkpoint() -> dict:
    """새 체크포인트를 생성한다."""
    return {
        "phase": "",                # collect / dedup / detail / boxoffice / embed / load / done
        "total_collected": 0,       # 수집된 총 영화 수
        "total_after_dedup": 0,     # 중복 제거 후 영화 수
        "detail_fetched": 0,        # 상세정보 수집 완료 수
        "total_converted": 0,       # MovieDocument 변환 완료 수
        "total_loaded": 0,          # DB 적재 완료 수
        "batch_offset": 0,          # 현재 적재 배치 오프셋
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
# Qdrant에서 기존 영화 목록 조회 (중복 제거용)
# ============================================================

def _get_existing_movies_from_qdrant() -> list[dict]:
    """
    Qdrant에서 기존 영화 payload를 조회한다 (id, title, title_en, release_year).

    중복 제거 시 dedup_kobis_movies()에 전달하기 위해
    필요한 최소 필드만 가져온다.
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
                "id": p.id,
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
# 메인 파이프라인
# ============================================================

async def run_kobis_load(
    use_cache: bool = True,
    detail_limit: int = DEFAULT_DETAIL_LIMIT,
    boxoffice_days: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embed_batch_size: int = DEFAULT_EMBED_BATCH,
) -> None:
    """
    KOBIS 데이터 수집 → 중복 제거 → 변환 → 임베딩 → 3DB 적재.

    흐름:
    1. KOBIS 목록 수집 (또는 캐시 로드)
    2. 기존 DB(Qdrant)와 중복 제거
    3. 상세정보 수집 (선택, 일일 한도 있음)
    4. 박스오피스 수집 (선택)
    5. MovieDocument 변환
    6. 임베딩 (Upstage API)
    7. Qdrant/Neo4j/ES 적재

    Args:
        use_cache: True이면 기존 캐시 사용 (기본: True)
        detail_limit: 상세정보 수집 건수 제한 (0이면 스킵)
        boxoffice_days: 박스오피스 히스토리 수집 일수 (0이면 스킵)
        batch_size: 적재 배치 크기
        embed_batch_size: Upstage 임베딩 API 배치 크기
    """
    pipeline_start = time.time()
    checkpoint = _load_checkpoint()

    # ── Step 0: DB 클라이언트 초기화 ──
    await init_all_clients()

    try:
        # ── Step 1: KOBIS 목록 수집 ──
        print("[Step 1] KOBIS 영화 목록 수집")

        kobis_movies: list[dict] = []

        if use_cache and KOBIS_CACHE_PATH.exists():
            cached = load_kobis_cache(str(KOBIS_CACHE_PATH))
            if cached:
                kobis_movies = cached
                print(f"  캐시 로드: {len(kobis_movies):,}건")
                logger.info("kobis_cache_loaded", count=len(kobis_movies))

        if not kobis_movies:
            if not settings.KOBIS_API_KEY:
                print("  [오류] KOBIS_API_KEY가 .env에 설정되지 않았습니다.")
                return

            async with KOBISCollector() as collector:
                kobis_movies = await collector.collect_all_movie_list()
                print(f"  API 수집: {len(kobis_movies):,}건")

                # 캐시 저장
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                save_kobis_cache(kobis_movies, str(KOBIS_CACHE_PATH))
                print(f"  캐시 저장: {KOBIS_CACHE_PATH}")

        checkpoint["total_collected"] = len(kobis_movies)
        checkpoint["phase"] = "collect"
        _save_checkpoint(checkpoint)

        # ── Step 2: 기존 DB와 중복 제거 ──
        print("\n[Step 2] 기존 DB와 중복 제거")

        db_movies = _get_existing_movies_from_qdrant()
        # 기존 KOBIS 소스 ID 제외 (재적재 방지)
        existing_kobis_ids = {
            m["id"] for m in db_movies if m.get("source") == "kobis"
        }

        deduped_movies = dedup_kobis_movies(
            kobis_movies=kobis_movies,
            db_movies=db_movies,
            exclude_ids=existing_kobis_ids,
        )

        checkpoint["total_after_dedup"] = len(deduped_movies)
        checkpoint["phase"] = "dedup"
        _save_checkpoint(checkpoint)

        print(f"  원본: {len(kobis_movies):,}건 → 중복 제거 후: {len(deduped_movies):,}건")

        if not deduped_movies:
            print("  적재할 새 영화가 없습니다.")
            return

        # ── Step 3: 상세정보 수집 (선택) ──
        detail_map: dict[str, dict] = {}

        if detail_limit > 0 and settings.KOBIS_API_KEY:
            print(f"\n[Step 3] 상세정보 수집 (최대 {detail_limit}건)")

            # 상세정보가 없는 영화의 movieCd 추출
            movie_cds = [m.get("movieCd", "") for m in deduped_movies if m.get("movieCd")]
            target_cds = movie_cds[:detail_limit]

            async with KOBISCollector() as collector:
                details = await collector.collect_movie_details_batch(target_cds)
                for d in details:
                    detail_map[d.movie_cd] = {
                        "actors": d.actors,
                        "staffs": d.staffs,
                        "audits": d.audits,
                        "show_tm": d.show_tm,
                        "companys": d.companys,
                    }
                print(f"  상세 수집 완료: {len(detail_map):,}건 (API 호출: {collector.call_count}회)")

            checkpoint["detail_fetched"] = len(detail_map)
            checkpoint["phase"] = "detail"
            _save_checkpoint(checkpoint)
        else:
            print("\n[Step 3] 상세정보 수집 스킵")

        # ── Step 4: 박스오피스 수집 (선택) ──
        boxoffice_map: dict[str, dict] = {}

        if boxoffice_days > 0 and settings.KOBIS_API_KEY:
            print(f"\n[Step 4] 박스오피스 히스토리 수집 ({boxoffice_days}일)")

            async with KOBISCollector() as collector:
                bo_data = await collector.collect_boxoffice_history(days=boxoffice_days)
                # KOBISBoxOffice → dict 변환
                for movie_cd, bo in bo_data.items():
                    boxoffice_map[movie_cd] = {
                        "audi_acc": bo.audi_acc,
                        "sales_acc": bo.sales_acc,
                        "scrn_cnt": bo.scrn_cnt,
                    }
                print(f"  박스오피스 수집: {len(boxoffice_map):,}건")

            checkpoint["phase"] = "boxoffice"
            _save_checkpoint(checkpoint)
        else:
            print("\n[Step 4] 박스오피스 수집 스킵")

        # ── Step 5: MovieDocument 변환 ──
        print("\n[Step 5] MovieDocument 변환")

        documents = convert_kobis_movies(
            kobis_movies=deduped_movies,
            detail_map=detail_map,
            boxoffice_map=boxoffice_map,
        )

        checkpoint["total_converted"] = len(documents)
        checkpoint["phase"] = "convert"
        _save_checkpoint(checkpoint)

        print(f"  변환 성공: {len(documents):,}건 / {len(deduped_movies):,}건")

        if not documents:
            print("  변환된 문서가 없습니다.")
            return

        # ── Step 6 & 7: 배치 단위 임베딩 → 적재 ──
        print(f"\n[Step 6-7] 임베딩 + 3DB 적재 (배치: {batch_size}건)")

        total_loaded = 0
        start_offset = checkpoint.get("batch_offset", 0)
        total_batches = (len(documents) - start_offset + batch_size - 1) // batch_size

        for batch_idx, batch_start in enumerate(
            range(start_offset, len(documents), batch_size)
        ):
            batch_end = min(batch_start + batch_size, len(documents))
            batch = documents[batch_start:batch_end]
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

            # 진행률 출력
            remaining = total_batches - batch_idx - 1
            eta = batch_elapsed * remaining if remaining > 0 else 0
            print(
                f"  [Batch {batch_idx + 1:>3}/{total_batches}] "
                f"{len(batch):,}건 적재 | "
                f"Qdrant: {qdrant_count} | ES: {es_count} | "
                f"소요: {batch_elapsed:.1f}s | ETA: {eta / 60:.1f}m"
            )

        # ── 완료 ──
        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[KOBIS 적재 완료]")
        print(f"  수집:      {checkpoint['total_collected']:>10,}건")
        print(f"  중복 제거: {checkpoint['total_after_dedup']:>10,}건")
        print(f"  변환:      {checkpoint['total_converted']:>10,}건")
        print(f"  적재:      {total_loaded:>10,}건")
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
    print("  KOBIS 적재 파이프라인 상태")
    print("=" * 60)
    print(f"  현재 단계:     {checkpoint.get('phase', '미시작')}")
    print(f"  수집:          {checkpoint.get('total_collected', 0):>10,}건")
    print(f"  중복 제거 후:  {checkpoint.get('total_after_dedup', 0):>10,}건")
    print(f"  상세 수집:     {checkpoint.get('detail_fetched', 0):>10,}건")
    print(f"  변환:          {checkpoint.get('total_converted', 0):>10,}건")
    print(f"  적재:          {checkpoint.get('total_loaded', 0):>10,}건")
    print(f"  마지막 업데이트: {checkpoint.get('last_updated', '-')}")
    print("=" * 60)


# ============================================================
# 진입점
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KOBIS 데이터 수집 + 적재 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (캐시 사용)
  PYTHONPATH=src uv run python scripts/run_kobis_load.py

  # 상세정보 + 박스오피스 포함
  PYTHONPATH=src uv run python scripts/run_kobis_load.py --detail-limit 2500 --boxoffice-days 365
        """,
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="캐시 무시하고 API 재수집",
    )
    parser.add_argument(
        "--detail-limit", type=int, default=DEFAULT_DETAIL_LIMIT,
        help=f"상세정보 수집 건수 제한 (기본: {DEFAULT_DETAIL_LIMIT}, 0이면 스킵)",
    )
    parser.add_argument(
        "--boxoffice-days", type=int, default=0,
        help="박스오피스 히스토리 수집 일수 (기본: 0=스킵)",
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
            run_kobis_load(
                use_cache=not args.no_cache,
                detail_limit=args.detail_limit,
                boxoffice_days=args.boxoffice_days,
                batch_size=args.batch_size,
                embed_batch_size=args.embed_batch_size,
            )
        )
