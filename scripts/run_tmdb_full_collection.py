"""
TMDB 전체 데이터 수집 스크립트.

TMDB Daily Export에서 전체 영화 ID(~1M+)를 다운로드하고,
14개 서브리소스를 포함한 완전한 영화 상세 데이터를 수집한다.

특징:
- 1개 API 키, 1개 IP로 동작 (TMDB Rate Limit: 10초당 40회)
- **동시 수집**: max_workers개 비동기 워커로 병렬 처리 (기본 20, 순차 대비 ~10배 빠름)
- 체크포인트 기반 재개: Ctrl+C로 중단해도 다음 실행에서 이어서 수집
- JSONL 형식 저장: 메모리 부담 없이 대량 데이터 처리
- 인기도 필터: 불필요한 저인기 영화 제외 가능

사용법:
    # 전체 수집 (인기도 0 이상 모든 영화, ~1M건, ~8시간)
    PYTHONPATH=src uv run python scripts/run_tmdb_full_collection.py

    # 인기도 1.0 이상만 수집 (~500K건, ~4시간)
    PYTHONPATH=src uv run python scripts/run_tmdb_full_collection.py --min-popularity 1.0

    # 인기도 5.0 이상만 수집 (~100K건, ~50분)
    PYTHONPATH=src uv run python scripts/run_tmdb_full_collection.py --min-popularity 5.0

    # 성인물 포함 수집
    PYTHONPATH=src uv run python scripts/run_tmdb_full_collection.py --include-adult

    # Daily Export 캐시 무시 (강제 재다운로드)
    PYTHONPATH=src uv run python scripts/run_tmdb_full_collection.py --no-cache

    # 체크포인트 초기화 (처음부터 다시 수집)
    PYTHONPATH=src uv run python scripts/run_tmdb_full_collection.py --reset

출력 파일:
    data/tmdb_full/daily_export_ids.json   — Daily Export에서 추출한 영화 ID 목록 (캐시)
    data/tmdb_full/tmdb_full_movies.jsonl  — 수집된 영화 데이터 (JSON Lines, append)
    data/tmdb_full/checkpoint.json         — 수집 진행 상태 (재개용)

예상 소요 시간:
    ~1,000,000건 / 35 req/sec = ~28,571초 = ~8시간
    Rate Limit 여유 + 재시도 포함 시 ~10시간
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monglepick.data_pipeline.tmdb_collector import TMDBCollector  # noqa: E402


async def main(args: argparse.Namespace) -> None:
    """
    TMDB 전체 수집 메인 로직.

    1단계: Daily Export 다운로드 → 영화 ID 목록 확보
    2단계: 체크포인트 기반으로 상세 데이터 수집
    """
    # ── 체크포인트 초기화 (--reset) ──
    if args.reset:
        checkpoint_file = Path("data/tmdb_full/checkpoint.json")
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("[RESET] 체크포인트 파일 삭제 완료. 처음부터 수집합니다.")
        else:
            print("[RESET] 체크포인트 파일이 없습니다.")

    async with TMDBCollector() as collector:
        # ── 1단계: Daily Export에서 전체 영화 ID 다운로드 ──
        print(f"\n{'='*60}")
        print("1단계: TMDB Daily Export 다운로드")
        print(f"{'='*60}")

        movie_ids = await collector.download_daily_export(
            min_popularity=args.min_popularity,
            exclude_adult=not args.include_adult,
            use_cache=not args.no_cache,
        )

        print(f"  총 영화 ID: {len(movie_ids):,}건")
        print(f"  최소 인기도 필터: {args.min_popularity}")
        print(f"  성인물 {'포함' if args.include_adult else '제외'}")

        # 예상 소요 시간 계산 (Rate Limit 3.5 req/sec 기준)
        estimated_seconds = len(movie_ids) / 3.5
        estimated_hours = estimated_seconds / 3600
        print(f"  동시 워커: {args.max_workers}개")
        print(f"  예상 소요: ~{estimated_hours:.1f}시간 ({estimated_seconds:,.0f}초)")

        # ── 2단계: 체크포인트 기반 상세 수집 ──
        print(f"\n{'='*60}")
        print("2단계: 영화 상세 데이터 수집 (14개 서브리소스)")
        print(f"{'='*60}")
        print("  Ctrl+C로 중단해도 다음 실행에서 이어서 수집합니다.\n")

        result = await collector.collect_full_details_with_checkpoint(
            movie_ids=movie_ids,
            batch_size=args.batch_log_interval,
            save_interval=args.save_interval,
            max_workers=args.max_workers,
        )

        # ── 결과 출력 ──
        print(f"\n{'='*60}")
        print("수집 완료 요약")
        print(f"{'='*60}")
        print(f"  전체 대상: {result['total']:,}건")
        print(f"  수집 성공: {result['collected']:,}건")
        print(f"  수집 실패: {result['failed']:,}건")
        print(f"  이번 수집: +{result.get('new_collected', 0):,}건")
        print(f"  출력 파일: {result['output_file']}")

        # JSONL 파일 크기 출력
        output_path = Path(result["output_file"])
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  파일 크기: {size_mb:,.1f} MB")


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="TMDB 전체 데이터 수집 (Daily Export 기반, 1M+ 영화)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--min-popularity",
        type=float,
        default=0.0,
        help="최소 인기도 필터. 0.0=전체(~1M), 1.0=~500K, 5.0=~100K (기본: 0.0)",
    )
    parser.add_argument(
        "--include-adult",
        action="store_true",
        help="성인물 포함 수집 (기본: 제외)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Daily Export 캐시 무시 (강제 재다운로드)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="체크포인트 초기화 (처음부터 다시 수집)",
    )
    parser.add_argument(
        "--batch-log-interval",
        type=int,
        default=1000,
        help="진행률 로그 출력 간격 (기본: 1000)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="JSONL flush + 체크포인트 저장 간격 (기본: 100)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="동시 수집 워커 수 (기본: 20, Rate Limit 내에서 병렬 처리)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
