"""
LangSmith Custom Dashboard 메트릭 수집 스크립트.

LangSmith API에서 트레이싱 데이터를 조회하여 노드별 성능 메트릭을 산출한다.

수집 메트릭:
1. 노드별 평균 지연시간 (Average Latency per Node)
2. 노드별 에러율 (Error Rate per Node)
3. 노드별 호출 횟수 (Invocation Count per Node)
4. 전체 그래프 평균 지연시간 (End-to-End Latency)
5. 의도별 분포 (Intent Distribution)
6. 시간대별 요청 추이 (Hourly Throughput)

사용법:
    # 최근 24시간 메트릭 조회
    PYTHONPATH=src uv run python scripts/run_dashboard_metrics.py

    # 최근 7일 메트릭 조회
    PYTHONPATH=src uv run python scripts/run_dashboard_metrics.py --days 7

    # JSON 형식으로 출력 (파이프라인 연동용)
    PYTHONPATH=src uv run python scripts/run_dashboard_metrics.py --json

    # 특정 프로젝트 지정
    PYTHONPATH=src uv run python scripts/run_dashboard_metrics.py --project monglepick

전제 조건:
- LANGCHAIN_API_KEY 환경변수 설정 필요
- LangSmith 트레이싱 활성화 상태에서 실행된 데이터가 있어야 함
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

# LangSmith 환경변수 설정 (import 전에 설정)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

from langsmith import Client  # noqa: E402


# ============================================================
# 14개 Chat Agent 노드 목록 (graph.py NODE_STATUS_MESSAGES 기준)
# ============================================================

CHAT_AGENT_NODES: list[str] = [
    "context_loader",
    "image_analyzer",
    "intent_emotion_classifier",
    "preference_refiner",
    "question_generator",
    "query_builder",
    "rag_retriever",
    "retrieval_quality_checker",
    "recommendation_ranker",
    "explanation_generator",
    "response_formatter",
    "error_handler",
    "general_responder",
    "tool_executor_node",
]

# LLM 체인 이름 (각 체인의 @traceable name)
LLM_CHAIN_NAMES: list[str] = [
    "classify_intent_and_emotion",
    "extract_preferences",
    "generate_explanation",
    "generate_explanations_batch",
    "generate_question",
    "analyze_image",
    "generate_general_response",
]

# 모든 추적 대상 이름
ALL_TRACEABLE_NAMES: list[str] = CHAT_AGENT_NODES + LLM_CHAIN_NAMES


# ============================================================
# 메트릭 수집 함수
# ============================================================


def collect_node_metrics(
    client: Client,
    project_name: str,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, dict[str, Any]]:
    """
    노드별 성능 메트릭을 수집한다.

    LangSmith API의 list_runs를 사용하여 각 노드/체인의
    지연시간, 에러율, 호출 횟수를 산출한다.

    Args:
        client: LangSmith Client 인스턴스
        project_name: LangSmith 프로젝트 이름
        start_time: 조회 시작 시각 (UTC)
        end_time: 조회 종료 시각 (UTC)

    Returns:
        노드별 메트릭 딕셔너리:
        {
            "node_name": {
                "count": int,           # 총 호출 횟수
                "error_count": int,     # 에러 횟수
                "error_rate": float,    # 에러율 (0.0~1.0)
                "avg_latency_ms": float,  # 평균 지연시간 (ms)
                "p50_latency_ms": float,  # 중앙값 지연시간 (ms)
                "p95_latency_ms": float,  # 95 퍼센타일 지연시간 (ms)
                "max_latency_ms": float,  # 최대 지연시간 (ms)
                "total_tokens": int,    # 총 토큰 사용량 (LLM 체인만)
            }
        }
    """
    node_metrics: dict[str, dict[str, Any]] = {}

    for name in ALL_TRACEABLE_NAMES:
        latencies: list[float] = []
        error_count = 0
        total_tokens = 0
        count = 0

        try:
            # LangSmith API로 해당 이름의 run 조회
            runs = client.list_runs(
                project_name=project_name,
                filter=f'eq(name, "{name}")',
                start_time=start_time,
                end_time=end_time,
                limit=1000,
            )

            for run in runs:
                count += 1

                # 에러 여부 확인
                if run.error is not None:
                    error_count += 1

                # 지연시간 계산 (start_time, end_time 모두 있는 경우)
                if run.start_time and run.end_time:
                    duration_ms = (
                        run.end_time - run.start_time
                    ).total_seconds() * 1000
                    latencies.append(duration_ms)

                # 토큰 사용량 (LLM 체인에만 해당)
                if run.total_tokens:
                    total_tokens += run.total_tokens

        except Exception as e:
            # 특정 노드 조회 실패 시 무시하고 계속 진행
            print(f"  ⚠️  {name} 조회 실패: {e}", file=sys.stderr)
            continue

        # 호출 기록이 없는 노드는 건너뛰기
        if count == 0:
            continue

        # 지연시간 통계 산출
        latencies.sort()
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p50_latency = _percentile(latencies, 50)
        p95_latency = _percentile(latencies, 95)
        max_latency = max(latencies) if latencies else 0

        node_metrics[name] = {
            "count": count,
            "error_count": error_count,
            "error_rate": round(error_count / count, 4) if count > 0 else 0,
            "avg_latency_ms": round(avg_latency, 1),
            "p50_latency_ms": round(p50_latency, 1),
            "p95_latency_ms": round(p95_latency, 1),
            "max_latency_ms": round(max_latency, 1),
            "total_tokens": total_tokens,
        }

    return node_metrics


def collect_e2e_metrics(
    client: Client,
    project_name: str,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, Any]:
    """
    Chat Agent 그래프 전체의 End-to-End 메트릭을 수집한다.

    최상위 run (run_type="chain", parent_run_id=None)을 조회하여
    전체 그래프 실행 시간, 성공률 등을 산출한다.

    Args:
        client: LangSmith Client 인스턴스
        project_name: LangSmith 프로젝트 이름
        start_time: 조회 시작 시각 (UTC)
        end_time: 조회 종료 시각 (UTC)

    Returns:
        E2E 메트릭 딕셔너리:
        {
            "total_requests": int,
            "success_count": int,
            "error_count": int,
            "success_rate": float,
            "avg_latency_ms": float,
            "p50_latency_ms": float,
            "p95_latency_ms": float,
            "max_latency_ms": float,
        }
    """
    latencies: list[float] = []
    error_count = 0
    total = 0

    try:
        # 최상위 run만 조회 (그래프 전체 실행)
        runs = client.list_runs(
            project_name=project_name,
            run_type="chain",
            filter='eq(name, "run_chat_agent_sync")',
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )

        for run in runs:
            total += 1

            if run.error is not None:
                error_count += 1

            if run.start_time and run.end_time:
                duration_ms = (
                    run.end_time - run.start_time
                ).total_seconds() * 1000
                latencies.append(duration_ms)

    except Exception as e:
        print(f"  ⚠️  E2E 메트릭 조회 실패: {e}", file=sys.stderr)

    latencies.sort()

    return {
        "total_requests": total,
        "success_count": total - error_count,
        "error_count": error_count,
        "success_rate": round((total - error_count) / total, 4) if total > 0 else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p50_latency_ms": round(_percentile(latencies, 50), 1),
        "p95_latency_ms": round(_percentile(latencies, 95), 1),
        "max_latency_ms": round(max(latencies), 1) if latencies else 0,
    }


def collect_intent_distribution(
    client: Client,
    project_name: str,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, int]:
    """
    의도별 분포를 수집한다.

    intent_emotion_classifier 노드의 출력에서 intent 값을 집계한다.

    Args:
        client: LangSmith Client 인스턴스
        project_name: LangSmith 프로젝트 이름
        start_time: 조회 시작 시각 (UTC)
        end_time: 조회 종료 시각 (UTC)

    Returns:
        의도별 횟수: {"recommend": 45, "general": 20, ...}
    """
    intent_counts: dict[str, int] = defaultdict(int)

    try:
        runs = client.list_runs(
            project_name=project_name,
            filter='eq(name, "intent_emotion_classifier")',
            start_time=start_time,
            end_time=end_time,
            limit=1000,
        )

        for run in runs:
            if run.outputs and "intent" in run.outputs:
                intent_obj = run.outputs["intent"]
                # intent는 Pydantic 모델 → dict로 직렬화됨
                if isinstance(intent_obj, dict):
                    intent_type = intent_obj.get("intent", "unknown")
                elif hasattr(intent_obj, "intent"):
                    intent_type = intent_obj.intent
                else:
                    intent_type = str(intent_obj)
                intent_counts[intent_type] += 1

    except Exception as e:
        print(f"  ⚠️  의도 분포 조회 실패: {e}", file=sys.stderr)

    return dict(intent_counts)


def collect_hourly_throughput(
    client: Client,
    project_name: str,
    start_time: datetime,
    end_time: datetime,
) -> list[dict[str, Any]]:
    """
    시간대별 요청 추이를 수집한다.

    1시간 단위로 그래프 실행 횟수와 에러 횟수를 집계한다.

    Args:
        client: LangSmith Client 인스턴스
        project_name: LangSmith 프로젝트 이름
        start_time: 조회 시작 시각 (UTC)
        end_time: 조회 종료 시각 (UTC)

    Returns:
        시간대별 데이터 리스트:
        [{"hour": "2026-03-12T09:00:00Z", "requests": 15, "errors": 1}, ...]
    """
    # 시간대별 카운터 초기화
    hourly: dict[str, dict[str, int]] = {}
    current = start_time.replace(minute=0, second=0, microsecond=0)
    while current < end_time:
        key = current.isoformat()
        hourly[key] = {"requests": 0, "errors": 0}
        current += timedelta(hours=1)

    try:
        # context_loader는 모든 요청에서 첫 번째로 실행되므로 요청 수 산출에 적합
        runs = client.list_runs(
            project_name=project_name,
            filter='eq(name, "context_loader")',
            start_time=start_time,
            end_time=end_time,
            limit=5000,
        )

        for run in runs:
            if run.start_time:
                hour_key = run.start_time.replace(
                    minute=0, second=0, microsecond=0
                ).isoformat()
                if hour_key in hourly:
                    hourly[hour_key]["requests"] += 1
                    if run.error is not None:
                        hourly[hour_key]["errors"] += 1

    except Exception as e:
        print(f"  ⚠️  시간대별 추이 조회 실패: {e}", file=sys.stderr)

    # 정렬된 리스트로 변환
    return [
        {"hour": k, **v}
        for k, v in sorted(hourly.items())
    ]


# ============================================================
# 유틸리티 함수
# ============================================================


def _percentile(sorted_data: list[float], pct: int) -> float:
    """
    정렬된 데이터에서 퍼센타일 값을 계산한다.

    Args:
        sorted_data: 오름차순 정렬된 float 리스트
        pct: 퍼센타일 (0~100)

    Returns:
        해당 퍼센타일 값 (데이터 없으면 0)
    """
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = int(n * pct / 100)
    # 인덱스 범위 제한
    idx = min(idx, n - 1)
    return sorted_data[idx]


# ============================================================
# 출력 함수
# ============================================================


def print_text_report(
    node_metrics: dict[str, dict[str, Any]],
    e2e_metrics: dict[str, Any],
    intent_dist: dict[str, int],
    hourly: list[dict[str, Any]],
    days: int,
) -> None:
    """메트릭을 터미널에 텍스트 형식으로 출력한다."""

    print()
    print("=" * 80)
    print(f"  몽글픽 Chat Agent 성능 대시보드  (최근 {days}일)")
    print("=" * 80)

    # ── 1. E2E 요약 ──
    print("\n[1] End-to-End 요약")
    print("-" * 50)
    print(f"  총 요청 수:     {e2e_metrics['total_requests']}")
    print(f"  성공:           {e2e_metrics['success_count']}")
    print(f"  실패:           {e2e_metrics['error_count']}")
    print(f"  성공률:         {e2e_metrics['success_rate'] * 100:.1f}%")
    print(f"  평균 지연:      {e2e_metrics['avg_latency_ms']:,.0f} ms")
    print(f"  P50 지연:       {e2e_metrics['p50_latency_ms']:,.0f} ms")
    print(f"  P95 지연:       {e2e_metrics['p95_latency_ms']:,.0f} ms")
    print(f"  최대 지연:      {e2e_metrics['max_latency_ms']:,.0f} ms")

    # ── 2. 노드별 성능 ──
    print("\n[2] 노드별 성능 메트릭")
    print("-" * 90)

    # 테이블 헤더
    header = f"{'노드/체인':<35} {'호출':>6} {'에러':>5} {'에러율':>7} {'평균(ms)':>9} {'P95(ms)':>9} {'최대(ms)':>9}"
    print(header)
    print("-" * 90)

    # 노드 먼저, 그다음 체인 순서로 출력
    for name in CHAT_AGENT_NODES:
        if name in node_metrics:
            m = node_metrics[name]
            _print_metric_row(name, m)

    # LLM 체인 구분선
    if any(name in node_metrics for name in LLM_CHAIN_NAMES):
        print(f"{'--- LLM 체인 ---':<35}")
        for name in LLM_CHAIN_NAMES:
            if name in node_metrics:
                m = node_metrics[name]
                _print_metric_row(name, m)

    # ── 3. 의도 분포 ──
    print("\n[3] 의도별 분포")
    print("-" * 40)
    if intent_dist:
        total_intents = sum(intent_dist.values())
        for intent, count in sorted(intent_dist.items(), key=lambda x: -x[1]):
            pct = count / total_intents * 100
            bar = "#" * int(pct / 2)
            print(f"  {intent:<15} {count:>5} ({pct:5.1f}%) {bar}")
    else:
        print("  (데이터 없음)")

    # ── 4. 시간대별 추이 (데이터가 있는 시간대만) ──
    active_hours = [h for h in hourly if h["requests"] > 0]
    if active_hours:
        print("\n[4] 시간대별 요청 추이 (활성 시간대만)")
        print("-" * 60)
        max_req = max(h["requests"] for h in active_hours)
        for h in active_hours:
            # UTC → KST (UTC+9) 변환하여 표시
            utc_time = datetime.fromisoformat(h["hour"])
            kst_time = utc_time + timedelta(hours=9)
            hour_label = kst_time.strftime("%m/%d %H:%M")
            bar_len = int(h["requests"] / max_req * 30) if max_req > 0 else 0
            bar = "#" * bar_len
            err_info = f" (err:{h['errors']})" if h["errors"] > 0 else ""
            print(f"  {hour_label}  {h['requests']:>4} {bar}{err_info}")

    # ── 5. 경고 (에러율 높은 노드) ──
    high_error_nodes = [
        (name, m)
        for name, m in node_metrics.items()
        if m["error_rate"] > 0.05 and m["count"] >= 5  # 에러율 5% 초과 + 최소 5회
    ]
    if high_error_nodes:
        print("\n[!] 주의: 에러율 높은 노드")
        print("-" * 50)
        for name, m in sorted(high_error_nodes, key=lambda x: -x[1]["error_rate"]):
            print(f"  {name}: {m['error_rate'] * 100:.1f}% ({m['error_count']}/{m['count']})")

    # ── 6. 병목 노드 (P95 지연시간 상위 3개) ──
    latency_sorted = sorted(
        [(name, m) for name, m in node_metrics.items() if m["count"] >= 3],
        key=lambda x: -x[1]["p95_latency_ms"],
    )
    if latency_sorted:
        print("\n[!] 병목 노드 (P95 지연시간 상위)")
        print("-" * 50)
        for name, m in latency_sorted[:3]:
            print(f"  {name}: P95={m['p95_latency_ms']:,.0f}ms, avg={m['avg_latency_ms']:,.0f}ms")

    print("\n" + "=" * 80)
    print()


def _print_metric_row(name: str, m: dict[str, Any]) -> None:
    """노드 메트릭을 한 행으로 출력한다."""
    error_rate_str = f"{m['error_rate'] * 100:.1f}%"
    print(
        f"  {name:<33} {m['count']:>6} {m['error_count']:>5} "
        f"{error_rate_str:>7} {m['avg_latency_ms']:>9,.1f} "
        f"{m['p95_latency_ms']:>9,.1f} {m['max_latency_ms']:>9,.1f}"
    )


def print_json_report(
    node_metrics: dict[str, dict[str, Any]],
    e2e_metrics: dict[str, Any],
    intent_dist: dict[str, int],
    hourly: list[dict[str, Any]],
    days: int,
) -> None:
    """메트릭을 JSON 형식으로 출력한다 (외부 도구 연동용)."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_days": days,
        "e2e_metrics": e2e_metrics,
        "node_metrics": node_metrics,
        "intent_distribution": intent_dist,
        "hourly_throughput": hourly,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


# ============================================================
# CLI 엔트리포인트
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="몽글픽 LangSmith Custom Dashboard 메트릭 수집",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="조회 기간 (일 단위, 기본: 1일)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("LANGCHAIN_PROJECT", "monglepick"),
        help="LangSmith 프로젝트 이름 (기본: LANGCHAIN_PROJECT 또는 'monglepick')",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="JSON 형식으로 출력 (외부 도구 연동용)",
    )

    args = parser.parse_args()

    # API 키 확인
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if not api_key:
        print("LANGCHAIN_API_KEY 환경변수가 설정되지 않았습니다.", file=sys.stderr)
        print(".env 파일에 LANGCHAIN_API_KEY=lsv2_pt_xxx 를 추가하세요.", file=sys.stderr)
        sys.exit(1)

    client = Client()

    # 조회 기간 설정 (UTC 기준)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=args.days)

    if not args.json_output:
        print(f"LangSmith 메트릭 수집 중... (프로젝트: {args.project}, 기간: {args.days}일)")
        print(f"  조회 범위: {start_time.isoformat()} ~ {end_time.isoformat()}")

    # 4가지 메트릭 병렬 수집 (순차 실행 — LangSmith API 호출)
    node_metrics = collect_node_metrics(client, args.project, start_time, end_time)
    e2e_metrics = collect_e2e_metrics(client, args.project, start_time, end_time)
    intent_dist = collect_intent_distribution(client, args.project, start_time, end_time)
    hourly = collect_hourly_throughput(client, args.project, start_time, end_time)

    # 출력
    if args.json_output:
        print_json_report(node_metrics, e2e_metrics, intent_dist, hourly, args.days)
    else:
        print_text_report(node_metrics, e2e_metrics, intent_dist, hourly, args.days)


if __name__ == "__main__":
    main()
