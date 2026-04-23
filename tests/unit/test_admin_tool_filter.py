"""
tool_filter.py 단위 테스트 — Step 7a 재설계(2026-04-23) 카테고리 기반 필터.

Qdrant 없이 규칙·키워드만으로 76 tool 을 intent_kind·발화에 맞게 축소하는 게 이 모듈의 책임.
검증 범위:
- classify_tool_kind(): 이름 컨벤션 기반 kind 분류
- shortlist_tools_by_category(): intent + keyword 힌트 교집합, fallback, max_tools 절단
"""

from __future__ import annotations

import pytest

from monglepick.tools.admin_tools.tool_filter import (
    classify_tool_kind,
    shortlist_tools_by_category,
)


# ============================================================
# 1) classify_tool_kind — 이름 컨벤션 분류
# ============================================================

@pytest.mark.parametrize(
    "tool_name, expected_kind",
    [
        # Stats prefix
        ("stats_overview", "stats"),
        ("stats_revenue", "stats"),
        ("stats_churn_risk", "stats"),
        ("dashboard_kpi", "stats"),
        ("dashboard_trends", "stats"),
        # Draft suffix — 우선순위 최상위 (stats_draft 가상 케이스도 draft 로 처리)
        ("notice_draft", "draft"),
        ("banner_draft", "draft"),
        ("point_pack_draft", "draft"),
        # Navigate prefix
        ("goto_user_detail", "navigate"),
        ("goto_order_refund", "navigate"),
        ("goto_audit_log", "navigate"),
        # 그 외는 read 로 폴백
        ("users_list", "read"),
        ("user_detail", "read"),
        ("reports_list", "read"),
        ("order_detail", "read"),
        ("chat_suggestions_list", "read"),
    ],
)
def test_classify_tool_kind(tool_name: str, expected_kind: str):
    """이름 컨벤션 → kind 분류 전수 검증."""
    assert classify_tool_kind(tool_name) == expected_kind


# ============================================================
# 2) shortlist_tools_by_category — 기본 동작
# ============================================================

def test_empty_role_returns_empty_list():
    """admin_role 이 빈 문자열이면 즉시 [] 반환 (권한 없음)."""
    assert shortlist_tools_by_category(
        user_message="아무거나",
        admin_role="",
        intent_kind="stats",
    ) == []


def test_stats_intent_returns_only_stats_category():
    """intent=stats 일 때 stats_* / dashboard_* 로만 좁혀져야 한다."""
    names = shortlist_tools_by_category(
        user_message="지난 7일 매출",
        admin_role="SUPER_ADMIN",
        intent_kind="stats",
    )
    assert names, "stats intent 에서 최소 1개는 반환돼야 한다"
    assert all(
        n.startswith("stats_") or n.startswith("dashboard_") for n in names
    ), f"stats 카테고리 외 이름이 섞여 있음: {names}"


def test_action_intent_with_notice_keyword_prioritizes_notice_tools():
    """intent=action + '공지' 키워드 → notice_draft / notices_list 가 상단에 배치."""
    names = shortlist_tools_by_category(
        user_message="공지사항 초안 하나 만들어줘",
        admin_role="SUPER_ADMIN",
        intent_kind="action",
        max_tools=30,
    )
    assert "notice_draft" in names
    assert names.index("notice_draft") < 5, (
        f"키워드 힌트가 상단(top5) 에 배치되지 않음: {names[:10]}"
    )


def test_action_intent_with_refund_keyword_prioritizes_goto_refund():
    """intent=action + '환불' 키워드 → goto_order_refund 가 상단."""
    names = shortlist_tools_by_category(
        user_message="chulsoo 환불해줘",
        admin_role="SUPER_ADMIN",
        intent_kind="action",
        max_tools=30,
    )
    assert "goto_order_refund" in names
    assert names.index("goto_order_refund") < 5


def test_max_tools_cap_is_applied():
    """max_tools 상한이 결과 길이를 절단한다."""
    names = shortlist_tools_by_category(
        user_message="아무거나",
        admin_role="SUPER_ADMIN",
        intent_kind="action",
        max_tools=5,
    )
    assert len(names) <= 5


def test_query_intent_excludes_draft_tools():
    """intent=query 는 read/stats/navigate 만 허용하고 draft 는 제외."""
    names = shortlist_tools_by_category(
        user_message="유저 목록 보여줘",
        admin_role="SUPER_ADMIN",
        intent_kind="query",
        max_tools=50,
    )
    draft_hits = [n for n in names if n.endswith("_draft")]
    assert draft_hits == [], f"query intent 에 draft tool 이 포함됨: {draft_hits}"


def test_role_filter_applied_before_category():
    """role 권한 밖 tool 은 카테고리 필터에서 제외된다.

    STATS_ADMIN 은 stats/dashboard 계열만 접근 가능하므로 query intent 로 호출해도
    read 계열 (users_list 등) 은 반환되면 안 된다.
    """
    names = shortlist_tools_by_category(
        user_message="유저 찾아줘",
        admin_role="STATS_ADMIN",
        intent_kind="query",
        max_tools=50,
    )
    assert "users_list" not in names, (
        "STATS_ADMIN 에 없는 users_list 가 반환됨 — Role 필터 누락"
    )
