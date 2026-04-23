"""
관리자 AI 에이전트 Step 4 단위 테스트.

Step 4 범위 — Tier 1 Read-only 15개 tool + query intent 파이프라인.

테스트 영역:
1. 레지스트리 총 개수 20 + 도메인별 구성 확인
2. Role matrix — Tier 1 별 접근 제어 (STATS_ADMIN 은 Tier 0 만 / FINANCE_ADMIN 은 payment 만 등)
3. admin_backend_client.unwrap_api_response — Page 응답 row_count 재계산
4. tool handler smoke — users_list / orders_list / reports_list / faqs_list (httpx mock)
5. 그래프 E2E — query intent → users_list 호출 happy path
"""

from __future__ import annotations

import pytest

from monglepick.agents.admin_assistant.graph import run_admin_assistant_sync
from monglepick.agents.admin_assistant.models import AdminIntent, ToolCall
from monglepick.api.admin_backend_client import (
    AdminApiResult,
    unwrap_api_response,
)
from monglepick.chains.admin_tool_selector_chain import SelectedTool
from monglepick.tools.admin_tools import (
    ADMIN_TOOL_REGISTRY,
    ToolContext,
    list_tools_for_role,
)
from monglepick.tools.admin_tools.content_read import _handle_reports_list
from monglepick.tools.admin_tools.payment_read import _handle_orders_list
from monglepick.tools.admin_tools.support_read import _handle_faqs_list
from monglepick.tools.admin_tools.users_read import _handle_users_list


# ============================================================
# 1) 레지스트리 구성
# ============================================================

class TestRegistryComposition:
    def test_total_count_is_at_least_twenty(self):
        """
        Step 2 stats 5 + Step 4 users 5 + content 4 + payment 3 + support 3 = **20** (Tier 0/1).
        Step 5a (HITL) 부터는 Tier 2 tool 이 추가되므로 정확 일치 대신 최소 20 을 강제.
        """
        assert len(ADMIN_TOOL_REGISTRY) >= 20

    def test_step4_tools_registered(self):
        """Tier 1 15개가 모두 등록돼 있는지."""
        required = {
            # users_read (5)
            "users_list", "user_detail", "user_activity",
            "user_points_history", "user_payments",
            # content_read (4)
            "reports_list", "toxicity_list", "posts_list", "reviews_list",
            # payment_read (3)
            "orders_list", "order_detail", "subscriptions_list",
            # support_read (3)
            "faqs_list", "help_articles_list", "tickets_list",
        }
        assert required.issubset(set(ADMIN_TOOL_REGISTRY.keys()))


# ============================================================
# 2) Role matrix — Tier 1 접근 제어
# ============================================================

class TestRoleMatrixTier1:
    def test_stats_admin_gets_only_tier0(self):
        """STATS_ADMIN 은 Tier 0 5개만 접근 가능 (Tier 1 전부 거절)."""
        tools = list_tools_for_role("STATS_ADMIN")
        names = {t.name for t in tools}
        assert len(tools) == 5
        assert all(t.tier == 0 for t in tools)
        # 개인정보/결제 리소스 전부 차단
        assert "users_list" not in names
        assert "orders_list" not in names
        assert "reports_list" not in names

    def test_finance_admin_sees_payment_plus_stats_plus_users(self):
        """FINANCE_ADMIN: stats(5) + payment(3) + users(5) = 13개."""
        tools = list_tools_for_role("FINANCE_ADMIN")
        names = {t.name for t in tools}
        assert "orders_list" in names
        assert "subscriptions_list" in names
        # 모더레이션(posts/reviews)과 support 는 제외
        assert "posts_list" not in names
        assert "faqs_list" not in names
        assert "tickets_list" not in names

    def test_moderator_sees_content_plus_stats_plus_users(self):
        """MODERATOR: stats(5) + content(4) + users(5) = 14개."""
        tools = list_tools_for_role("MODERATOR")
        names = {t.name for t in tools}
        assert "reports_list" in names
        assert "posts_list" in names
        # 결제·support 는 접근 불가
        assert "orders_list" not in names
        assert "faqs_list" not in names

    def test_support_admin_sees_support_plus_stats_plus_users(self):
        """SUPPORT_ADMIN: stats(5) + support(3) + users(5) = 13개."""
        tools = list_tools_for_role("SUPPORT_ADMIN")
        names = {t.name for t in tools}
        assert "tickets_list" in names
        assert "faqs_list" in names
        assert "reports_list" not in names
        assert "orders_list" not in names

    def test_super_admin_sees_everything(self):
        # Step 5a 부터 Tier 2 tool 추가로 수가 늘 수 있어 정확 일치 대신 레지스트리 전체와 비교.
        assert len(list_tools_for_role("SUPER_ADMIN")) == len(ADMIN_TOOL_REGISTRY)


# ============================================================
# 3) unwrap_api_response — Page 응답 row_count 재계산
# ============================================================

class TestUnwrapPageResponse:
    def test_spring_page_content_row_count(self):
        """Backend Page 응답: {data:{content:[...], totalElements, ...}} 언래핑."""
        wrapped = AdminApiResult(
            ok=True, status_code=200,
            data={
                "success": True,
                "data": {
                    "content": [{"id": 1}, {"id": 2}, {"id": 3}],
                    "totalElements": 100,
                    "totalPages": 5,
                    "number": 0,
                    "size": 20,
                },
                "error": None,
            },
        )
        result = unwrap_api_response(wrapped)
        assert isinstance(result.data, dict)
        assert result.data["totalElements"] == 100
        # row_count = len(content) 자동 재계산 — narrator 프롬프트에 "표시 3건" 힌트
        assert result.row_count == 3

    def test_plain_list_response(self):
        """리스트 직접 응답도 row_count 자동 계산 (기존 Step 2 케이스 회귀)."""
        wrapped = AdminApiResult(
            ok=True, status_code=200,
            data={"success": True, "data": [10, 20, 30]},
        )
        result = unwrap_api_response(wrapped)
        assert result.data == [10, 20, 30]
        assert result.row_count == 3


# ============================================================
# 4) Tier 1 handler smoke (httpx mock)
# ============================================================

class TestTier1Handlers:
    @staticmethod
    def _ctx(role: str = "SUPER_ADMIN") -> ToolContext:
        return ToolContext(
            admin_jwt="jwt-xxx",
            admin_role=role,
            admin_id="admin-001",
            session_id="sess-001",
            invocation_id="admin:abcd",
        )

    @pytest.mark.asyncio
    async def test_users_list_passes_filters(self, monkeypatch):
        captured: dict = {}

        async def fake_get_admin_json(path, *, admin_jwt, params=None,
                                      invocation_id="", timeout=None):
            captured["path"] = path
            captured["params"] = params
            return AdminApiResult(
                ok=True, status_code=200,
                data={
                    "success": True,
                    "data": {"content": [], "totalElements": 0, "size": 20, "number": 0},
                },
            )

        # users_read 모듈 네임스페이스의 get_admin_json 을 교체.
        monkeypatch.setattr(
            "monglepick.tools.admin_tools.users_read.get_admin_json",
            fake_get_admin_json,
        )

        result = await _handle_users_list(
            self._ctx(), keyword="gmail", status="ACTIVE", role="USER",
            page=0, size=20,
        )
        assert result.ok
        assert captured["path"] == "/api/v1/admin/users"
        # 빈 문자열 필터는 포함하지 않아야 한다
        assert captured["params"] == {
            "page": 0, "size": 20,
            "keyword": "gmail", "status": "ACTIVE", "role": "USER",
        }

    @pytest.mark.asyncio
    async def test_orders_list_drops_blank_filters(self, monkeypatch):
        captured: dict = {}

        async def fake(path, *, admin_jwt, params=None, invocation_id="", timeout=None):
            captured["path"] = path
            captured["params"] = params
            return AdminApiResult(
                ok=True, status_code=200,
                data={"success": True, "data": {"content": [], "totalElements": 0}},
            )

        monkeypatch.setattr(
            "monglepick.tools.admin_tools.payment_read.get_admin_json", fake,
        )

        await _handle_orders_list(
            self._ctx("FINANCE_ADMIN"), status="REFUNDED",
        )
        assert captured["path"] == "/api/v1/admin/payment/orders"
        # status 만 전달, orderType/userId 는 빈 문자열이라 params 에서 제외
        assert captured["params"] == {"page": 0, "size": 20, "status": "REFUNDED"}

    @pytest.mark.asyncio
    async def test_reports_list_handles_http_error(self, monkeypatch):
        async def fake(path, *, admin_jwt, params=None, invocation_id="", timeout=None):
            return AdminApiResult(
                ok=False, status_code=403, error="forbidden",
            )

        monkeypatch.setattr(
            "monglepick.tools.admin_tools.content_read.get_admin_json", fake,
        )
        result = await _handle_reports_list(self._ctx("MODERATOR"), status="pending")
        assert not result.ok
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_faqs_list_default_params(self, monkeypatch):
        captured: dict = {}

        async def fake(path, *, admin_jwt, params=None, invocation_id="", timeout=None):
            captured["params"] = params
            return AdminApiResult(
                ok=True, status_code=200,
                data={"success": True, "data": {"content": []}},
            )

        monkeypatch.setattr(
            "monglepick.tools.admin_tools.support_read.get_admin_json", fake,
        )
        await _handle_faqs_list(self._ctx("SUPPORT_ADMIN"))
        # 카테고리 없으면 page/size 만 전달
        assert captured["params"] == {"page": 0, "size": 20}


# ============================================================
# 5) 그래프 E2E — query intent happy path
# ============================================================

class TestGraphQueryE2E:
    @pytest.mark.asyncio
    async def test_query_happy_path(self, monkeypatch, mock_ollama):
        """
        'user_id=abc 결제 내역 보여줘' → intent=query → tool_selector →
          user_payments tool → tool_executor → narrator → response_formatter.
        """
        # intent=query
        mock_ollama.set_structured_response(
            AdminIntent(kind="query", confidence=0.9, reason="유저 결제 조회"),
        )
        # narrator 응답
        mock_ollama.set_response(
            "user_id=abc 의 결제 주문 3건을 찾았어요. [출처: user_payments · page=0]",
        )

        async def fake_select(**kwargs):
            return SelectedTool(
                name="user_payments",
                arguments={"userId": "abc", "page": 0, "size": 20},
                rationale="fake",
            )
        monkeypatch.setattr(
            "monglepick.agents.admin_assistant.nodes.select_admin_tool",
            fake_select,
        )

        async def fake_handler(ctx: ToolContext, userId: str,
                               page: int = 0, size: int = 20):
            return AdminApiResult(
                ok=True, status_code=200,
                data={"content": [{"orderId": "A"}, {"orderId": "B"}, {"orderId": "C"}],
                      "totalElements": 3},
                row_count=3,
                latency_ms=50,
            )
        # monkeypatch.setattr 로 격리 보장 — 직접 대입 시 다른 테스트(예: TestAdminAssistantGraph)
        # 의 graph 호출에서 fake handler 가 leak 되어 KeyError 'risk_gate' 등 예측 불가 회귀 발생.
        monkeypatch.setattr(
            ADMIN_TOOL_REGISTRY["user_payments"], "handler", fake_handler,
        )

        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="SUPER_ADMIN",
            admin_jwt="jwt-t",
            session_id="",
            user_message="user_id=abc 결제 내역 보여줘",
        )
        assert state.get("intent").kind == "query"
        call = state.get("pending_tool_call")
        assert isinstance(call, ToolCall)
        assert call.tool_name == "user_payments"
        assert call.tier == 1
        assert call.arguments["userId"] == "abc"
        assert "3건" in state.get("response_text", "")

    @pytest.mark.asyncio
    async def test_query_no_matching_tool_uses_placeholder(
        self, monkeypatch, mock_ollama,
    ):
        """selector 가 None 반환 시 query placeholder ("적합한 도구를 찾지 못했어요")."""
        mock_ollama.set_structured_response(
            AdminIntent(kind="query", confidence=0.9),
        )
        async def fake_select(**kwargs):
            return None
        monkeypatch.setattr(
            "monglepick.agents.admin_assistant.nodes.select_admin_tool",
            fake_select,
        )
        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="SUPER_ADMIN",
            admin_jwt="",
            session_id="",
            user_message="지구 반대편 날씨 알려줘",
        )
        assert state.get("pending_tool_call") is None
        assert "적합한 도구" in state.get("response_text", "")
