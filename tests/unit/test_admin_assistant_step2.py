"""
관리자 AI 에이전트 Step 2 단위 테스트.

Step 2 범위 — stats intent 에서 실제 Admin Stats API 호출 파이프라인.

테스트 영역:
1. admin_backend_client — summarize_for_llm (list/dict/스칼라/에러)
2. stats tool — _unwrap_api_response (ApiResponse 래퍼 언래핑)
3. Tool 레지스트리 — role matrix 필터
4. tool_selector 노드 — fake select_admin_tool 로 pending_tool_call 설정
5. tool_executor 노드 — 성공/실패/tier block/role 거절/args 검증 실패
6. narrator 노드 — Solar fake 응답
7. 그래프 E2E — stats intent 의 전체 경로
"""

from __future__ import annotations

import pytest

from monglepick.agents.admin_assistant.graph import run_admin_assistant_sync
from monglepick.agents.admin_assistant.models import (
    AdminAssistantState,
    AdminIntent,
    ToolCall,
)
from monglepick.agents.admin_assistant.nodes import (
    narrator,
    tool_executor,
    tool_selector,
)
from monglepick.api.admin_backend_client import AdminApiResult, summarize_for_llm
from monglepick.chains.admin_tool_selector_chain import SelectedTool
from monglepick.tools.admin_tools import (
    ADMIN_TOOL_REGISTRY,
    ToolContext,
    list_tools_for_role,
)
from monglepick.tools.admin_tools.stats import _unwrap_api_response


# ============================================================
# 1) admin_backend_client.summarize_for_llm
# ============================================================

class TestSummarizeForLlm:
    def test_list_truncated(self):
        rows = [{"id": i} for i in range(10)]
        result = AdminApiResult(ok=True, status_code=200, data=rows, row_count=10)
        summary = summarize_for_llm(result, sample_rows=3)
        assert summary["ok"] is True
        assert summary["total_rows"] == 10
        assert len(summary["sample_rows"]) == 3
        assert summary["truncated"] is True

    def test_dict_with_long_string_shrunk(self):
        result = AdminApiResult(
            ok=True, status_code=200,
            data={"note": "A" * 1000, "count": 42},
        )
        summary = summarize_for_llm(result, max_str_len=50)
        assert summary["data"]["count"] == 42
        assert summary["data"]["note"].endswith("...(생략)")
        assert len(summary["data"]["note"]) <= 60

    def test_scalar_passthrough(self):
        result = AdminApiResult(ok=True, status_code=200, data=42)
        summary = summarize_for_llm(result)
        assert summary["data"] == 42

    def test_error_returns_minimal(self):
        result = AdminApiResult(ok=False, status_code=401, error="unauthorized")
        summary = summarize_for_llm(result)
        assert summary["ok"] is False
        assert summary["status_code"] == 401
        assert summary["error"] == "unauthorized"
        assert "data" not in summary


# ============================================================
# 2) stats._unwrap_api_response
# ============================================================

class TestUnwrapApiResponse:
    def test_wrapper_unwrapped(self):
        """Backend 의 {success, data, error} 래퍼에서 내부 data 만 꺼낸다."""
        wrapped = AdminApiResult(
            ok=True, status_code=200,
            data={"success": True, "data": {"dau": 1234}, "error": None},
        )
        unwrapped = _unwrap_api_response(wrapped)
        assert unwrapped.data == {"dau": 1234}

    def test_list_data_row_count_recomputed(self):
        wrapped = AdminApiResult(
            ok=True, status_code=200,
            data={"success": True, "data": [1, 2, 3, 4]},
            row_count=None,
        )
        unwrapped = _unwrap_api_response(wrapped)
        assert unwrapped.data == [1, 2, 3, 4]
        assert unwrapped.row_count == 4

    def test_bare_response_unchanged(self):
        """래퍼가 없는 응답은 원본 그대로."""
        bare = AdminApiResult(ok=True, status_code=200, data={"dau": 100})
        assert _unwrap_api_response(bare).data == {"dau": 100}

    def test_error_result_untouched(self):
        err = AdminApiResult(ok=False, status_code=500, error="boom")
        assert _unwrap_api_response(err) is err


# ============================================================
# 3) Tool 레지스트리 role matrix
# ============================================================

class TestToolRegistry:
    def test_five_tools_registered(self):
        assert len(ADMIN_TOOL_REGISTRY) >= 5
        for name in [
            "stats_overview", "stats_trends", "stats_revenue",
            "stats_ai_service_overview", "stats_community_overview",
        ]:
            assert name in ADMIN_TOOL_REGISTRY

    def test_super_admin_sees_all(self):
        tools = list_tools_for_role("SUPER_ADMIN")
        assert len(tools) == len(ADMIN_TOOL_REGISTRY)

    def test_stats_admin_sees_stats(self):
        """STATS_ADMIN 은 Tier 0 stats 전부 접근 가능."""
        tools = list_tools_for_role("STATS_ADMIN")
        names = {t.name for t in tools}
        assert "stats_overview" in names

    def test_empty_role_rejects_all(self):
        assert list_tools_for_role("") == []


# ============================================================
# 4) tool_selector 노드 (fake select_admin_tool)
# ============================================================

class TestToolSelectorNode:
    @pytest.mark.asyncio
    async def test_selects_and_injects_tier(self, monkeypatch):
        async def fake_select(user_message, admin_role, intent_kind, request_id="",
                            tool_history_summary=None, hop_count=0, max_hops=5,
                            allowed_tool_names=None):
            return SelectedTool(
                name="stats_overview",
                arguments={"period": "7d"},
                rationale="fake",
            )

        monkeypatch.setattr(
            "monglepick.agents.admin_assistant.nodes.select_admin_tool",
            fake_select,
        )

        state: AdminAssistantState = {
            "admin_id": "admin-001",
            "admin_role": "STATS_ADMIN",
            "user_message": "DAU 알려줘",
            "intent": AdminIntent(kind="stats", confidence=0.9),
        }
        result = await tool_selector(state)
        assert isinstance(result["pending_tool_call"], ToolCall)
        assert result["pending_tool_call"].tool_name == "stats_overview"
        # 레지스트리에서 tier 자동 주입
        assert result["pending_tool_call"].tier == 0

    @pytest.mark.asyncio
    async def test_no_role_returns_none(self, monkeypatch):
        async def fake_select(**kwargs):
            raise AssertionError("LLM 호출이 발생하면 안 됨")

        monkeypatch.setattr(
            "monglepick.agents.admin_assistant.nodes.select_admin_tool",
            fake_select,
        )
        state: AdminAssistantState = {
            "admin_id": "", "admin_role": "",
            "user_message": "DAU", "intent": AdminIntent(kind="stats"),
        }
        result = await tool_selector(state)
        assert result["pending_tool_call"] is None


# ============================================================
# 5) tool_executor 노드
# ============================================================

class TestToolExecutorNode:
    @pytest.mark.asyncio
    async def test_success_flow(self, monkeypatch):
        """레지스트리 handler 가 ok=True 반환 → cache 저장 + ref_id 발급."""

        async def fake_handler(ctx: ToolContext, period: str = "7d"):
            return AdminApiResult(
                ok=True, status_code=200,
                data={"dau": 1234}, latency_ms=42,
            )

        # stats_overview handler 만 swap (다른 tool 영향 없음).
        # monkeypatch.setattr 만 사용 — 직접 대입(`.handler = fake`) 은 teardown 안 돼서
        # 다른 테스트로 leak 됨 (격리 깨뜨림). pytest 가 finalize 단계에 원본 handler 복구.
        monkeypatch.setattr(
            ADMIN_TOOL_REGISTRY["stats_overview"], "handler", fake_handler,
        )

        state: AdminAssistantState = {
            "admin_id": "admin-001",
            "admin_role": "STATS_ADMIN",
            "admin_jwt": "jwt-xyz",
            "session_id": "sess-001",
            "pending_tool_call": ToolCall(
                tool_name="stats_overview",
                arguments={"period": "7d"},
                tier=0,
            ),
        }
        result = await tool_executor(state)
        assert result["latest_tool_ref_id"].startswith("tr_")
        cached = result["tool_results_cache"][result["latest_tool_ref_id"]]
        assert isinstance(cached, AdminApiResult)
        assert cached.ok is True
        assert cached.data == {"dau": 1234}

    @pytest.mark.asyncio
    async def test_role_denied(self):
        """권한 없는 role 로 실행 시도 시 차단 (캐시 미갱신)."""
        state: AdminAssistantState = {
            "admin_role": "",  # 어떤 tool 도 허용 안됨
            "pending_tool_call": ToolCall(
                tool_name="stats_overview", arguments={}, tier=0,
            ),
        }
        result = await tool_executor(state)
        assert result["latest_tool_ref_id"] == ""

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        state: AdminAssistantState = {
            "admin_role": "SUPER_ADMIN",
            "pending_tool_call": ToolCall(
                tool_name="nonexistent_tool", arguments={}, tier=0,
            ),
        }
        result = await tool_executor(state)
        assert result["latest_tool_ref_id"] == ""

    @pytest.mark.asyncio
    async def test_args_validation_fail_captured(self, monkeypatch):
        """args 검증 실패도 tool_results_cache 에 error 결과로 저장된다."""

        # _PeriodArgs 는 "7d/30d/90d" 외 거부. 이상값 주입.
        state: AdminAssistantState = {
            "admin_role": "STATS_ADMIN",
            "pending_tool_call": ToolCall(
                tool_name="stats_overview",
                arguments={"period": "999d"},  # Literal 위반
                tier=0,
            ),
        }
        result = await tool_executor(state)
        ref_id = result["latest_tool_ref_id"]
        assert ref_id.startswith("tr_")
        cached = result["tool_results_cache"][ref_id]
        assert isinstance(cached, AdminApiResult)
        assert cached.ok is False
        assert "args_validation" in cached.error


# ============================================================
# 6) narrator 노드
# ============================================================

class TestNarratorNode:
    @pytest.mark.asyncio
    async def test_narrator_uses_cached_result(self, mock_ollama):
        mock_ollama.set_response(
            "최근 7일 DAU는 1,234명이에요. [출처: stats_overview · 기간=7d]"
        )
        state: AdminAssistantState = {
            "user_message": "DAU 알려줘",
            "pending_tool_call": ToolCall(
                tool_name="stats_overview",
                arguments={"period": "7d"},
                tier=0,
            ),
            "latest_tool_ref_id": "tr_abc",
            "tool_results_cache": {
                "tr_abc": AdminApiResult(
                    ok=True, status_code=200, data={"dau": 1234},
                ),
            },
        }
        result = await narrator(state)
        assert "1,234" in result["response_text"]

    @pytest.mark.asyncio
    async def test_no_ref_id_noop(self, mock_ollama):
        """ref_id 가 없으면 아무 것도 하지 않고 response_text 미설정."""
        state: AdminAssistantState = {"latest_tool_ref_id": "", "tool_results_cache": {}}
        result = await narrator(state)
        assert result == {}


# ============================================================
# 7) 그래프 E2E — stats intent 전체 경로
# ============================================================

class TestGraphStatsE2E:
    @pytest.mark.asyncio
    async def test_stats_happy_path(self, monkeypatch, mock_ollama):
        """
        '지난 7일 DAU 알려줘' → intent=stats → tool_selector → tool_executor
          → narrator → response_formatter 전체 경로.
        """
        # 1) Intent 분류: stats
        mock_ollama.set_structured_response(
            AdminIntent(kind="stats", confidence=0.95, reason="DAU"),
        )
        # 2) narrator 자유텍스트 응답
        mock_ollama.set_response(
            "최근 7일 DAU는 1,234명이에요. 🎬 [출처: stats_overview · 기간=7d]"
        )

        # 3) select_admin_tool 을 fake 로 교체 (Solar bind_tools 우회)
        # v3 Phase D: tool_selector 가 tool_history_summary/hop_count/max_hops 를 추가 전달하므로 **kwargs 수용
        async def fake_select(user_message, admin_role, intent_kind, request_id="", **_ignored):
            return SelectedTool(
                name="stats_overview",
                arguments={"period": "7d"},
                rationale="fake-e2e",
            )
        monkeypatch.setattr(
            "monglepick.agents.admin_assistant.nodes.select_admin_tool",
            fake_select,
        )

        # 4) stats_overview handler 를 fake 로 — Backend HTTP 호출 우회.
        # monkeypatch.setattr 로 격리 보장 (직접 대입 시 다른 테스트로 leak 발생).
        async def fake_handler(ctx: ToolContext, period: str = "7d"):
            return AdminApiResult(
                ok=True, status_code=200,
                data={"dau": 1234, "mau": 9876},
                latency_ms=42,
            )
        monkeypatch.setattr(
            ADMIN_TOOL_REGISTRY["stats_overview"], "handler", fake_handler,
        )

        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="STATS_ADMIN",
            admin_jwt="jwt-test",
            session_id="",
            user_message="지난 7일 DAU 알려줘",
        )
        assert state.get("intent").kind == "stats"
        call = state.get("pending_tool_call")
        assert isinstance(call, ToolCall)
        assert call.tool_name == "stats_overview"
        assert state.get("latest_tool_ref_id", "").startswith("tr_")
        response = state.get("response_text", "")
        assert "1,234" in response

    @pytest.mark.asyncio
    async def test_stats_no_tool_goes_to_smart_fallback(
        self, monkeypatch, mock_ollama,
    ):
        """
        Step 6c(2026-04-23): stats intent 이지만 selector None → smart_fallback_responder
        가 LLM 역제안을 생성한다 (기존 placeholder 직결 대신).
        """
        mock_ollama.set_structured_response(
            AdminIntent(kind="stats", confidence=0.9),
        )
        mock_ollama.set_response(
            "해당 수치 통계 도구는 없어요. 'DAU 추이' 나 '매출 추이' 로 물어봐 주실래요?"
        )

        async def fake_select(**kwargs):
            return None
        monkeypatch.setattr(
            "monglepick.agents.admin_assistant.nodes.select_admin_tool",
            fake_select,
        )

        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="STATS_ADMIN",
            admin_jwt="",
            session_id="",
            user_message="우리 회사 주식 가격 얼마야?",  # tool 범위 밖
        )
        assert state.get("pending_tool_call") is None
        # smart_fallback_responder 가 LLM 응답을 response_text 에 담아줌 (빈 응답 아님)
        assert len(state.get("response_text", "")) > 0
