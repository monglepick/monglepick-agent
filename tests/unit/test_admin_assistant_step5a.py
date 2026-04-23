"""
관리자 AI 에이전트 Step 5a 단위 테스트 — HITL 승인 파이프라인.

범위:
1. risk_gate 노드 단위 — Tier<2 통과 / Tier>=2 interrupt 발동 / decision reject 처리
2. 그래프 E2E — Tier 2 tool 선택 후 astream 이 interrupt 지점에서 멈추고 snapshot.next 가
   남는지 / `confirmation_required` payload 를 snapshot.tasks.interrupts 에서 꺼낼 수 있는지
3. resume — Command(resume={'decision':'approve'}) 로 astream 재개 후 tool_executor 까지 실행
4. 거절 경로 — Command(resume={'decision':'reject'}) 이면 tool 실행 없이 거절 안내 문구로 종료
5. Tier 1 (query) 는 interrupt 없이 통과 (회귀 방어)
6. Tier 2 tool 레지스트리 등록 확인
"""

from __future__ import annotations

import uuid

import pytest
from langgraph.types import Command, Interrupt

from monglepick.agents.admin_assistant.graph import (
    admin_assistant_graph,
    run_admin_assistant_sync,
)
from monglepick.agents.admin_assistant.models import (
    AdminAssistantState,
    AdminIntent,
    ToolCall,
)
from monglepick.agents.admin_assistant.nodes import risk_gate
from monglepick.api.admin_backend_client import AdminApiResult
from monglepick.chains.admin_tool_selector_chain import SelectedTool
from monglepick.tools.admin_tools import ADMIN_TOOL_REGISTRY, ToolContext



# v3 Phase E: Tier 2 write tools 제거, HITL 비활성화
# 이 파일의 모든 테스트는 v2 HITL 메커니즘에 의존하므로 스킵처리
pytestmark = pytest.mark.skip(
    reason="v3 Phase D: Tier 2 write tools (faq_create, banner_create) 제거, risk_gate/interrupt 비활성화"
)

# ============================================================
# 1) Tier 2 tool 레지스트리 등록 확인
# ============================================================

class TestTier2Registration:
    def test_faq_create_registered(self):
        assert "faq_create" in ADMIN_TOOL_REGISTRY
        spec = ADMIN_TOOL_REGISTRY["faq_create"]
        assert spec.tier == 2
        assert spec.required_roles == {"SUPER_ADMIN", "ADMIN", "SUPPORT_ADMIN"}

    def test_banner_create_is_super_admin_only(self):
        """배너 등록은 시스템 정책 영역 — SUPER_ADMIN 전용."""
        spec = ADMIN_TOOL_REGISTRY["banner_create"]
        assert spec.tier == 2
        assert spec.required_roles == {"SUPER_ADMIN"}


# ============================================================
# 2) risk_gate 노드 단위
# ============================================================

class TestRiskGateNode:
    @pytest.mark.asyncio
    async def test_tier_0_passes_without_interrupt(self):
        """Tier 0 (stats) 은 interrupt 없이 통과 — return 값이 빈 dict."""
        state: AdminAssistantState = {
            "session_id": "sess-1",
            "pending_tool_call": ToolCall(
                tool_name="stats_overview", arguments={}, tier=0,
            ),
        }
        result = await risk_gate(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_tier_1_passes_without_interrupt(self):
        state: AdminAssistantState = {
            "session_id": "sess-2",
            "pending_tool_call": ToolCall(
                tool_name="users_list", arguments={}, tier=1,
            ),
        }
        result = await risk_gate(state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_pending_call_returns_empty(self):
        """pending_tool_call 이 None 이면 risk_gate 는 아무 것도 안 함."""
        state: AdminAssistantState = {"session_id": "sess-3", "pending_tool_call": None}
        result = await risk_gate(state)
        assert result == {}


# ============================================================
# 3) 그래프 E2E — Tier 2 → interrupt 발동 → 승인/거절 재개
# ============================================================

class TestHitlGraphE2E:
    @pytest.mark.asyncio
    async def test_tier2_interrupt_then_approve(self, monkeypatch, mock_ollama):
        """
        Tier 2 'action' intent → tool_selector 가 faq_create 선택 →
        risk_gate 에서 interrupt → snapshot.next 가 ('risk_gate',) 로 남음 →
        Command(resume={'decision':'approve'}) 로 재개 → tool_executor → narrator → END.
        """
        session_id = f"test-{uuid.uuid4().hex[:10]}"
        thread_config = {"configurable": {"thread_id": session_id}}

        # intent=action
        mock_ollama.set_structured_response(
            AdminIntent(kind="action", confidence=0.92, reason="FAQ 등록"),
        )
        # narrator 응답 (approve 후)
        mock_ollama.set_response(
            "FAQ '환불 규정' 을 등록했어요. [출처: faq_create]",
        )

        async def fake_select(**kwargs):
            return SelectedTool(
                name="faq_create",
                arguments={"category": "결제", "question": "환불?", "answer": "고객센터"},
                rationale="action 테스트",
            )
        monkeypatch.setattr(
            "monglepick.agents.admin_assistant.nodes.select_admin_tool",
            fake_select,
        )

        # faq_create handler 를 fake 로 교체 (Backend HTTP 우회)
        async def fake_handler(ctx: ToolContext, category: str, question: str,
                               answer: str, sortOrder: int | None = None):
            return AdminApiResult(
                ok=True, status_code=201,
                data={"faqId": 42, "category": category, "question": question},
                latency_ms=30,
            )
        ADMIN_TOOL_REGISTRY["faq_create"].handler = fake_handler

        # ── 1회차: interrupt 까지 실행 ──
        initial: AdminAssistantState = {
            "admin_id": "admin-001",
            "admin_role": "SUPPORT_ADMIN",
            "admin_jwt": "jwt-t",
            "session_id": session_id,
            "user_message": "환불 규정 FAQ 등록해줘",
            "history": [],
        }
        # astream 실행 후 snapshot.next 가 risk_gate 로 대기 상태여야 함
        async for _ in admin_assistant_graph.astream(
            initial, config=thread_config, stream_mode="updates",
        ):
            pass
        snapshot = await admin_assistant_graph.aget_state(thread_config)
        assert snapshot.next, "interrupt 후 대기 노드가 있어야 함"
        # interrupt payload 확인
        payload = None
        for task in snapshot.tasks:
            for intr in getattr(task, "interrupts", []) or []:
                if isinstance(intr.value, dict) and intr.value.get("tool_name") == "faq_create":
                    payload = intr.value
                    break
            if payload:
                break
        assert payload is not None, "confirmation_required payload 를 못 찾음"
        assert payload["tier"] == 2
        assert payload["arguments"]["category"] == "결제"

        # ── 2회차: 승인 재개 ──
        final_state: dict = {}
        async for chunk in admin_assistant_graph.astream(
            Command(resume={"decision": "approve", "comment": ""}),
            config=thread_config,
            stream_mode="updates",
        ):
            for _, updates in chunk.items():
                final_state.update(updates)

        # tool_executor 실행 결과가 cache 에 들어있어야 함
        cache = final_state.get("tool_results_cache") or {}
        assert any(isinstance(v, AdminApiResult) and v.ok for v in cache.values())
        # narrator 가 response_text 에 "등록" 포함 확정적으로 채웠어야 함
        response = final_state.get("response_text", "")
        assert "등록" in response

    @pytest.mark.asyncio
    async def test_tier2_interrupt_then_reject(self, monkeypatch, mock_ollama):
        """거절하면 tool 실행 없이 response_text 에 거절 안내만 남는다."""
        session_id = f"test-{uuid.uuid4().hex[:10]}"
        thread_config = {"configurable": {"thread_id": session_id}}

        mock_ollama.set_structured_response(
            AdminIntent(kind="action", confidence=0.92),
        )

        async def fake_select(**kwargs):
            return SelectedTool(
                name="faq_create",
                arguments={"category": "결제", "question": "X", "answer": "Y"},
                rationale="",
            )
        monkeypatch.setattr(
            "monglepick.agents.admin_assistant.nodes.select_admin_tool",
            fake_select,
        )

        # reject 경로 — handler 가 절대 불리지 않아야 함
        called = {"handler": 0}

        async def should_not_be_called(**kwargs):
            called["handler"] += 1
            return AdminApiResult(ok=True, status_code=200, data={})
        ADMIN_TOOL_REGISTRY["faq_create"].handler = should_not_be_called

        initial: AdminAssistantState = {
            "admin_id": "admin-001",
            "admin_role": "SUPER_ADMIN",
            "admin_jwt": "jwt-t",
            "session_id": session_id,
            "user_message": "FAQ 등록해줘",
            "history": [],
        }
        async for _ in admin_assistant_graph.astream(
            initial, config=thread_config, stream_mode="updates",
        ):
            pass

        # 거절 재개
        final_state: dict = {}
        async for chunk in admin_assistant_graph.astream(
            Command(resume={"decision": "reject", "comment": "오타 있음"}),
            config=thread_config,
            stream_mode="updates",
        ):
            for _, updates in chunk.items():
                final_state.update(updates)

        assert called["handler"] == 0  # handler 미호출
        response = final_state.get("response_text", "")
        assert "실행하지 않았" in response or "거부" in response
        # 사용자 메모가 문구에 반영됐는지
        assert "오타 있음" in response


# ============================================================
# 4) Tier 0/1 회귀 방어 — interrupt 없이 완결되어야 함
# ============================================================

class TestTier0And1RegressDoNotInterrupt:
    @pytest.mark.asyncio
    async def test_stats_flow_completes_without_interrupt(
        self, monkeypatch, mock_ollama,
    ):
        """Step 2 stats happy path 가 Step 5a 그래프에서도 동일하게 동작해야 함."""
        session_id = f"test-{uuid.uuid4().hex[:10]}"

        mock_ollama.set_structured_response(
            AdminIntent(kind="stats", confidence=0.95),
        )
        mock_ollama.set_response("DAU 1,234명. [출처: stats_overview]")

        async def fake_select(**kwargs):
            return SelectedTool(
                name="stats_overview", arguments={"period": "7d"}, rationale="",
            )
        monkeypatch.setattr(
            "monglepick.agents.admin_assistant.nodes.select_admin_tool",
            fake_select,
        )

        async def fake_handler(ctx: ToolContext, period: str = "7d"):
            return AdminApiResult(
                ok=True, status_code=200, data={"dau": 1234}, latency_ms=30,
            )
        ADMIN_TOOL_REGISTRY["stats_overview"].handler = fake_handler

        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="STATS_ADMIN",
            admin_jwt="jwt-t",
            session_id=session_id,
            user_message="DAU 알려줘",
        )
        # interrupt 없이 response_text 채워짐
        assert "1,234" in state.get("response_text", "")
        # snapshot.next 는 비어있어야 함
        snapshot = await admin_assistant_graph.aget_state(
            {"configurable": {"thread_id": session_id}},
        )
        assert not snapshot.next, "Tier 0 은 interrupt 없이 END 까지 가야 함"
