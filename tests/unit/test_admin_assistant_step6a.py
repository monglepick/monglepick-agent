"""
관리자 AI 에이전트 Step 6a 단위 테스트 — 감사 로그 자동 기록.

범위:
1. admin_audit_client.log_agent_action — JWT 없으면 조기 스킵, 2xx 성공/4xx 실패/타임아웃
   처리, description 포맷, JWT forwarding 헤더.
2. tool_executor 가 Tier 2 실행 후 log_agent_action 을 호출하는지 (spy). Tier 0/1 은 호출
   안 하는지 (감사 볼륨 폭증 방지 정책 §7.2).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from monglepick.agents.admin_assistant.models import ToolCall
from monglepick.agents.admin_assistant.nodes import tool_executor
from monglepick.api.admin_audit_client import log_agent_action
from monglepick.api.admin_backend_client import AdminApiResult
from monglepick.tools.admin_tools import ADMIN_TOOL_REGISTRY, ToolContext


# ============================================================
# 1) admin_audit_client.log_agent_action
# ============================================================

class TestLogAgentAction:
    @pytest.mark.asyncio
    async def test_no_jwt_returns_false_without_http_call(self, monkeypatch):
        """JWT 가 비어있으면 네트워크 호출 자체를 안 한다(dev 환경 보호)."""
        called = {"http": 0}

        async def _fail_http_client():
            called["http"] += 1
            raise AssertionError("JWT 없으면 HTTP 호출 금지")

        monkeypatch.setattr(
            "monglepick.api.admin_audit_client._get_admin_http_client",
            _fail_http_client,
        )
        ok = await log_agent_action(
            admin_jwt="",
            tool_name="faq_create",
            arguments={"category": "결제"},
            ok=True,
        )
        assert ok is False
        assert called["http"] == 0

    @pytest.mark.asyncio
    async def test_success_201_returns_true_with_expected_payload(self, monkeypatch):
        captured: dict = {}

        # post 가 호출되면 payload/headers 캡처, 201 응답 반환
        class _FakeResp:
            status_code = 201

            def json(self):
                return {}

        async def _fake_post(path, *, json, headers, timeout):
            captured["path"] = path
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return _FakeResp()

        class _FakeClient:
            async def post(self, path, **kwargs):
                return await _fake_post(path, **kwargs)

        async def _client():
            return _FakeClient()

        monkeypatch.setattr(
            "monglepick.api.admin_audit_client._get_admin_http_client", _client,
        )
        ok = await log_agent_action(
            admin_jwt="jwt-xyz",
            tool_name="faq_create",
            arguments={"category": "결제", "question": "환불?", "answer": "고객센터"},
            ok=True,
            user_prompt="환불 FAQ 등록해줘",
            target_type="FAQ",
            target_id="42",
            invocation_id="admin:abcd1234",
        )
        assert ok is True
        assert captured["path"] == "/api/v1/admin/audit-logs/agent"
        body = captured["json"]
        # 기본 actionType 은 AGENT_EXECUTED 로 채워진다
        assert body["actionType"] == "AGENT_EXECUTED"
        assert body["targetType"] == "FAQ"
        assert body["targetId"] == "42"
        desc = body["description"]
        assert "faq_create" in desc and "ok" in desc
        assert "환불 FAQ 등록해줘" in desc
        assert "category" in desc  # 인자 JSON 포함
        # 헤더: Bearer JWT + 추적 ID
        assert captured["headers"]["Authorization"] == "Bearer jwt-xyz"
        assert captured["headers"]["X-Agent-Invocation-Id"] == "admin:abcd1234"

    @pytest.mark.asyncio
    async def test_failure_included_in_description(self, monkeypatch):
        """ok=False 면 description 에 'fail' 태그 + error 사유가 포함된다."""
        captured: dict = {}

        class _FakeResp:
            status_code = 201

            def json(self):
                return {}

        class _FakeClient:
            async def post(self, path, **kwargs):
                captured["body"] = kwargs["json"]
                return _FakeResp()

        async def _client():
            return _FakeClient()

        monkeypatch.setattr(
            "monglepick.api.admin_audit_client._get_admin_http_client", _client,
        )
        await log_agent_action(
            admin_jwt="jwt-xyz",
            tool_name="banner_create",
            arguments={"title": "X", "imageUrl": "https://..."},
            ok=False,
            error="http_500",
        )
        desc = captured["body"]["description"]
        assert "fail" in desc
        assert "http_500" in desc

    @pytest.mark.asyncio
    async def test_non_2xx_returns_false_without_raising(self, monkeypatch):
        class _FakeResp:
            status_code = 403

            def json(self):
                return {"detail": "forbidden"}

        class _FakeClient:
            async def post(self, path, **kwargs):
                return _FakeResp()

        async def _client():
            return _FakeClient()

        monkeypatch.setattr(
            "monglepick.api.admin_audit_client._get_admin_http_client", _client,
        )
        ok = await log_agent_action(
            admin_jwt="jwt-xyz",
            tool_name="faq_create",
            arguments={},
            ok=True,
        )
        assert ok is False


# ============================================================
# 2) tool_executor 의 자동 감사 호출
# ============================================================

class TestToolExecutorAudit:
    pytestmark = pytest.mark.skip(
        reason="v3 Phase D: Tier 2 audit concept removed with write tools"
    )
    @pytest.mark.asyncio
    async def test_tier2_execution_triggers_audit(self, monkeypatch):
        """
        Tier 2 `faq_create` 실행 후 tool_executor 가 log_agent_action 을 호출해야 한다.
        실제 Backend 호출은 하지 않도록 handler 와 audit 클라이언트를 모두 패치.
        """
        # Tier 2 handler fake (실행 성공)
        async def fake_handler(ctx: ToolContext, category: str, question: str,
                               answer: str, sortOrder: int | None = None):
            return AdminApiResult(
                ok=True, status_code=201,
                data={"faqId": 42}, latency_ms=20,
            )
        monkeypatch.setattr(
            ADMIN_TOOL_REGISTRY["faq_create"], "handler", fake_handler,
        )

        # log_agent_action 을 AsyncMock 으로 감시 — 호출 여부/인자 확인
        audit_spy = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "monglepick.api.admin_audit_client.log_agent_action", audit_spy,
        )

        state = {
            "admin_id": "admin-001",
            "admin_role": "SUPPORT_ADMIN",
            "admin_jwt": "jwt-t",
            "session_id": f"t-{uuid.uuid4().hex[:8]}",
            "user_message": "환불 FAQ 등록해줘",
            "pending_tool_call": ToolCall(
                tool_name="faq_create",
                arguments={"category": "결제", "question": "환불?", "answer": "센터"},
                tier=2,
            ),
        }
        result = await tool_executor(state)
        # 성공적으로 캐시 저장
        assert result["latest_tool_ref_id"].startswith("tr_")

        # 감사 로그 1회 호출 — admin_jwt / tool_name / ok / user_prompt 동반
        audit_spy.assert_called_once()
        kwargs = audit_spy.call_args.kwargs
        assert kwargs["admin_jwt"] == "jwt-t"
        assert kwargs["tool_name"] == "faq_create"
        assert kwargs["ok"] is True
        assert kwargs["user_prompt"] == "환불 FAQ 등록해줘"
        # Tier 2 쓰기이므로 arguments 가 그대로 전달됐는지
        assert kwargs["arguments"]["category"] == "결제"

    @pytest.mark.asyncio
    async def test_tier0_execution_does_not_trigger_audit(self, monkeypatch):
        """Tier 0 (stats) 는 감사 미기록 — 볼륨 폭증 방지(§7.2)."""
        async def fake_handler(ctx: ToolContext, period: str = "7d"):
            return AdminApiResult(ok=True, status_code=200, data={"dau": 1})
        monkeypatch.setattr(
            ADMIN_TOOL_REGISTRY["stats_overview"], "handler", fake_handler,
        )
        audit_spy = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "monglepick.api.admin_audit_client.log_agent_action", audit_spy,
        )

        state = {
            "admin_id": "admin-001",
            "admin_role": "STATS_ADMIN",
            "admin_jwt": "jwt-t",
            "session_id": f"t-{uuid.uuid4().hex[:8]}",
            "user_message": "DAU 알려줘",
            "pending_tool_call": ToolCall(
                tool_name="stats_overview",
                arguments={"period": "7d"},
                tier=0,
            ),
        }
        result = await tool_executor(state)
        assert result["latest_tool_ref_id"].startswith("tr_")
        audit_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_tier2_failure_still_audited_with_ok_false(self, monkeypatch):
        """Tier 2 실행이 실패해도 감사 로그는 남는다(ok=False + error 포함)."""
        async def failing_handler(ctx: ToolContext, **kwargs):
            return AdminApiResult(ok=False, status_code=500, error="boom")
        monkeypatch.setattr(
            ADMIN_TOOL_REGISTRY["faq_create"], "handler", failing_handler,
        )
        audit_spy = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "monglepick.api.admin_audit_client.log_agent_action", audit_spy,
        )

        state = {
            "admin_id": "admin-001",
            "admin_role": "SUPPORT_ADMIN",
            "admin_jwt": "jwt-t",
            "session_id": f"t-{uuid.uuid4().hex[:8]}",
            "user_message": "FAQ 추가",
            "pending_tool_call": ToolCall(
                tool_name="faq_create",
                arguments={"category": "AI", "question": "?", "answer": "!"},
                tier=2,
            ),
        }
        await tool_executor(state)
        audit_spy.assert_called_once()
        kwargs = audit_spy.call_args.kwargs
        assert kwargs["ok"] is False
        assert kwargs["error"] == "boom"
