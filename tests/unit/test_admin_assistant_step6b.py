"""
관리자 AI 에이전트 Step 6b 단위 테스트.

범위:
1. Tier 3 tool 2개 레지스트리 — user_suspend / points_manual_adjust + confirm_keyword.
2. Role matrix — Tier 3 는 SUPER_ADMIN/ADMIN (+FINANCE for points) 만.
3. ToolSpec.confirm_keyword 가 risk_gate → ConfirmationPayload.required_keyword 로 전달.
4. user_suspend handler — GET(before) → PUT(실행) → GET(after) 3회 호출 + AdminApiResult
   에 before/after 주입.
5. points_manual_adjust handler — before 잔액 스냅샷 + Backend 응답을 after_data 로 보존.
6. tool_executor 가 audit 호출에 before_data/after_data + target_type/target_id 를 전달.
7. admin_backend_client.write_admin_json — PUT 메서드가 client.request('PUT', ...) 로 위임.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from monglepick.agents.admin_assistant.graph import admin_assistant_graph
from monglepick.agents.admin_assistant.models import (
    AdminIntent,
    ConfirmationPayload,
    ToolCall,
)
from monglepick.agents.admin_assistant.nodes import _infer_audit_target, tool_executor
from monglepick.api.admin_backend_client import AdminApiResult, write_admin_json
from monglepick.chains.admin_tool_selector_chain import SelectedTool
from monglepick.tools.admin_tools import (
    ADMIN_TOOL_REGISTRY,
    ToolContext,
    list_tools_for_role,
)
from monglepick.tools.admin_tools.points_write import _handle_points_manual_adjust
from monglepick.tools.admin_tools.users_write import _handle_user_suspend


# ============================================================
# 1) Tier 3 레지스트리 등록
# ============================================================

class TestTier3Registration:
    def test_user_suspend_registered(self):
        assert "user_suspend" in ADMIN_TOOL_REGISTRY
        spec = ADMIN_TOOL_REGISTRY["user_suspend"]
        assert spec.tier == 3
        assert spec.required_roles == {"SUPER_ADMIN", "ADMIN"}
        assert spec.confirm_keyword == "정지"

    def test_points_manual_adjust_registered(self):
        spec = ADMIN_TOOL_REGISTRY["points_manual_adjust"]
        assert spec.tier == 3
        assert spec.required_roles == {"SUPER_ADMIN", "ADMIN", "FINANCE_ADMIN"}
        assert spec.confirm_keyword == "포인트 조정"


# ============================================================
# 2) Role matrix (Step 6b 추가 반영)
# ============================================================

class TestStep6bRoleMatrix:
    def test_moderator_cannot_suspend_or_adjust_points(self):
        """MODERATOR 는 Tier 3 계정/포인트 접근 불가."""
        names = {t.name for t in list_tools_for_role("MODERATOR")}
        assert "user_suspend" not in names
        assert "points_manual_adjust" not in names

    def test_finance_admin_can_adjust_points_but_not_suspend(self):
        names = {t.name for t in list_tools_for_role("FINANCE_ADMIN")}
        assert "points_manual_adjust" in names
        assert "user_suspend" not in names

    def test_stats_admin_sees_only_tier0(self):
        """Step 6b 이후에도 STATS_ADMIN 은 여전히 Tier 0 만 (5개)."""
        names = {t.name for t in list_tools_for_role("STATS_ADMIN")}
        assert len(names) == 5
        assert all(ADMIN_TOOL_REGISTRY[n].tier == 0 for n in names)


# ============================================================
# 3) risk_gate 가 confirm_keyword 를 ConfirmationPayload 에 전달
# ============================================================

class TestRiskGateInjectsKeyword:
    """
    risk_gate 가 ToolSpec.confirm_keyword 를 ConfirmationPayload.required_keyword 로
    실어 보내는지 — 실제 interrupt 를 태우지 않고 payload 생성 로직만 직접 검증.
    graph 실행은 Step 5a 테스트에서 이미 커버했으므로 여기선 단위 검증.
    """

    def test_confirmation_payload_carries_keyword(self):
        """ConfirmationPayload 자체가 required_keyword 를 필드로 가지는지."""
        payload = ConfirmationPayload(
            tool_name="user_suspend",
            tier=3,
            required_keyword="정지",
        )
        dumped = payload.model_dump()
        assert dumped["required_keyword"] == "정지"

    @pytest.mark.asyncio
    async def test_risk_gate_reads_spec_confirm_keyword(self, monkeypatch):
        """
        risk_gate 에서 interrupt() 가 호출될 때 넘기는 dict 에 required_keyword 가
        포함되는지 간접 검증 — interrupt 를 가짜로 swap 해 인자를 캡처.
        """
        from monglepick.agents.admin_assistant import nodes as node_mod

        captured: dict = {}

        def fake_interrupt(payload_dict):
            captured.update(payload_dict)
            # interrupt 는 원래 graph 에서 재개 시 값을 반환 — 테스트에서는 즉시 reject 로.
            return {"decision": "reject", "comment": ""}

        monkeypatch.setattr(node_mod, "interrupt", fake_interrupt)

        state = {
            "session_id": "t-1",
            "pending_tool_call": ToolCall(
                tool_name="user_suspend",
                arguments={"userId": "u1"},
                tier=3,
            ),
        }
        await node_mod.risk_gate(state)
        assert captured["tool_name"] == "user_suspend"
        assert captured["required_keyword"] == "정지"
        assert captured["tier"] == 3


# ============================================================
# 4) user_suspend handler — 3번의 HTTP 호출 + 스냅샷
# ============================================================

class TestUserSuspendHandler:
    @pytest.mark.asyncio
    async def test_suspend_captures_before_and_after(self, monkeypatch):
        """
        정상 실행: GET 1회(before) → PUT 1회(실행) → GET 1회(after).
        AdminApiResult.before_data / after_data 채워져야 한다.
        """
        calls: list[tuple[str, str]] = []

        async def fake_get(path, *, admin_jwt, params=None,
                           invocation_id="", timeout=None):
            calls.append(("GET", path))
            return AdminApiResult(
                ok=True, status_code=200,
                # user_detail 응답을 ApiResponse 래퍼로 가정
                data={
                    "success": True,
                    "data": {"userId": "u1", "status": "ACTIVE"},
                },
            )

        async def fake_write(method, path, *, admin_jwt, json_body=None,
                             invocation_id="", timeout=None):
            calls.append((method, path))
            return AdminApiResult(
                ok=True, status_code=200,
                data={"success": True, "data": "계정이 정지되었습니다."},
            )

        monkeypatch.setattr(
            "monglepick.tools.admin_tools.users_write.get_admin_json", fake_get,
        )
        monkeypatch.setattr(
            "monglepick.tools.admin_tools.users_write.write_admin_json", fake_write,
        )

        ctx = ToolContext(
            admin_jwt="jwt-t", admin_role="SUPER_ADMIN",
            admin_id="admin-1", session_id="sess", invocation_id="inv",
        )
        result = await _handle_user_suspend(
            ctx, userId="u1", reason="비속어 반복", durationDays=7,
        )
        assert result.ok
        # 호출 순서: GET(before) → PUT(실행) → GET(after)
        assert calls == [
            ("GET", "/api/v1/admin/users/u1"),
            ("PUT", "/api/v1/admin/users/u1/suspend"),
            ("GET", "/api/v1/admin/users/u1"),
        ]
        # before / after 에 유저 프로필 dict 주입
        assert result.before_data == {"userId": "u1", "status": "ACTIVE"}
        assert result.after_data == {"userId": "u1", "status": "ACTIVE"}

    @pytest.mark.asyncio
    async def test_suspend_failure_keeps_before_but_skips_after(self, monkeypatch):
        async def fake_get(path, **kwargs):
            return AdminApiResult(
                ok=True, status_code=200,
                data={"success": True, "data": {"userId": "u2", "status": "ACTIVE"}},
            )

        async def fake_write(method, path, **kwargs):
            return AdminApiResult(ok=False, status_code=500, error="boom")

        monkeypatch.setattr(
            "monglepick.tools.admin_tools.users_write.get_admin_json", fake_get,
        )
        monkeypatch.setattr(
            "monglepick.tools.admin_tools.users_write.write_admin_json", fake_write,
        )

        ctx = ToolContext(
            admin_jwt="jwt-t", admin_role="ADMIN",
            admin_id="a", session_id="s", invocation_id="i",
        )
        result = await _handle_user_suspend(ctx, userId="u2")
        assert not result.ok
        # before 는 여전히 감사 가치가 있어 보존, after 는 None (실행 실패라 snapshot 안 찍음)
        assert result.before_data == {"userId": "u2", "status": "ACTIVE"}
        assert result.after_data is None


# ============================================================
# 5) points_manual_adjust handler
# ============================================================

class TestPointsManualAdjustHandler:
    @pytest.mark.asyncio
    async def test_success_uses_backend_response_as_after(self, monkeypatch):
        """Backend 응답 자체가 balanceBefore/After 를 포함 → after_data 에 그대로 저장."""

        async def fake_get(path, **kwargs):
            return AdminApiResult(
                ok=True, status_code=200,
                data={
                    "success": True,
                    "data": {
                        "userId": "u3",
                        "pointBalance": 100,
                        "gradeCode": "BRONZE",
                        "nickname": "tester",
                    },
                },
            )

        async def fake_write(method, path, *, admin_jwt, json_body=None, **kwargs):
            # Backend ManualPointResponse 시뮬레이션 — ApiResponse 래퍼로 래핑된 상태
            return AdminApiResult(
                ok=True, status_code=200,
                data={
                    "success": True,
                    "data": {
                        "userId": "u3",
                        "deltaApplied": json_body["amount"],
                        "balanceBefore": 100,
                        "balanceAfter": 100 + json_body["amount"],
                        "pointType": "bonus",
                        "reason": json_body["reason"],
                        "historyId": 77,
                    },
                },
            )

        monkeypatch.setattr(
            "monglepick.tools.admin_tools.points_write.get_admin_json", fake_get,
        )
        monkeypatch.setattr(
            "monglepick.tools.admin_tools.points_write.write_admin_json", fake_write,
        )

        ctx = ToolContext(
            admin_jwt="jwt-t", admin_role="FINANCE_ADMIN",
            admin_id="fa", session_id="s", invocation_id="i",
        )
        result = await _handle_points_manual_adjust(
            ctx, userId="u3", amount=500, reason="CS 사과",
        )
        assert result.ok
        # before — 축약된 user 잔액 요약
        assert result.before_data == {
            "userId": "u3",
            "pointBalance": 100,
            "gradeCode": "BRONZE",
            "nickname": "tester",
        }
        # after — Backend 응답 그대로
        assert result.after_data["balanceBefore"] == 100
        assert result.after_data["balanceAfter"] == 600
        assert result.after_data["historyId"] == 77


# ============================================================
# 6) tool_executor 가 audit 호출에 before/after + target 전달
# ============================================================

class TestToolExecutorAuditStep6b:
    @pytest.mark.asyncio
    async def test_tier3_audit_receives_snapshots_and_target(self, monkeypatch):
        """Tier 3 실행 시 audit 호출 kwargs 에 before_data/after_data/target_type/target_id."""

        async def fake_handler(ctx, userId, reason="", durationDays=None):
            # Tier 3 tool 이 이미 before/after 를 채운 AdminApiResult 반환
            return AdminApiResult(
                ok=True, status_code=200, data={"msg": "suspended"},
                before_data={"userId": "u9", "status": "ACTIVE"},
                after_data={"userId": "u9", "status": "SUSPENDED"},
            )

        monkeypatch.setattr(
            ADMIN_TOOL_REGISTRY["user_suspend"], "handler", fake_handler,
        )

        audit_spy = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "monglepick.api.admin_audit_client.log_agent_action", audit_spy,
        )

        state = {
            "admin_id": "admin-1",
            "admin_role": "SUPER_ADMIN",
            "admin_jwt": "jwt-t",
            "session_id": f"t-{uuid.uuid4().hex[:6]}",
            "user_message": "user_id=u9 7일 정지",
            "pending_tool_call": ToolCall(
                tool_name="user_suspend",
                arguments={"userId": "u9", "durationDays": 7},
                tier=3,
            ),
        }
        await tool_executor(state)
        audit_spy.assert_called_once()
        kw = audit_spy.call_args.kwargs
        assert kw["tool_name"] == "user_suspend"
        assert kw["ok"] is True
        assert kw["target_type"] == "USER"
        assert kw["target_id"] == "u9"
        assert kw["before_data"] == {"userId": "u9", "status": "ACTIVE"}
        assert kw["after_data"] == {"userId": "u9", "status": "SUSPENDED"}


# ============================================================
# 7) write_admin_json — PUT 도 정상 라우팅
# ============================================================

class TestWriteAdminJsonPut:
    @pytest.mark.asyncio
    async def test_put_method_is_routed_via_client_request(self, monkeypatch):
        captured: dict = {}

        class _FakeResp:
            status_code = 200
            content = b'{"ok":true}'

            def json(self):
                return {"ok": True}

        class _FakeClient:
            async def request(self, method, path, **kwargs):
                captured["method"] = method
                captured["path"] = path
                captured["kwargs"] = kwargs
                return _FakeResp()

        async def _client():
            return _FakeClient()

        monkeypatch.setattr(
            "monglepick.api.admin_backend_client._get_admin_http_client", _client,
        )
        result = await write_admin_json(
            "PUT",
            "/api/v1/admin/users/u1/suspend",
            admin_jwt="j",
            json_body={"reason": "x"},
            invocation_id="i",
        )
        assert result.ok
        assert captured["method"] == "PUT"
        assert captured["path"] == "/api/v1/admin/users/u1/suspend"
        assert captured["kwargs"]["json"] == {"reason": "x"}
        assert captured["kwargs"]["headers"]["Authorization"] == "Bearer j"

    @pytest.mark.asyncio
    async def test_unsupported_method_returns_false(self):
        result = await write_admin_json(
            "GET", "/api/v1/admin/foo", admin_jwt="j",
        )
        assert not result.ok
        assert "unsupported_method" in result.error


# ============================================================
# 8) _infer_audit_target 유닛
# ============================================================

class TestInferAuditTarget:
    def test_user_suspend_picks_userId(self):
        assert _infer_audit_target("user_suspend", {"userId": "u1"}) == ("USER", "u1")

    def test_points_manual_adjust_picks_userId(self):
        assert _infer_audit_target(
            "points_manual_adjust", {"userId": "u2", "amount": 100},
        ) == ("USER", "u2")

    def test_faq_create_target_faq_no_id(self):
        assert _infer_audit_target("faq_create", {"category": "AI"}) == ("FAQ", None)

    def test_unknown_tool_returns_none(self):
        assert _infer_audit_target("xxxxxxx", {"foo": "bar"}) == (None, None)
