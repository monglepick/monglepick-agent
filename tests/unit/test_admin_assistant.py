"""
관리자 AI 에이전트 단위 테스트 (Step 1).

테스트 범위:
1. normalize_admin_role — 8종 + 별칭 + 비관리자
2. classify_admin_intent — 6종 kind 분기 + confidence 보정 + 에러 fallback
3. run_admin_assistant_sync — 그래프 E2E 분기
   - smalltalk → smalltalk_responder 실행
   - query/action/stats/report/sql → placeholder 메시지
   - admin_role="" → 진입 차단 메시지
"""

from __future__ import annotations

import pytest

from monglepick.agents.admin_assistant.graph import run_admin_assistant_sync
from monglepick.agents.admin_assistant.models import (
    AdminIntent,
    normalize_admin_role,
)
from monglepick.chains.admin_intent_chain import classify_admin_intent


# ============================================================
# normalize_admin_role
# ============================================================

class TestNormalizeAdminRole:
    """AdminRoleEnum 8종 + 별칭 정규화."""

    def test_super_admin_passes_through(self):
        assert normalize_admin_role("SUPER_ADMIN") == "SUPER_ADMIN"

    def test_plain_admin_aliased_to_super(self):
        """Backend 가 단순 'ADMIN' 을 JWT role 로 넣는 현 상황 대응."""
        assert normalize_admin_role("ADMIN") == "SUPER_ADMIN"

    def test_lowercase_and_whitespace_normalized(self):
        assert normalize_admin_role(" moderator ") == "MODERATOR"
        assert normalize_admin_role("finance_admin") == "FINANCE_ADMIN"

    def test_all_eight_roles_accepted(self):
        for role in [
            "SUPER_ADMIN", "ADMIN", "MODERATOR", "FINANCE_ADMIN",
            "SUPPORT_ADMIN", "DATA_ADMIN", "AI_OPS_ADMIN", "STATS_ADMIN",
        ]:
            assert normalize_admin_role(role) != ""

    def test_user_role_rejected(self):
        """일반 유저 역할은 빈 문자열로 차단."""
        assert normalize_admin_role("USER") == ""

    def test_empty_and_none_return_blank(self):
        assert normalize_admin_role("") == ""
        assert normalize_admin_role(None) == ""

    def test_unknown_role_rejected(self):
        assert normalize_admin_role("HACKER") == ""


# ============================================================
# classify_admin_intent
# ============================================================

class TestClassifyAdminIntent:
    """6종 intent 분류 + confidence 보정 + 에러 fallback."""

    @pytest.mark.asyncio
    async def test_smalltalk_passthrough(self, mock_ollama):
        mock_ollama.set_structured_response(
            AdminIntent(kind="smalltalk", confidence=0.9, reason="인사말"),
        )
        result = await classify_admin_intent(
            user_message="안녕", admin_role="SUPER_ADMIN",
        )
        assert result.kind == "smalltalk"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_stats_kind_high_confidence(self, mock_ollama):
        mock_ollama.set_structured_response(
            AdminIntent(kind="stats", confidence=0.92, reason="DAU 키워드"),
        )
        result = await classify_admin_intent(
            user_message="지난 7일 DAU 추이", admin_role="STATS_ADMIN",
        )
        assert result.kind == "stats"

    @pytest.mark.asyncio
    async def test_action_kind(self, mock_ollama):
        mock_ollama.set_structured_response(
            AdminIntent(kind="action", confidence=0.88, reason="배너 등록"),
        )
        result = await classify_admin_intent(
            user_message="겨울 프로모 배너 등록해줘", admin_role="ADMIN",
        )
        assert result.kind == "action"

    @pytest.mark.asyncio
    async def test_sql_kind(self, mock_ollama):
        """sql 은 분류는 성공하지만 downstream 에서 placeholder 로 거절된다."""
        mock_ollama.set_structured_response(
            AdminIntent(kind="sql", confidence=0.85, reason="JOIN 필요"),
        )
        result = await classify_admin_intent(
            user_message="복잡한 JOIN 쿼리", admin_role="SUPER_ADMIN",
        )
        assert result.kind == "sql"

    @pytest.mark.asyncio
    async def test_low_confidence_corrected_to_smalltalk(self, mock_ollama):
        """confidence < 0.5 → smalltalk 로 강제 보정."""
        mock_ollama.set_structured_response(
            AdminIntent(kind="stats", confidence=0.3, reason="애매함"),
        )
        result = await classify_admin_intent(
            user_message="뭐가 많아?", admin_role="ADMIN",
        )
        assert result.kind == "smalltalk"
        # 원래 분류 기록은 reason 에 남아야 함
        assert "low_confidence_fallback" in result.reason

    @pytest.mark.asyncio
    async def test_error_returns_smalltalk_fallback(self, mock_ollama):
        """LLM 예외 발생 시 smalltalk + confidence=0 으로 graceful fallback."""
        mock_ollama.set_error(RuntimeError("Solar API down"))
        result = await classify_admin_intent(
            user_message="뭐라도 해줘", admin_role="ADMIN",
        )
        assert result.kind == "smalltalk"
        assert result.confidence == 0.0
        assert "classify_error" in result.reason


# ============================================================
# 그래프 E2E (run_admin_assistant_sync)
# ============================================================

class TestAdminAssistantGraph:
    """Step 1 그래프 분기 E2E."""

    @pytest.mark.asyncio
    async def test_non_admin_blocked(self, mock_ollama):
        """
        admin_role="" (비관리자) 로 진입하면 LLM 호출 없이 차단 메시지.

        intent_classifier 는 admin_role 이 비어있으면 LLM 을 태우지 않고
        smalltalk intent 로 고정한다. response_formatter 가 _NOT_ADMIN_MESSAGE 를 반환.
        """
        # LLM 호출 자체가 없어야 하지만 혹시 발생해도 테스트가 죽지 않도록 기본값 제공
        mock_ollama.set_structured_response(
            AdminIntent(kind="smalltalk", confidence=0.9),
        )
        state = await run_admin_assistant_sync(
            admin_id="someone",
            admin_role="",  # 비관리자
            admin_jwt="",
            session_id="",
            user_message="너 뭐 할 수 있어?",
        )
        assert state.get("admin_role") == ""
        assert "관리자 권한" in state.get("response_text", "")

    @pytest.mark.asyncio
    async def test_smalltalk_flow_generates_response(self, mock_ollama):
        """smalltalk → smalltalk_responder → response_formatter 경로 실행."""
        # Intent 분류용 구조화 응답 + smalltalk 자유 텍스트 응답을 모두 설정
        mock_ollama.set_structured_response(
            AdminIntent(kind="smalltalk", confidence=0.9, reason="인사"),
        )
        mock_ollama.set_response("안녕하세요! 관리자 어시스턴트예요. 뭐 도와드릴까요?")

        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="SUPER_ADMIN",
            admin_jwt="",
            session_id="",
            user_message="안녕",
        )
        assert state.get("admin_role") == "SUPER_ADMIN"
        assert isinstance(state.get("intent"), AdminIntent)
        assert state["intent"].kind == "smalltalk"
        response = state.get("response_text", "")
        assert response.startswith("안녕하세요")

    @pytest.mark.asyncio
    async def test_action_intent_goes_through_tool_selector_path(self, mock_ollama):
        """
        action intent — route_after_intent 에 의해 tool_selector 로 분기된다 (Step 5a+).
        mock 환경에선 bind_tools 가 MagicMock 이라 selector 가 None → smart_fallback_responder
        가 Solar fallback 응답을 생성. mock_ollama 자유 텍스트가 그대로 response_text 에 담김.
        """
        mock_ollama.set_structured_response(
            AdminIntent(kind="action", confidence=0.95, reason="배너 등록"),
        )
        mock_ollama.set_response("이렇게 바꿔 말씀해 주실래요?")
        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="SUPER_ADMIN",
            admin_jwt="",
            session_id="",
            user_message="겨울 프로모 배너 등록해줘",
        )
        response = state.get("response_text", "")
        # tool 실행 없이 smart_fallback 가 돌린 LLM 응답이 나와야 한다
        assert len(response) > 0
        assert state.get("pending_tool_call") is None
        assert len(state.get("session_id") or "") >= 10

    @pytest.mark.asyncio
    async def test_query_intent_falls_back_to_smart_fallback(self, mock_ollama):
        """
        Step 6c(2026-04-23): query tool 매칭 실패 → smart_fallback_responder 가 Solar
        fallback 응답 생성 (기존의 고정 placeholder 대신 LLM 역제안).
        """
        mock_ollama.set_structured_response(
            AdminIntent(kind="query", confidence=0.95, reason="유저 조회"),
        )
        mock_ollama.set_response("이 질문은 사용자 목록 조회로 바꿔 말씀해 주실래요?")
        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="SUPER_ADMIN",
            admin_jwt="",
            session_id="",
            user_message="user_id=abc 결제 내역 보여줘",
        )
        response = state.get("response_text", "")
        assert len(response) > 0
        assert state.get("pending_tool_call") is None
        assert len(state.get("session_id") or "") >= 10

    @pytest.mark.asyncio
    async def test_stats_intent_falls_back_to_smart_fallback(self, mock_ollama):
        """
        Step 6c(2026-04-23): stats tool 매칭 실패 → smart_fallback_responder 경로.
        """
        mock_ollama.set_structured_response(
            AdminIntent(kind="stats", confidence=0.95, reason="DAU"),
        )
        mock_ollama.set_response("DAU 관련은 '지난 7일 DAU 추이' 로 물어봐 주세요.")
        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="STATS_ADMIN",
            admin_jwt="",
            session_id="",
            user_message="지난 7일 DAU 알려줘",
        )
        response = state.get("response_text", "")
        assert len(response) > 0
        assert state.get("pending_tool_call") is None
        assert len(state.get("session_id") or "") >= 10

    @pytest.mark.asyncio
    async def test_sql_intent_returns_unsupported_notice(self, mock_ollama):
        """sql 은 Phase 5 이후 예정 — 미지원 안내."""
        mock_ollama.set_structured_response(
            AdminIntent(kind="sql", confidence=0.9, reason="자유 쿼리"),
        )
        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="SUPER_ADMIN",
            admin_jwt="",
            session_id="",
            user_message="자유 SELECT 날려줘",
        )
        response = state.get("response_text", "")
        assert "지원하지 않" in response

    @pytest.mark.asyncio
    async def test_session_id_autogenerated_when_empty(self, mock_ollama):
        mock_ollama.set_structured_response(
            AdminIntent(kind="smalltalk", confidence=0.9),
        )
        mock_ollama.set_response("ok")
        state = await run_admin_assistant_sync(
            admin_id="admin-001",
            admin_role="ADMIN",  # 별칭 경로 — context_loader 에서 SUPER_ADMIN 으로 정규화됨
            admin_jwt="",
            session_id="",
            user_message="안녕",
        )
        sid = state.get("session_id", "")
        assert sid  # 빈 문자열 아님
        assert len(sid) >= 10  # uuid4 형태
        # "ADMIN" 별칭이 "SUPER_ADMIN" 으로 정규화되어 state 에 들어갔는지
        assert state.get("admin_role") == "SUPER_ADMIN"
