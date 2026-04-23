"""
게스트 쿼터 클라이언트 단위 테스트 (2026-04-22, Phase 3).

monglepick.api.guest_quota_client 의 책임:
- 쿠키 값 파싱 (HMAC 검증은 Backend 담당)
- 클라이언트 IP 추출 (X-Forwarded-For > request.client.host)
- Backend `/api/v1/guest/quota/{check,consume}` 호출 + fail-open 정책

Backend httpx 호출은 mock 처리하여 네트워크 의존성 없이 검증한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from monglepick.api.guest_quota_client import (
    GuestQuotaCheckResult,
    GuestQuotaConsumeResult,
    check_quota,
    consume_quota,
    extract_client_ip,
    parse_guest_cookie,
)


# ============================================================
# parse_guest_cookie
# ============================================================

class TestParseGuestCookie:
    """쿠키 값에서 guestId 파싱 — 서명 검증은 Backend 가 담당하므로 단순 split 만."""

    def test_valid_cookie_returns_guest_id(self):
        """정상 포맷 `{guestId}.{signature}` 에서 guestId 추출."""
        raw = "550e8400-e29b-41d4-a716-446655440000.abcXYZ123signature"
        assert parse_guest_cookie(raw) == "550e8400-e29b-41d4-a716-446655440000"

    def test_none_returns_none(self):
        """None 입력은 None 반환."""
        assert parse_guest_cookie(None) is None

    def test_empty_string_returns_none(self):
        """빈 문자열은 None 반환."""
        assert parse_guest_cookie("") is None

    def test_no_separator_returns_none(self):
        """구분자(.) 없으면 None 반환 — 포맷 불량."""
        assert parse_guest_cookie("no-dot-here") is None

    def test_empty_guest_id_returns_none(self):
        """guestId 부분이 빈 문자열이면 None 반환."""
        assert parse_guest_cookie(".only-signature") is None

    def test_empty_signature_returns_none(self):
        """signature 부분이 빈 문자열이면 None 반환."""
        assert parse_guest_cookie("only-id.") is None

    def test_multiple_dots_split_on_first_only(self):
        """guestId 내부에 예상치 못한 . 가 있어도 첫 번째 구분자로만 분리."""
        # split(".", 1) 이므로 guestId 에 . 이 들어간 경우는 guestId 의 일부로 취급.
        # 실제로 UUID 는 . 을 포함하지 않지만, 방어적 로직 검증.
        assert parse_guest_cookie("uuid.with.dots") == "uuid"


# ============================================================
# extract_client_ip
# ============================================================

class TestExtractClientIp:
    """FastAPI Request 에서 클라이언트 IP 추출 — Nginx XFF 우선."""

    def _make_request(self, xff: str | None = None, client_host: str | None = "127.0.0.1"):
        """Request 객체 최소 mock (headers.get + client.host 만 필요).

        FastAPI Request.headers.get 의 대소문자 무시 동작을 모사하되,
        실제 구현이 'x-forwarded-for' 소문자 키로 조회하므로 그 키만 매칭한다.
        """
        request = MagicMock()
        # MagicMock 의 headers 에 get 메서드를 람다로 주입.
        headers = MagicMock()
        headers.get = lambda key, default=None: (
            xff if key.lower() == "x-forwarded-for" and xff is not None else default
        )
        request.headers = headers
        if client_host is not None:
            client_mock = MagicMock()
            client_mock.host = client_host
            request.client = client_mock
        else:
            request.client = None
        return request

    def test_x_forwarded_for_first_hop_wins(self):
        """체인 IP 는 첫 번째 항목이 원본 클라이언트."""
        request = self._make_request(xff="203.0.113.10, 10.0.0.1, 172.20.0.5")
        assert extract_client_ip(request) == "203.0.113.10"

    def test_x_forwarded_for_single_ip(self):
        """체인이 한 IP 뿐이면 그대로."""
        request = self._make_request(xff="198.51.100.42")
        assert extract_client_ip(request) == "198.51.100.42"

    def test_falls_back_to_client_host(self):
        """XFF 헤더 없으면 request.client.host 사용."""
        request = self._make_request(xff=None, client_host="192.168.1.5")
        assert extract_client_ip(request) == "192.168.1.5"

    def test_empty_xff_falls_back(self):
        """XFF 가 빈 문자열이면 client.host 로 폴백."""
        request = self._make_request(xff="", client_host="192.168.1.5")
        assert extract_client_ip(request) == "192.168.1.5"

    def test_unknown_when_both_absent(self):
        """둘 다 없으면 'unknown' 반환 (빈 문자열 방지)."""
        request = self._make_request(xff=None, client_host=None)
        assert extract_client_ip(request) == "unknown"


# ============================================================
# check_quota — fail-open 정책 검증
# ============================================================

class TestCheckQuota:
    """Backend 장애 시 fail-open 으로 추천 진행 (UX 저해 방지)."""

    @pytest.mark.asyncio
    async def test_missing_identity_blocks(self):
        """guest_id / client_ip 둘 다 비면 즉시 차단 (설계 버그 방어)."""
        result = await check_quota("", "")
        assert result.allowed is False
        assert result.reason == "MISSING_IDENTITY"

    @pytest.mark.asyncio
    async def test_success_returns_backend_payload(self):
        """Backend 200 응답의 allowed/reason 을 그대로 반환."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"allowed": True, "reason": "OK"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "monglepick.api.guest_quota_client._get_http_client",
            AsyncMock(return_value=mock_client),
        ):
            result = await check_quota("guest-1", "1.2.3.4")

        assert result.allowed is True
        assert result.reason == "OK"

    @pytest.mark.asyncio
    async def test_backend_blocks_returns_reason(self):
        """Backend 가 차단을 내리면 allowed=False + reason 그대로 전달."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"allowed": False, "reason": "GUEST_COOKIE_USED"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "monglepick.api.guest_quota_client._get_http_client",
            AsyncMock(return_value=mock_client),
        ):
            result = await check_quota("guest-1", "1.2.3.4")

        assert result.allowed is False
        assert result.reason == "GUEST_COOKIE_USED"

    @pytest.mark.asyncio
    async def test_http_5xx_fails_open(self):
        """Backend 5xx 에러 시 fail-open — allowed=True + FALLBACK_OPEN."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "monglepick.api.guest_quota_client._get_http_client",
            AsyncMock(return_value=mock_client),
        ):
            result = await check_quota("guest-1", "1.2.3.4")

        assert result.allowed is True
        assert result.reason == "FALLBACK_OPEN"

    @pytest.mark.asyncio
    async def test_network_error_fails_open(self):
        """httpx 네트워크 예외도 fail-open 으로 흡수."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch(
            "monglepick.api.guest_quota_client._get_http_client",
            AsyncMock(return_value=mock_client),
        ):
            result = await check_quota("guest-1", "1.2.3.4")

        assert result.allowed is True
        assert result.reason == "FALLBACK_OPEN"


# ============================================================
# consume_quota — fail-open 정책 검증
# ============================================================

class TestConsumeQuota:
    """소비 단계에서도 Backend 장애 시 movie_card 는 유지 (graceful)."""

    @pytest.mark.asyncio
    async def test_missing_identity_blocks(self):
        """식별자 빠지면 즉시 success=False, MISSING_IDENTITY."""
        result = await consume_quota("", "")
        assert result.success is False
        assert result.reason == "MISSING_IDENTITY"

    @pytest.mark.asyncio
    async def test_first_time_consume(self):
        """Backend 200 + success=true → 최초 소비 성공."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "reason": "OK"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "monglepick.api.guest_quota_client._get_http_client",
            AsyncMock(return_value=mock_client),
        ):
            result = await consume_quota("guest-1", "1.2.3.4")

        assert result.success is True
        assert result.reason == "OK"

    @pytest.mark.asyncio
    async def test_already_consumed_returns_reason(self):
        """이미 소비된 경우 success=False + reason 그대로."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": False, "reason": "ALREADY_CONSUMED"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "monglepick.api.guest_quota_client._get_http_client",
            AsyncMock(return_value=mock_client),
        ):
            result = await consume_quota("guest-1", "1.2.3.4")

        assert result.success is False
        assert result.reason == "ALREADY_CONSUMED"

    @pytest.mark.asyncio
    async def test_http_error_fails_open_for_graceful_ux(self):
        """consume 단계 실패는 fail-open (movie_card 이미 발행 준비됨)."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "monglepick.api.guest_quota_client._get_http_client",
            AsyncMock(return_value=mock_client),
        ):
            result = await consume_quota("guest-1", "1.2.3.4")

        assert result.success is True
        assert result.reason == "FALLBACK_OPEN"

    @pytest.mark.asyncio
    async def test_network_error_fails_open(self):
        """네트워크 예외도 graceful."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("Read timeout"))

        with patch(
            "monglepick.api.guest_quota_client._get_http_client",
            AsyncMock(return_value=mock_client),
        ):
            result = await consume_quota("guest-1", "1.2.3.4")

        assert result.success is True
        assert result.reason == "FALLBACK_OPEN"


# ============================================================
# Pydantic 모델
# ============================================================

class TestResultModels:
    """반환 모델의 기본값/타입 검증."""

    def test_check_result_defaults(self):
        r = GuestQuotaCheckResult(allowed=True)
        assert r.reason == "OK"

    def test_consume_result_defaults(self):
        r = GuestQuotaConsumeResult(success=True)
        assert r.reason == "OK"
