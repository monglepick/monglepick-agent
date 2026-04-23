"""
Backend 게스트(비로그인) 쿼터 API 비동기 클라이언트.

비로그인 사용자는 "해당 디바이스에서 평생 1회" 의 무료 AI 추천 체험 기회를 갖는다.
- 체크 (check): 채팅 SSE 진입 초기에 호출해 차단 여부 조회
- 소비 (consume): recommendation_ranker 완료 → movie_card 발행 직전에 호출

Backend 측 Redis 키:
  chat:guest_used:{guest_id}      (쿠키 기반, TTL 365일)
  chat:guest_used_ip:{client_ip}  (IP 기반, TTL 365일)

인증: X-Service-Key 헤더 (settings.SERVICE_API_KEY).
실패 정책: fail-open (Backend/Redis 장애 시 추천은 정상 진행, 로그만 warning).
  - Phase 3 설계서 결정사항: 게스트 체험 UX 저해 방지 + 장애 상황은 로그로 감사.
"""

from __future__ import annotations

import asyncio

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


# ── 응답 모델 ──

class GuestQuotaCheckResult(BaseModel):
    """
    게스트 쿼터 체크 결과.

    - allowed: True 면 진행 가능, False 면 차단 (SSE error 이벤트 GUEST_QUOTA_EXCEEDED)
    - reason: "OK" / "GUEST_COOKIE_USED" / "GUEST_IP_USED" / "FALLBACK_OPEN"
      (FALLBACK_OPEN 은 Backend 장애로 fail-open 된 상태 — 로그 감사용)
    """

    allowed: bool
    reason: str = "OK"


class GuestQuotaConsumeResult(BaseModel):
    """
    게스트 쿼터 소비 결과.

    - success: True 면 최초 소비 성공, False 면 이미 소비됨
    - reason: "OK" / "ALREADY_CONSUMED" / "GUEST_COOKIE_USED" / "GUEST_IP_USED" / "FALLBACK_OPEN"
    """

    success: bool
    reason: str = "OK"


# ── 싱글턴 클라이언트 (point_client 와 동일 패턴) ──

_client = None
_client_lock = asyncio.Lock()


async def _get_http_client():
    """
    httpx.AsyncClient 싱글턴 반환. 앱 수명 동안 커넥션 풀을 재사용한다.

    point_client._get_http_client 와 별도 인스턴스로 유지 — 타임아웃/베이스 URL 이
    같더라도 게스트 쿼터 경로는 쿠키 발급 흐름과 구분되는 책임이므로 격리한다.
    """
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is None:
            import httpx

            from monglepick.config import settings

            _client = httpx.AsyncClient(
                base_url=settings.BACKEND_BASE_URL,
                timeout=httpx.Timeout(10.0, connect=5.0),
                headers={"X-Service-Key": settings.SERVICE_API_KEY},
            )
    return _client


async def close_client():
    """앱 종료 시 httpx 클라이언트를 정리한다."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ── API 함수 ──

async def check_quota(guest_id: str, client_ip: str) -> GuestQuotaCheckResult:
    """
    Backend `POST /api/v1/guest/quota/check` 호출.

    소비하지 않고 쿠키+IP 양쪽 Redis 키를 조회해 차단 여부만 반환한다.

    Args:
        guest_id:  HMAC 검증 완료된 UUID (Agent 측에서 쿠키 파싱 후 전달)
        client_ip: Nginx X-Forwarded-For 첫 항목 또는 request.client.host

    Returns:
        GuestQuotaCheckResult. Backend 장애 시 fail-open (allowed=True, reason=FALLBACK_OPEN).
    """
    if not guest_id or not client_ip:
        # 둘 중 하나라도 비면 쿠키 식별 불가 — 일반적으로 쿠키를 아예 안 보낸 상황.
        # 이 경우 Client 측 Agent 진입 로직이 쿠키를 먼저 발급받게 되어 있으므로
        # 여기 도달하면 설계 버그. 안전하게 차단.
        logger.warning("guest_quota_check_missing_id", guest_id=guest_id, client_ip=client_ip)
        return GuestQuotaCheckResult(allowed=False, reason="MISSING_IDENTITY")

    try:
        client = await _get_http_client()
        resp = await client.post(
            "/api/v1/guest/quota/check",
            json={"guestId": guest_id, "clientIp": client_ip},
        )
        if resp.status_code == 200:
            data = resp.json()
            return GuestQuotaCheckResult(
                allowed=data.get("allowed", True),
                reason=data.get("reason", "OK"),
            )

        # 4xx/5xx: Backend 의 버그/설정 문제 — fail-open 으로 게스트 체험 유지
        logger.warning(
            "guest_quota_check_failed",
            guest_id=guest_id,
            status_code=resp.status_code,
            body=resp.text[:200],
        )
        return GuestQuotaCheckResult(allowed=True, reason="FALLBACK_OPEN")

    except Exception as e:
        # 네트워크/타임아웃 — fail-open (로그만)
        logger.error(
            "guest_quota_check_error",
            guest_id=guest_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return GuestQuotaCheckResult(allowed=True, reason="FALLBACK_OPEN")


async def consume_quota(guest_id: str, client_ip: str) -> GuestQuotaConsumeResult:
    """
    Backend `POST /api/v1/guest/quota/consume` 호출.

    Redis SETNX 로 쿠키 키 + IP 키를 원자적으로 소비한다. 둘 중 하나라도 이미
    존재하면 "이미 소비됨" 으로 간주하되 롤백하지 않는다 (우회 창구 방지).

    호출 시점: agents/chat/graph.py `recommendation_ranker` 완료 직후,
    첫 `movie_card` yield 직전. 로그인 유저의 `point_client.consume_point` 와 동일 위치.

    Args:
        guest_id:  HMAC 검증 완료된 UUID
        client_ip: 실제 클라이언트 IP

    Returns:
        GuestQuotaConsumeResult. Backend 장애 시 fail-open (success=True, reason=FALLBACK_OPEN).
    """
    if not guest_id or not client_ip:
        logger.warning("guest_quota_consume_missing_id", guest_id=guest_id, client_ip=client_ip)
        return GuestQuotaConsumeResult(success=False, reason="MISSING_IDENTITY")

    try:
        client = await _get_http_client()
        resp = await client.post(
            "/api/v1/guest/quota/consume",
            json={"guestId": guest_id, "clientIp": client_ip},
        )
        if resp.status_code == 200:
            data = resp.json()
            return GuestQuotaConsumeResult(
                success=data.get("success", False),
                reason=data.get("reason", "OK"),
            )

        logger.warning(
            "guest_quota_consume_failed",
            guest_id=guest_id,
            status_code=resp.status_code,
            body=resp.text[:200],
        )
        # movie_card 는 이미 생성된 상태 — 차감 실패를 fail-open 으로 덮고 다음 턴에서 재확인
        return GuestQuotaConsumeResult(success=True, reason="FALLBACK_OPEN")

    except Exception as e:
        logger.error(
            "guest_quota_consume_error",
            guest_id=guest_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return GuestQuotaConsumeResult(success=True, reason="FALLBACK_OPEN")


# ── 쿠키 파싱 (HMAC 검증은 Backend 책임) ──

def parse_guest_cookie(raw_cookie_value: str | None) -> str | None:
    """
    쿠키 값 `{guestId}.{signature}` 에서 guestId 만 파싱한다.

    HMAC 서명 검증은 Backend 에서 수행하므로 Agent 는 단순 split 만 한다.
    쿠키 값이 None, 빈 문자열, 구분자 없음이면 None 반환.

    Args:
        raw_cookie_value: Request.cookies.get("mongle_guest") 결과

    Returns:
        guestId 또는 None
    """
    if not raw_cookie_value:
        return None
    parts = raw_cookie_value.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0]


def extract_client_ip(request) -> str:
    """
    FastAPI Request 에서 실제 클라이언트 IP 추출.

    Nginx 프록시 환경에서는 X-Forwarded-For 첫 항목이 원본 IP.
    헤더가 없으면 request.client.host 로 폴백.

    Args:
        request: starlette.requests.Request

    Returns:
        IP 문자열 (빈 값 방지, 최소 "unknown")
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"
