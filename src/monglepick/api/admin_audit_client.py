"""
관리자 AI 에이전트 → Backend 감사 로그 기록 클라이언트 (Step 6a).

설계서: docs/관리자_AI에이전트_설계서.md §7 감사 로그

역할:
- Agent 의 `tool_executor` 가 Tier≥2 쓰기 tool 을 실행한 직후 Backend 의
  `POST /api/v1/admin/audit-logs/agent` 에 callback 하여 감사 기록을 남긴다.
- actor 식별은 Backend 의 `AdminAuditService.resolveCurrentActor()` 가 SecurityContext 에서
  자동 추출하므로, Agent 는 **관리자 JWT 를 그대로 forwarding** 해야 한다(§5.1).
- 실패는 graceful. 감사 기록 호출 실패가 원 작업 응답을 깨뜨리지 않는다 — Backend 측
  `AdminAuditService.log` 도 REQUIRES_NEW 로 격리되어 있어 이중 방어.

호출 지점(tool_executor 내부):
- Tier 0/1: 감사 로그 남기지 않음(볼륨 폭증 방지, §7.2 Tier 0/1 읽기 정책).
- Tier 2/3 : 실행 성공/실패 여부와 무관하게 한 건 기록 (`ok` 필드로 결과 구분).
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import structlog

from monglepick.api.admin_backend_client import _get_admin_http_client
from monglepick.config import settings

logger = structlog.get_logger(__name__)


# 감사 로그 기록 실패가 원 작업에 영향을 주지 않도록 timeout 을 짧게(3s) 유지한다.
_AUDIT_TIMEOUT_SEC: float = 3.0


async def log_agent_action(
    *,
    admin_jwt: str,
    tool_name: str,
    arguments: dict[str, Any],
    ok: bool,
    user_prompt: str = "",
    target_type: str | None = None,
    target_id: str | None = None,
    before_data: dict[str, Any] | str | None = None,
    after_data: dict[str, Any] | str | None = None,
    error: str = "",
    invocation_id: str = "",
) -> bool:
    """
    Tier 2/3 tool 실행 1건을 Backend 감사 로그에 기록한다.

    Args:
        admin_jwt: 관리자 JWT (SecurityContext actor 식별에 필수). 없으면 호출 자체를
            조기 스킵한다 (개발 환경에서 JWT 없이 동작하는 경우를 그레이스풀 처리).
        tool_name: 실행된 tool 이름 (예: "faq_create").
        arguments: tool 에 전달된 인자 dict. description 프롬프트에 JSON 으로 요약된다.
        ok: tool 실행이 성공했는지 여부. 실패도 기록하되 description 에 표기.
        user_prompt: 원래 관리자 발화. description 앞쪽에 포함되어 "어떤 요청이 이 tool 을
            유발했는지" 추적을 쉽게 한다. 길면 200자로 자른다.
        target_type / target_id: 영향받은 리소스 (nullable). Tier 3 쓰기는 가능하면 채운다.
        before_data / after_data: 변경 전/후 스냅샷. dict 면 JSON 으로 직렬화해 전송.
        error: ok=False 일 때 사유 요약.
        invocation_id: X-Agent-Invocation-Id 헤더 — Backend 감사 로그와 Agent 턴을
            양방향 추적하기 위한 식별자.

    Returns:
        True 면 Backend 가 201 로 수신 완료. 그 외(네트워크 실패/4xx/5xx) 는 False —
        단, 예외를 전파하지 않고 로그만 남긴다(원 작업 응답 보호).
    """
    # JWT 가 없으면 Backend SecurityContext 가 actor 를 식별할 수 없으므로 조기 스킵.
    # (운영에서는 JWT forwarding 이 보장되지만 dev 모드에서는 SERVICE_API_KEY 로도
    # Backend `/api/v1/admin/**` 를 통과할 수 있어 단순 스킵이 가장 덜 위험하다.)
    if not admin_jwt:
        logger.info(
            "admin_audit_skipped_no_jwt",
            tool_name=tool_name,
            reason="admin_jwt unavailable — dev mode likely",
        )
        return False

    description = _format_description(
        tool_name=tool_name,
        arguments=arguments,
        ok=ok,
        user_prompt=user_prompt,
        error=error,
    )

    payload: dict[str, Any] = {
        "actionType": "AGENT_EXECUTED",
        "description": description,
    }
    if target_type:
        payload["targetType"] = target_type
    if target_id:
        payload["targetId"] = target_id
    if before_data is not None:
        payload["beforeData"] = _to_json_str(before_data)
    if after_data is not None:
        payload["afterData"] = _to_json_str(after_data)

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {admin_jwt}",
    }
    if invocation_id:
        headers["X-Agent-Invocation-Id"] = invocation_id

    started = time.perf_counter()
    try:
        client = await _get_admin_http_client()
        resp = await client.post(
            "/api/v1/admin/audit-logs/agent",
            json=payload,
            headers=headers,
            timeout=_AUDIT_TIMEOUT_SEC,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if 200 <= resp.status_code < 300:
            logger.info(
                "admin_audit_ok",
                tool_name=tool_name,
                status=resp.status_code,
                latency_ms=elapsed_ms,
                target_type=target_type,
                target_id=target_id,
            )
            return True
        detail = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("message") or ""
        except Exception:
            pass
        logger.warning(
            "admin_audit_non_2xx",
            tool_name=tool_name,
            status=resp.status_code,
            detail=detail[:200],
            latency_ms=elapsed_ms,
        )
        return False
    except httpx.TimeoutException:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "admin_audit_timeout", tool_name=tool_name, elapsed_ms=elapsed_ms,
        )
        return False
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "admin_audit_unexpected_error",
            tool_name=tool_name, error=str(e), error_type=type(e).__name__,
            elapsed_ms=elapsed_ms,
        )
        return False


# ============================================================
# 내부 유틸
# ============================================================

def _format_description(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    ok: bool,
    user_prompt: str,
    error: str,
) -> str:
    """
    사람이 읽을 수 있는 description 문자열 생성.

    형식: `[tool=faq_create][ok|fail] 관리자 프롬프트: '...' / 인자: {...}`
    - 길면 Backend 쪽 2000자 상한에서 자르므로 Agent 에서는 user_prompt 만 200자로 자른다.
    """
    ok_tag = "ok" if ok else "fail"
    prompt_preview = (user_prompt or "").strip()
    if len(prompt_preview) > 200:
        prompt_preview = prompt_preview[:200] + "..."

    try:
        args_json = json.dumps(arguments or {}, ensure_ascii=False)
    except Exception:
        args_json = str(arguments)

    parts = [f"[tool={tool_name}]", f"[{ok_tag}]"]
    if prompt_preview:
        parts.append(f"관리자 프롬프트: '{prompt_preview}'")
    parts.append(f"인자: {args_json}")
    if not ok and error:
        parts.append(f"에러: {error}")
    return " / ".join(parts)


def _to_json_str(value: dict[str, Any] | str) -> str:
    """dict 면 JSON 직렬화, 문자열이면 그대로 반환. 실패 시 repr fallback."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return repr(value)
