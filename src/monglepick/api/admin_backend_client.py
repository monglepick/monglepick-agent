"""
관리자 AI 에이전트 → Backend Admin API 호출 공통 클라이언트.

설계서: docs/관리자_AI에이전트_설계서.md §5.1 (요청-스코프 Ephemeral JWT Forwarding), §7 (감사 로그)

핵심 원칙:
- Backend 는 기존 Admin EP 인가(JWT "ADMIN" role) 를 그대로 사용한다. 에이전트는 별도
  X-Service-Key 를 쓰지 않고 **관리자 JWT 를 forwarding** 해서 Backend 의 책임 추적을 유지한다.
- 모든 호출은 요청-스코프. JWT 는 호출 함수 인자로 받아 헤더에 실어보내고, 클라이언트 인스턴스는
  기본 헤더에 민감 정보를 넣지 않는다 (타 요청에 잔존 방지).
- 응답은 `AdminApiResult` 로 정규화한다 — tool_executor 가 `ref_id` 캐시에 저장하면 LLM 은
  원본 JSON 을 다시 보지 않고 `summary_stub`/`row_count` 같은 축약 정보만 받는다.
- 에러는 전파 대신 `ok=False` 결과로 정규화한다(설계서 §3 에러 전파 금지 원칙).

httpx 클라이언트는 `point_client.py` 와 같은 Backend(BACKEND_BASE_URL) 를 가리키지만 **별도
싱글턴**을 유지한다. 이유: (1) 기본 헤더에 X-Service-Key 를 넣지 않음 / (2) timeout 을
짧게(5s) 운영 — Admin 조회 EP 는 전용 페이지 응답 수준이라 빠르다.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

from monglepick.config import settings

logger = structlog.get_logger(__name__)


# ============================================================
# 응답 DTO (tool 결과 공통 컨테이너)
# ============================================================

class AdminApiResult(BaseModel):
    """
    Admin API 호출 결과 공통 래퍼.

    Tool 함수들은 이 DTO 를 그대로 반환한다. tool_executor 가 ok=True 결과는
    state.tool_results_cache 에 저장하고, LLM 에는 축약본(row_count / summary_stub)만 전달해
    raw rows 재주입을 막는다 (설계서 §6.2).

    Fields:
    - ok:          HTTP 2xx 인지 여부
    - status_code: HTTP 상태 코드 (네트워크 오류 시 0)
    - data:        응답 JSON 원본 (ok=True 일 때만 채워진다)
    - error:       사람이 읽을 수 있는 에러 사유 (ok=False 일 때만 채워진다)
    - latency_ms:  응답까지 걸린 시간 (관찰성 메트릭)
    - row_count:   data 가 list 형이면 길이, dict 면 None (narration 프롬프트 힌트용)
    - before_data: (Step 6b) Tier 3 handler 가 실행 직전 리소스 GET 해서 담는 스냅샷.
                   tool_executor 가 감사 로그 `beforeData` 로 전달. Tier 0/1/2 에선 미사용.
    - after_data:  (Step 6b) 실행 후 재 GET 스냅샷. 변경 전/후 비교 가능.
    """

    ok: bool = False
    status_code: int = 0
    data: Any = None
    error: str = ""
    latency_ms: int = 0
    row_count: int | None = None
    before_data: Any = None
    after_data: Any = None


# ============================================================
# httpx 싱글턴 (point_client 와 분리)
# ============================================================

_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_admin_http_client() -> httpx.AsyncClient:
    """
    Backend 호출용 httpx.AsyncClient 싱글턴을 반환한다.

    기본 헤더에 X-Service-Key 나 특정 JWT 를 넣지 않는다 — 호출 시마다 admin_jwt 를
    per-call 헤더로 주입해 잔존을 방지한다.

    Timeout: connect 3s / read 5s. Admin 조회 EP 는 전용 화면 응답 정도라 충분.
    """
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is None:
            _client = httpx.AsyncClient(
                base_url=settings.BACKEND_BASE_URL,
                timeout=httpx.Timeout(5.0, connect=3.0),
                # 기본 헤더 비움 — admin_jwt / invocation_id 는 호출별로 별도 주입
                headers={},
            )
    return _client


async def close_admin_client() -> None:
    """앱 종료 시 httpx 클라이언트 정리 (main.py lifespan shutdown 에서 호출)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ============================================================
# 핵심 호출 함수
# ============================================================

async def get_admin_json(
    path: str,
    *,
    admin_jwt: str,
    params: dict[str, Any] | None = None,
    invocation_id: str = "",
    timeout: float | None = None,
) -> AdminApiResult:
    """
    Backend Admin GET EP 를 JWT forwarding 으로 호출한다.

    Args:
        path: `/api/v1/admin/...` 형태의 절대 경로 (BACKEND_BASE_URL 기준 상대)
        admin_jwt: 관리자 JWT 원문 — `Authorization: Bearer {admin_jwt}` 로 그대로 전달
        params: 쿼리스트링 dict
        invocation_id: 추적용 식별자. `X-Agent-Invocation-Id` 헤더로 전달되며 Backend 가
                       감사 로그 description 에 포함해 "어느 에이전트 턴이 촉발했는지" 를
                       양방향으로 추적할 수 있게 한다 (설계서 §5.1).
        timeout: 기본값 5초 override (보고서 생성 등 무거운 GET 에서 활용)

    Returns:
        AdminApiResult — ok/status/data/error/latency_ms/row_count 정규화.
        네트워크/타임아웃/JSON 파싱 에러 등은 전부 ok=False 로 집약한다.
    """
    started = time.perf_counter()
    headers: dict[str, str] = {}
    if admin_jwt:
        headers["Authorization"] = f"Bearer {admin_jwt}"
    else:
        # JWT_SECRET 미설정 개발 환경 호환 — ServiceKey 로 fallback 해 Backend 의
        # SecurityConfig authenticated() 통과를 시도한다 (운영에서는 admin_jwt 강제).
        if settings.SERVICE_API_KEY:
            headers["X-Service-Key"] = settings.SERVICE_API_KEY
    if invocation_id:
        headers["X-Agent-Invocation-Id"] = invocation_id

    try:
        client = await _get_admin_http_client()
        resp = await client.get(
            path,
            params=params,
            headers=headers,
            timeout=timeout if timeout is not None else 5.0,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if 200 <= resp.status_code < 300:
            try:
                data = resp.json()
            except Exception as je:
                logger.warning(
                    "admin_api_json_decode_failed",
                    path=path,
                    status=resp.status_code,
                    error=str(je),
                )
                return AdminApiResult(
                    ok=False,
                    status_code=resp.status_code,
                    error=f"json_decode_error:{type(je).__name__}",
                    latency_ms=elapsed_ms,
                )
            row_count = len(data) if isinstance(data, list) else None
            logger.info(
                "admin_api_ok",
                path=path,
                status=resp.status_code,
                latency_ms=elapsed_ms,
                row_count=row_count,
            )
            return AdminApiResult(
                ok=True,
                status_code=resp.status_code,
                data=data,
                latency_ms=elapsed_ms,
                row_count=row_count,
            )

        # 2xx 외 → Backend 에러. body 의 detail/message 가 있으면 error 필드에 넣는다.
        detail = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("message") or ""
        except Exception:
            pass

        logger.warning(
            "admin_api_non_2xx",
            path=path,
            status=resp.status_code,
            detail=detail[:200],
            latency_ms=elapsed_ms,
        )
        return AdminApiResult(
            ok=False,
            status_code=resp.status_code,
            error=detail or f"http_{resp.status_code}",
            latency_ms=elapsed_ms,
        )

    except httpx.TimeoutException as te:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("admin_api_timeout", path=path, elapsed_ms=elapsed_ms)
        return AdminApiResult(
            ok=False,
            status_code=0,
            error=f"timeout:{type(te).__name__}",
            latency_ms=elapsed_ms,
        )
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "admin_api_unexpected_error",
            path=path,
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=elapsed_ms,
        )
        return AdminApiResult(
            ok=False,
            status_code=0,
            error=f"unexpected:{type(e).__name__}",
            latency_ms=elapsed_ms,
        )


# ============================================================
# 쓰기 (POST/PUT/PATCH/DELETE) — Tier 2/3 tool 에서 HITL 승인 통과 후 호출
# ============================================================

async def write_admin_json(
    method: str,
    path: str,
    *,
    admin_jwt: str,
    json_body: dict[str, Any] | None = None,
    invocation_id: str = "",
    timeout: float | None = None,
) -> AdminApiResult:
    """
    Backend Admin 쓰기 EP 를 JWT forwarding 으로 호출한다 (POST/PUT/PATCH/DELETE).

    Step 5a(HITL) 부터 Tier 2 쓰기 tool 이 이 함수를 호출한다. `risk_gate` 에서 사용자 승인
    이 떨어진 뒤에만 tool_executor → handler → write_admin_json 경로로 흘러온다.

    Step 6b 에서 PUT 도 지원하기 위해 `post_admin_json` 을 이 함수로 일반화했다. POST 전용
    레거시 호출부는 아래 `post_admin_json` wrapper 로 하위 호환 유지.

    Args:
        method: HTTP 메서드 — "POST"/"PUT"/"PATCH"/"DELETE". 대소문자 무관.
        path:   `/api/v1/admin/...` 상대 경로.
        admin_jwt, json_body, invocation_id, timeout: 공통 옵션.

    Returns:
        AdminApiResult. 네트워크·HTTP·JSON 에러 전부 ok=False 로 수렴(§3).
    """
    verb = method.upper()
    if verb not in ("POST", "PUT", "PATCH", "DELETE"):
        return AdminApiResult(
            ok=False, status_code=0, error=f"unsupported_method:{verb}",
        )

    started = time.perf_counter()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if admin_jwt:
        headers["Authorization"] = f"Bearer {admin_jwt}"
    else:
        if settings.SERVICE_API_KEY:
            headers["X-Service-Key"] = settings.SERVICE_API_KEY
    if invocation_id:
        headers["X-Agent-Invocation-Id"] = invocation_id

    try:
        client = await _get_admin_http_client()
        kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": timeout if timeout is not None else 8.0,
        }
        # POST/PUT/PATCH 는 body 전송, DELETE 는 body 없이
        if verb in ("POST", "PUT", "PATCH"):
            kwargs["json"] = json_body or {}
        resp = await client.request(verb, path, **kwargs)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if 200 <= resp.status_code < 300:
            try:
                # 204 No Content 같은 경우 body 가 비어있을 수 있음
                data = resp.json() if resp.content else None
            except Exception as je:
                logger.warning(
                    "admin_api_write_json_decode_failed",
                    method=verb, path=path, status=resp.status_code, error=str(je),
                )
                return AdminApiResult(
                    ok=False, status_code=resp.status_code,
                    error=f"json_decode_error:{type(je).__name__}",
                    latency_ms=elapsed_ms,
                )
            logger.info(
                "admin_api_write_ok",
                method=verb, path=path, status=resp.status_code, latency_ms=elapsed_ms,
            )
            return AdminApiResult(
                ok=True, status_code=resp.status_code,
                data=data, latency_ms=elapsed_ms,
            )

        # 2xx 외 → 에러 바디 detail/message 추출
        detail = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body.get("detail") or body.get("message") or ""
        except Exception:
            pass
        logger.warning(
            "admin_api_write_non_2xx",
            method=verb, path=path, status=resp.status_code,
            detail=detail[:200], latency_ms=elapsed_ms,
        )
        return AdminApiResult(
            ok=False, status_code=resp.status_code,
            error=detail or f"http_{resp.status_code}",
            latency_ms=elapsed_ms,
        )

    except httpx.TimeoutException as te:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "admin_api_write_timeout", method=verb, path=path, elapsed_ms=elapsed_ms,
        )
        return AdminApiResult(
            ok=False, status_code=0,
            error=f"timeout:{type(te).__name__}", latency_ms=elapsed_ms,
        )
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "admin_api_write_unexpected_error",
            method=verb, path=path, error=str(e), error_type=type(e).__name__,
            elapsed_ms=elapsed_ms,
        )
        return AdminApiResult(
            ok=False, status_code=0,
            error=f"unexpected:{type(e).__name__}", latency_ms=elapsed_ms,
        )


async def post_admin_json(
    path: str,
    *,
    admin_jwt: str,
    json_body: dict[str, Any] | None = None,
    invocation_id: str = "",
    timeout: float | None = None,
) -> AdminApiResult:
    """
    POST 호환 래퍼 — Step 5a `faq_create`/`banner_create` 가 이 이름으로 호출.
    내부는 `write_admin_json("POST", ...)` 에 위임한다.
    """
    return await write_admin_json(
        "POST", path,
        admin_jwt=admin_jwt, json_body=json_body,
        invocation_id=invocation_id, timeout=timeout,
    )


# ============================================================
# Backend ApiResponse 래퍼 언래핑
# ============================================================

def unwrap_api_response(result: AdminApiResult) -> AdminApiResult:
    """
    Backend 의 `{success, data, error}` 래퍼에서 내부 `data` 만 꺼낸다.

    Step 2 에서 `tools/admin_tools/stats.py` 에만 있던 `_unwrap_api_response` 를
    admin_backend_client 로 승격해 전 도메인 tool (users_read/content_read 등) 이 공유.

    - ok=False 또는 data 가 dict 가 아니면 그대로 반환 (graceful).
    - 래퍼 시그니처 (`data` 키 + `success` 또는 `error` 키 동반) 를 감지한 경우에만 언래핑.
    - 언래핑 후 data 가 list 면 row_count 를 재계산. Page 응답(`{content, totalElements, ...}`)
      은 dict 로 유지하되 `row_count = len(content)` 로 세팅해 narrator 가 "총 N건 중 표시 M건"
      같은 서술을 만들 수 있도록 돕는다.
    """
    if not result.ok or not isinstance(result.data, dict):
        return result
    body = result.data
    if "data" in body and ("success" in body or "error" in body):
        inner = body.get("data")
        new_row_count: int | None
        if isinstance(inner, list):
            new_row_count = len(inner)
        elif isinstance(inner, dict) and isinstance(inner.get("content"), list):
            # Spring Data Page 응답 — content 배열의 길이를 row_count 로.
            new_row_count = len(inner["content"])
        else:
            new_row_count = None
        return AdminApiResult(
            ok=result.ok,
            status_code=result.status_code,
            data=inner,
            error=result.error,
            latency_ms=result.latency_ms,
            row_count=new_row_count,
        )
    return result


# ============================================================
# 요약 유틸 — LLM 컨텍스트 삽입용 축약본
# ============================================================

def summarize_for_llm(
    result: AdminApiResult,
    *,
    sample_rows: int = 3,
    max_str_len: int = 400,
) -> dict[str, Any]:
    """
    AdminApiResult 를 LLM narrator 프롬프트에 넣기 좋은 축약 dict 로 변환한다.

    원칙: **raw 리스트 본문을 그대로 프롬프트에 넣지 않는다**.
    - list 응답: 총 건수 + 선두 `sample_rows` 행만 포함. 나머지는 `... 외 N건 생략` 표기.
    - dict 응답: 최상위 키/값을 그대로 유지하되, 값이 긴 문자열이면 `max_str_len` 로 자른다.
    - ok=False: error/status_code 만 얇게 남긴다.

    이 결과는 narrator 의 Solar 프롬프트 context 에 포함되어 수치·문장을 "만들지 않고
    인용"만 할 수 있도록 돕는다 (설계서 §6.1).
    """
    if not result.ok:
        return {
            "ok": False,
            "status_code": result.status_code,
            "error": result.error,
        }

    data = result.data
    if isinstance(data, list):
        total = len(data)
        head = data[:sample_rows]
        return {
            "ok": True,
            "status_code": result.status_code,
            "total_rows": total,
            "sample_rows": head,
            "truncated": total > sample_rows,
            "latency_ms": result.latency_ms,
        }

    if isinstance(data, dict):
        shrunk: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > max_str_len:
                shrunk[k] = v[:max_str_len] + "...(생략)"
            elif isinstance(v, list) and len(v) > sample_rows:
                shrunk[k] = {
                    "__list_total__": len(v),
                    "sample": v[:sample_rows],
                    "truncated": True,
                }
            else:
                shrunk[k] = v
        return {
            "ok": True,
            "status_code": result.status_code,
            "data": shrunk,
            "latency_ms": result.latency_ms,
        }

    # 스칼라/기타
    return {
        "ok": True,
        "status_code": result.status_code,
        "data": data,
        "latency_ms": result.latency_ms,
    }


# ============================================================
# (참고) 후속 Step — POST/PUT/DELETE 래퍼는 Tier 2/3 도입 시 추가
# ============================================================
# Step 2 는 읽기만 지원하므로 `get_admin_json` 만 공개한다.
# 후속 Step 에서 `write_admin_json(method, path, json, admin_jwt, invocation_id)` 가
# HITL 승인 플로우와 함께 추가될 예정 — 쓰기는 반드시 risk_gate 통과 후에만 호출.
