"""
Upstage Solar API 호출 사용량/비용 추적.

설계:
- LangChain `AsyncCallbackHandler` 를 ChatOpenAI(=Solar API 호환) 인스턴스에 부착해
  매 호출의 token usage 를 수집한다.
- 호출 직후 비용을 산정해 Backend `POST /api/v1/admin/stats/solar-usage/log` 에
  fire-and-forget 으로 적재한다 (X-Service-Key 인증).
- 호출 주체(agent_name) 는 `contextvars.ContextVar` 로 전달한다 — FastAPI
  미들웨어가 URL 경로에서 추출해 set 하면, 같은 asyncio Task 흐름 안에서
  발생하는 모든 LLM 호출이 자동 attribute 된다.

운영 메모:
- 적재 실패는 사용자 흐름에 영향을 주지 않는다 (fire-and-forget). 단, 실패 카운터는
  로깅된다 — Prometheus 메트릭은 추후 추가 가능.
- 콜백이 LLM 응답에서 usage 를 추출하지 못한 경우(LangChain 버전·모델 응답 형식 차이)
  로그만 남기고 적재는 스킵한다.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from datetime import datetime
from typing import Any
from uuid import UUID

import httpx
import structlog
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from monglepick.config import settings
from monglepick.llm.solar_pricing import estimate_cost_usd

logger = structlog.get_logger(__name__)


# ============================================================
# Agent 식별 — ContextVar
# ============================================================
#
# FastAPI 미들웨어가 요청 경로 → agent_name 으로 매핑해 set 한다.
# 미들웨어가 부착되지 않은 경로(스크립트, 단위 테스트 등)는 "unknown" 으로 적재된다.
_current_agent_name: ContextVar[str] = ContextVar(
    "monglepick_solar_current_agent",
    default="unknown",
)


def set_current_agent(name: str) -> None:
    """현재 실행 컨텍스트의 agent_name 을 세팅한다 (FastAPI 미들웨어/스크립트에서 호출)."""
    _current_agent_name.set(name)


def get_current_agent() -> str:
    """현재 실행 컨텍스트의 agent_name 을 반환한다 — 콜백에서 사용."""
    return _current_agent_name.get()


# ============================================================
# httpx 싱글턴 — Backend 적재 전용
# ============================================================
#
# 다른 backend client 와 분리한 이유:
# (1) X-Service-Key 가 기본 헤더에 포함됨. (2) 짧은 timeout — 적재 실패해도 사용자
# 흐름을 절대 막지 않기 위함. (3) lifespan 종료 시 정리.
_log_client: httpx.AsyncClient | None = None
_log_client_lock = asyncio.Lock()


async def _get_log_http_client() -> httpx.AsyncClient:
    """Backend 적재 전용 싱글턴 클라이언트."""
    global _log_client
    if _log_client is not None:
        return _log_client
    async with _log_client_lock:
        if _log_client is None:
            _log_client = httpx.AsyncClient(
                base_url=settings.BACKEND_BASE_URL,
                # 적재 자체는 best-effort — 짧게 잡는다 (connect 1s / read 2s).
                # 타임아웃되어도 사용자 흐름엔 영향 없음.
                timeout=httpx.Timeout(2.0, connect=1.0),
                headers={"X-Service-Key": settings.SERVICE_API_KEY},
            )
    return _log_client


async def close_solar_usage_client() -> None:
    """앱 종료 lifespan 에서 호출 — httpx 클라이언트 정리."""
    global _log_client
    if _log_client is not None:
        await _log_client.aclose()
        _log_client = None


# ============================================================
# 콜백 핸들러
# ============================================================


class SolarUsageCallback(AsyncCallbackHandler):
    """
    Solar API 호출 1건마다 token usage 를 추출해 Backend 에 적재하는 콜백.

    LangChain 의 `AsyncCallbackHandler` 를 상속하며, ChatOpenAI 인스턴스의
    `callbacks=[SolarUsageCallback(model_name)]` 로 부착된다.

    추적 흐름:
    1. `on_llm_start` — 호출 시작 시각 기록 (kwargs 의 run_id 키로 매핑).
    2. `on_llm_end` — usage 추출 → 비용 산정 → Backend POST (asyncio.create_task).
    3. `on_llm_error` — 시작 시각 기록만 정리 (적재는 하지 않음 — usage 데이터 없음).

    실패 처리:
    - usage 누락(LangChain 버전 차이) → 경고 로그 후 스킵.
    - HTTP 실패/타임아웃 → 경고 로그 후 스킵 (재시도 없음 — best-effort).
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name
        # run_id → 시작 시각(unix timestamp ms) 매핑 — duration 계산용.
        self._start_times_ms: dict[UUID, int] = {}

    # ────────────────────────────────────────────
    # LangChain 훅
    # ────────────────────────────────────────────

    async def on_llm_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002 — LangChain 시그니처 준수
        prompts: list[str],          # noqa: ARG002
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        tags: list[str] | None = None,       # noqa: ARG002
        metadata: dict[str, Any] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        self._start_times_ms[run_id] = _now_ms()

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],   # noqa: ARG002
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        tags: list[str] | None = None,       # noqa: ARG002
        metadata: dict[str, Any] | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        # Chat 모델 호출은 별도 훅으로 들어옴 (on_llm_start 가 아닐 수 있음).
        await self.on_llm_start(
            serialized,
            [],
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        started_ms = self._start_times_ms.pop(run_id, None)
        duration_ms = (_now_ms() - started_ms) if started_ms is not None else None

        usage = _extract_usage(response)
        if usage is None:
            # 응답 형식이 달라 usage 를 못 뽑은 경우 — 경고 후 스킵.
            logger.debug(
                "solar_usage_callback_no_usage",
                model=self._model_name,
                hint="LLMResult.llm_output['token_usage'] 가 비어 있음",
            )
            return

        prompt_tokens, completion_tokens, total_tokens = usage

        agent_name = get_current_agent()
        cost_usd = estimate_cost_usd(self._model_name, prompt_tokens, completion_tokens)

        # fire-and-forget — 응답 흐름을 기다리지 않음
        asyncio.create_task(
            _post_usage_log(
                model=self._model_name,
                agent_name=agent_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                started_ms=started_ms,
                duration_ms=duration_ms,
            )
        )

    async def on_llm_error(
        self,
        error: BaseException,         # noqa: ARG002
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        # 시작 시각만 정리 — 적재는 안 함 (usage 데이터 없음).
        self._start_times_ms.pop(run_id, None)


# ============================================================
# 내부 헬퍼
# ============================================================


def _now_ms() -> int:
    """현재 unix timestamp ms — duration 계산용."""
    return int(datetime.now().timestamp() * 1000)


def _extract_usage(response: LLMResult) -> tuple[int, int, int] | None:
    """
    LLMResult 에서 (prompt, completion, total) 토큰 수를 추출한다.

    LangChain 버전·모델별 응답 형식 차이를 흡수하기 위해 두 경로를 모두 시도한다.

    1) `response.llm_output["token_usage"]` — 전통적 위치 (ChatOpenAI 기본).
    2) `response.generations[*][0].message.usage_metadata` — 최신 ChatOpenAI 위치.

    Returns:
        (prompt, completion, total) 또는 None (둘 다 비면).
    """
    # 경로 1: llm_output["token_usage"]
    llm_output = response.llm_output or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage")
    if usage:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        if prompt or completion or total:
            return prompt, completion, total

    # 경로 2: 첫 generation 의 usage_metadata
    try:
        first_gen = response.generations[0][0]
        msg = getattr(first_gen, "message", None)
        if msg is not None:
            meta = getattr(msg, "usage_metadata", None)
            if meta:
                prompt = int(meta.get("input_tokens") or 0)
                completion = int(meta.get("output_tokens") or 0)
                total = int(meta.get("total_tokens") or (prompt + completion))
                if prompt or completion or total:
                    return prompt, completion, total
    except (IndexError, AttributeError):
        pass

    return None


async def _post_usage_log(
    *,
    model: str,
    agent_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cost_usd: Any,  # Decimal — JSON 직렬화 시 str 캐스팅
    started_ms: int | None,
    duration_ms: int | None,
) -> None:
    """
    Backend 에 사용 로그 1건을 적재한다 (fire-and-forget).

    실패 시 경고 로그만 남기고 무시 — 사용자 흐름을 보호한다.
    """
    try:
        client = await _get_log_http_client()
        # request_started_at 은 콜백이 잡은 시작 시각. 없으면 현재 시각으로 폴백.
        if started_ms is not None:
            started_at = datetime.fromtimestamp(started_ms / 1000.0)
        else:
            started_at = datetime.now()
        payload = {
            "model": model,
            "agentName": agent_name,
            "promptTokens": int(prompt_tokens),
            "completionTokens": int(completion_tokens),
            "totalTokens": int(total_tokens),
            # Decimal 은 JSON 으로 직접 직렬화되지 않으므로 문자열로 보냄.
            # Backend 가 BigDecimal 로 역직렬화한다.
            "estimatedCostUsd": str(cost_usd),
            "requestStartedAt": started_at.isoformat(),
            "durationMs": duration_ms,
        }
        resp = await client.post("/api/v1/admin/stats/solar-usage/log", json=payload)
        if resp.status_code >= 400:
            logger.warning(
                "solar_usage_post_non_2xx",
                status=resp.status_code,
                model=model,
                agent_name=agent_name,
                body_preview=resp.text[:200],
            )
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.warning(
            "solar_usage_post_failed",
            model=model,
            agent_name=agent_name,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning(
            "solar_usage_post_unexpected_error",
            model=model,
            agent_name=agent_name,
            error=str(exc),
            error_type=type(exc).__name__,
        )
