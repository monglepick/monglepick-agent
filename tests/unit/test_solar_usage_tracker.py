"""
Upstage Solar API 사용량/비용 추적 단위 테스트.

검증 범위:
- `solar_pricing.estimate_cost_usd` — 알려진 모델 / 알 수 없는 모델 폴백, 임베딩(출력 0).
- `solar_usage_tracker._extract_usage` — legacy llm_output 경로 + 신 usage_metadata 경로.
- `SolarUsageCallback` — on_llm_start/on_llm_end run_id 매핑, agent_name ContextVar 주입,
  Backend POST 페이로드 형태(httpx mock 으로 가로챔).
- `SolarUsageAttributionMiddleware` 의 path → agent_name 매핑은 별도 미들웨어 테스트로
  분리 가능하나, 본 테스트에서는 ContextVar 만 직접 set/get 검증.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.outputs import Generation, LLMResult

from monglepick.llm.solar_pricing import (
    SOLAR_PRICING,
    estimate_cost_usd,
    lookup_pricing,
)
from monglepick.llm.solar_usage_tracker import (
    SolarUsageCallback,
    _extract_usage,
    get_current_agent,
    set_current_agent,
)


# ============================================================
# solar_pricing
# ============================================================


class TestSolarPricing:
    """모델 단가 테이블 + 비용 산정 정합성."""

    def test_lookup_known_model(self):
        """등록된 모델은 단가 그대로 반환."""
        p = lookup_pricing("solar-pro")
        assert p.prompt_per_mtok_usd == Decimal("0.15")
        assert p.completion_per_mtok_usd == Decimal("0.60")

    def test_lookup_unknown_model_falls_back(self):
        """미등록 모델은 폴백 (solar-pro 기준) — 비용이 0 이 아니어야 함."""
        cost = estimate_cost_usd("nonexistent-model-vNext", 1000, 500)
        # solar-pro 기준 (0.15/M * 1000) + (0.60/M * 500) = 0.00015 + 0.0003 = 0.00045
        assert cost == Decimal("0.000450")

    def test_estimate_cost_solar_pro(self):
        """solar-pro: prompt 1000 + completion 500 = 0.000450 USD"""
        cost = estimate_cost_usd("solar-pro", 1000, 500)
        assert cost == Decimal("0.000450")

    def test_estimate_cost_solar_mini_symmetric(self):
        """solar-mini: 입력=출력 같은 단가 → 토큰만 합쳐 계산."""
        cost = estimate_cost_usd("solar-mini", 1000, 1000)
        # 0.15/M * 2000 = 0.0003
        assert cost == Decimal("0.000300")

    def test_estimate_cost_zero_tokens(self):
        """0 토큰은 0 USD."""
        assert estimate_cost_usd("solar-pro", 0, 0) == Decimal("0.000000")

    def test_estimate_cost_embedding_output_zero(self):
        """임베딩은 completion=0 — prompt 단가만 적용."""
        # solar-embedding-1-large: 입력 0.10, 출력 0.00
        cost = estimate_cost_usd("solar-embedding-1-large", 1000, 0)
        # 0.10/M * 1000 = 0.0001
        assert cost == Decimal("0.000100")

    def test_pricing_table_contains_pro_variants(self):
        """운영에서 호출되는 모든 변형이 등록돼 있어야 함."""
        for variant in ("solar-pro", "solar-pro2", "solar-pro3", "solar-mini"):
            assert variant in SOLAR_PRICING


# ============================================================
# _extract_usage
# ============================================================


class TestExtractUsage:
    """LangChain 응답에서 token usage 추출."""

    def test_legacy_llm_output_path(self):
        """전통적 llm_output['token_usage'] 경로."""
        result = LLMResult(
            generations=[[Generation(text="hi")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            },
        )
        assert _extract_usage(result) == (100, 50, 150)

    def test_legacy_llm_output_aliased_usage_key(self):
        """ChatOpenAI 일부 버전이 'usage' 키로 노출하는 경우도 흡수."""
        result = LLMResult(
            generations=[[Generation(text="hi")]],
            llm_output={
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 20,
                    "total_tokens": 50,
                }
            },
        )
        assert _extract_usage(result) == (30, 20, 50)

    def test_total_tokens_inferred_when_missing(self):
        """total_tokens 누락 시 prompt+completion 합산."""
        result = LLMResult(
            generations=[[Generation(text="hi")]],
            llm_output={
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}
            },
        )
        assert _extract_usage(result) == (10, 5, 15)

    def test_empty_returns_none(self):
        """usage 데이터가 전혀 없으면 None."""
        result = LLMResult(
            generations=[[Generation(text="hi")]],
            llm_output={},
        )
        assert _extract_usage(result) is None

    def test_usage_metadata_path(self):
        """신 usage_metadata 경로 — message.usage_metadata 에서 추출."""
        # langchain_core.messages.AIMessage 사용
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration

        msg = AIMessage(
            content="hi",
            usage_metadata={
                "input_tokens": 80,
                "output_tokens": 40,
                "total_tokens": 120,
            },
        )
        result = LLMResult(
            generations=[[ChatGeneration(message=msg)]],
            llm_output={},
        )
        assert _extract_usage(result) == (80, 40, 120)


# ============================================================
# SolarUsageCallback
# ============================================================


class TestSolarUsageCallback:
    """콜백 라이프사이클 + Backend POST 페이로드."""

    @pytest.mark.asyncio
    async def test_on_llm_start_records_run_id(self):
        cb = SolarUsageCallback("solar-pro")
        rid = uuid4()
        await cb.on_llm_start({}, ["p"], run_id=rid)
        assert rid in cb._start_times_ms

    @pytest.mark.asyncio
    async def test_on_llm_end_clears_run_id(self):
        cb = SolarUsageCallback("solar-pro")
        rid = uuid4()
        await cb.on_llm_start({}, ["p"], run_id=rid)
        result = LLMResult(
            generations=[[Generation(text="hi")]],
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )
        with patch("monglepick.llm.solar_usage_tracker._post_usage_log", new=AsyncMock()):
            await cb.on_llm_end(result, run_id=rid)
        assert rid not in cb._start_times_ms

    @pytest.mark.asyncio
    async def test_on_llm_error_clears_run_id_without_post(self):
        """에러 시 run_id 정리만 하고 적재는 하지 않는다."""
        cb = SolarUsageCallback("solar-pro")
        rid = uuid4()
        await cb.on_llm_start({}, ["p"], run_id=rid)
        post_mock = AsyncMock()
        with patch("monglepick.llm.solar_usage_tracker._post_usage_log", new=post_mock):
            await cb.on_llm_error(RuntimeError("boom"), run_id=rid)
        assert rid not in cb._start_times_ms
        post_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_llm_end_no_usage_skips_post(self):
        """usage 가 비면 POST 시도하지 않음 (조용히 스킵)."""
        cb = SolarUsageCallback("solar-pro")
        rid = uuid4()
        await cb.on_llm_start({}, ["p"], run_id=rid)
        result_empty = LLMResult(generations=[[Generation(text="hi")]], llm_output={})
        post_mock = AsyncMock()
        with patch("monglepick.llm.solar_usage_tracker._post_usage_log", new=post_mock):
            await cb.on_llm_end(result_empty, run_id=rid)
        post_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_payload_includes_agent_name_from_contextvar(self):
        """ContextVar 에 set 한 agent_name 이 페이로드에 실린다."""
        cb = SolarUsageCallback("solar-pro")
        rid = uuid4()
        set_current_agent("chat")
        await cb.on_llm_start({}, ["p"], run_id=rid)

        result = LLMResult(
            generations=[[Generation(text="hi")]],
            llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
        )

        captured: dict[str, Any] = {}

        async def fake_post(**kwargs):
            captured.update(kwargs)

        with patch("monglepick.llm.solar_usage_tracker._post_usage_log", new=fake_post):
            await cb.on_llm_end(result, run_id=rid)
            # fire-and-forget 으로 만들어진 task 가 완료되도록 대기
            await asyncio.sleep(0.05)

        assert captured["model"] == "solar-pro"
        assert captured["agent_name"] == "chat"
        assert captured["prompt_tokens"] == 100
        assert captured["completion_tokens"] == 50
        assert captured["total_tokens"] == 150
        # solar-pro: 100*0.15/1M + 50*0.60/1M = 0.000015 + 0.000030 = 0.000045 USD
        assert captured["cost_usd"] == Decimal("0.000045")
        assert captured["duration_ms"] is not None and captured["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_post_payload_uses_unknown_when_no_contextvar(self):
        """미설정 컨텍스트는 'unknown' 으로 폴백."""
        cb = SolarUsageCallback("solar-mini")
        rid = uuid4()
        # 새로운 코루틴을 별도 Task 로 띄워 ContextVar 를 fresh 상태로 시작
        captured: dict[str, Any] = {}

        async def fake_post(**kwargs):
            captured.update(kwargs)

        async def run():
            # set_current_agent 호출 X
            await cb.on_llm_start({}, ["p"], run_id=rid)
            result = LLMResult(
                generations=[[Generation(text="hi")]],
                llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            )
            with patch("monglepick.llm.solar_usage_tracker._post_usage_log", new=fake_post):
                await cb.on_llm_end(result, run_id=rid)
                await asyncio.sleep(0.05)

        # 새 Task 로 분리해 부모 Context 의 set_current_agent 영향 차단
        await asyncio.create_task(run())

        assert captured["agent_name"] == "unknown"


# ============================================================
# ContextVar
# ============================================================


class TestAgentContextVar:
    """set/get_current_agent 의 단순 동작."""

    def test_set_get_round_trip(self):
        set_current_agent("test-agent-x")
        assert get_current_agent() == "test-agent-x"

    @pytest.mark.asyncio
    async def test_isolated_per_task(self):
        """asyncio.create_task 가 ContextVar 스냅샷을 복사 — 부모/자식 격리."""
        set_current_agent("parent")

        result = {"child": None}

        async def child_task():
            # 부모 값을 상속받음
            assert get_current_agent() == "parent"
            # 자식에서 변경
            set_current_agent("child")
            result["child"] = get_current_agent()

        await asyncio.create_task(child_task())

        # 부모 컨텍스트는 변경되지 않음
        assert get_current_agent() == "parent"
        assert result["child"] == "child"
