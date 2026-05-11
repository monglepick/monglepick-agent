"""
Upstage Solar API 모델 단가 — 비용 추정 기준.

스코프:
- monglepick-agent 의 LangChain 콜백 (`SolarUsageCallback`) 에서 호출 1건마다 비용을 산정해
  Backend `solar_api_usage_log` 테이블에 적재하는 데 사용된다.
- Backend 는 Agent 가 보낸 비용을 그대로 누적하므로, 단가가 변경되면 본 모듈의 상수만
  수정하면 신규 호출부터 반영된다 (과거 행은 갱신되지 않음 — 이는 의도된 설계).

값의 출처:
- `scripts/run_mood_enrichment.py` 의 estimate_cost() 와 동일하게 `solar-pro/pro2/pro3` 의
  입력 $0.15 · 출력 $0.60 (per 1M tokens), `solar-mini` 의 입력/출력 $0.15 를 기본값으로 둔다.
- 알 수 없는 모델명이 들어오면 안전한 폴백 (solar-pro 기준) 단가를 사용하고 경고 로그를 남긴다.

운영 메모:
- 단가가 Upstage 측에서 인상/인하되면 본 dict 만 수정한 뒤 Agent 재배포로 반영.
- 임베딩(예: solar-embedding-1-large) 은 출력 토큰이 0 이므로 출력 단가가 적용되지 않으며
  입력 단가만 곱해진다.
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple

import structlog

logger = structlog.get_logger(__name__)


class ModelPricing(NamedTuple):
    """모델별 단가 ($/1M tokens) — 입력/출력 분리."""

    prompt_per_mtok_usd: Decimal
    completion_per_mtok_usd: Decimal


# ============================================================
# 단가 테이블
# ============================================================
#
# Upstage Solar 공식 가격 (2026 년 5월 기준 추정).
# 변경 시 본 dict 만 수정 + 코드 리뷰 + Agent 재배포로 반영된다.
# Backend 에는 이미 적재된 행을 갱신하지 않음 — 단가 변경은 신규 호출부터 적용.
SOLAR_PRICING: dict[str, ModelPricing] = {
    # solar-pro 계열 — 한국어 추론 품질 모델 (102B MoE)
    "solar-pro": ModelPricing(Decimal("0.15"), Decimal("0.60")),
    "solar-pro2": ModelPricing(Decimal("0.15"), Decimal("0.60")),
    "solar-pro3": ModelPricing(Decimal("0.15"), Decimal("0.60")),
    # solar-mini — 경량 모델 (입력=출력 같은 단가)
    "solar-mini": ModelPricing(Decimal("0.15"), Decimal("0.15")),
    # 임베딩 — 출력 토큰 0 이라 prompt 단가만 의미가 있음
    "solar-embedding-1-large": ModelPricing(Decimal("0.10"), Decimal("0.00")),
    "solar-embedding-1-small": ModelPricing(Decimal("0.05"), Decimal("0.00")),
}


# 알 수 없는 모델 폴백 — solar-pro 기준 (가장 자주 쓰는 모델).
# 이 폴백이 적용되면 경고 로그가 남으므로 운영 중 새 모델이 등장하면 곧바로 발견 가능.
_UNKNOWN_MODEL_FALLBACK: ModelPricing = ModelPricing(Decimal("0.15"), Decimal("0.60"))


# 1M = 10^6 — 단가 분모 정수.
_PER_MILLION: Decimal = Decimal("1000000")


def lookup_pricing(model: str) -> ModelPricing:
    """
    모델명을 받아 단가를 반환한다. 미등록 모델은 폴백 + 경고 로그.

    Args:
        model: Solar 모델명 (예: "solar-pro", "solar-mini").

    Returns:
        ModelPricing — (prompt_per_mtok_usd, completion_per_mtok_usd)
    """
    pricing = SOLAR_PRICING.get(model)
    if pricing is None:
        logger.warning(
            "solar_pricing_fallback_unknown_model",
            model=model,
            fallback_prompt=str(_UNKNOWN_MODEL_FALLBACK.prompt_per_mtok_usd),
            fallback_completion=str(_UNKNOWN_MODEL_FALLBACK.completion_per_mtok_usd),
            hint="monglepick.llm.solar_pricing.SOLAR_PRICING 에 모델 등록 필요",
        )
        return _UNKNOWN_MODEL_FALLBACK
    return pricing


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
    """
    단가 적용 후 호출 1건의 USD 비용을 산정한다.

    공식: cost = (prompt_tokens × prompt_단가 + completion_tokens × completion_단가) / 1_000_000

    Args:
        model: Solar 모델명.
        prompt_tokens: 입력 토큰 수 (>= 0).
        completion_tokens: 출력 토큰 수 (>= 0). 임베딩은 0.

    Returns:
        Decimal — 6자리까지 보존한 USD 비용 (마이크로 USD 단위).
    """
    pricing = lookup_pricing(model)
    prompt_cost = Decimal(int(prompt_tokens)) * pricing.prompt_per_mtok_usd / _PER_MILLION
    completion_cost = (
        Decimal(int(completion_tokens)) * pricing.completion_per_mtok_usd / _PER_MILLION
    )
    # quantize 해서 6자리(마이크로 USD)로 truncate — Backend DECIMAL(12,6) 와 정합.
    return (prompt_cost + completion_cost).quantize(Decimal("0.000001"))
