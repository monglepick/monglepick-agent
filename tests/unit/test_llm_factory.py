"""
LLM 팩토리 단위 테스트 — 하이브리드 LLM (Ollama + Solar API) 지원.

테스트 대상:
- get_llm() / get_ollama_llm(): ChatOllama(Ollama) 생성 + 캐싱
- get_structured_llm(): 구조화 출력 Runnable 캐싱
- get_intent_llm(): INTENT_MODEL 사용 확인 (local_only 모드)
- get_conversation_llm(): CONVERSATION_MODEL 사용 확인
- get_explanation_llm(): EXPLANATION_MODEL 사용 확인

캐시 변수:
- _ollama_cache: Ollama 모델 캐시
- _solar_cache: Solar API 모델 캐시
- _vllm_cache: vLLM 모델 캐시
- _structured_cache: 구조화 출력 캐시
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import monglepick.llm.factory as factory_module
from monglepick.agents.chat.models import IntentResult
from monglepick.config import settings


def _clear_all_caches():
    """모든 LLM 캐시를 초기화한다."""
    factory_module._ollama_cache.clear()
    factory_module._solar_cache.clear()
    factory_module._vllm_cache.clear()
    factory_module._structured_cache.clear()


class TestGetLlm:
    """get_llm() / get_ollama_llm() 함수 테스트."""

    def setup_method(self):
        """각 테스트 전 캐시 초기화."""
        _clear_all_caches()

    def test_creates_chat_ollama_with_correct_params(self):
        """ChatOllama(Ollama)가 올바른 파라미터로 생성된다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            factory_module.get_llm(
                model="test-model",
                temperature=0.3,
                format="json",
            )
            # num_ctx, keep_alive 파라미터가 추가되었으므로 핵심 파라미터만 검증
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == "test-model"
            assert call_kwargs["temperature"] == 0.3
            assert call_kwargs["base_url"] == settings.OLLAMA_BASE_URL
            assert call_kwargs["format"] == "json"

    def test_cache_hit_same_params(self):
        """동일 파라미터로 호출하면 캐시에서 동일 인스턴스를 반환한다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            llm1 = factory_module.get_llm(model="test", temperature=0.5)
            llm2 = factory_module.get_llm(model="test", temperature=0.5)

            # 생성자는 1번만 호출
            assert mock_cls.call_count == 1
            # 동일 인스턴스
            assert llm1 is llm2

    def test_different_params_separate_instances(self):
        """다른 파라미터로 호출하면 별도 인스턴스를 생성한다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls:
            mock_cls.side_effect = [MagicMock(), MagicMock()]

            llm1 = factory_module.get_llm(model="model-a", temperature=0.3)
            llm2 = factory_module.get_llm(model="model-b", temperature=0.7)

            assert mock_cls.call_count == 2
            assert llm1 is not llm2

    def test_default_values(self):
        """기본값이 settings.CONVERSATION_MODEL, temp=0.5이다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            factory_module.get_llm()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == settings.CONVERSATION_MODEL
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["base_url"] == settings.OLLAMA_BASE_URL


class TestGetStructuredLlm:
    """get_structured_llm() 함수 테스트."""

    def setup_method(self):
        """각 테스트 전 캐시 초기화."""
        _clear_all_caches()

    def test_with_structured_output_called(self):
        """with_structured_output이 schema와 함께 호출된다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_ollama_cls, \
             patch("monglepick.llm.factory.ChatOpenAI") as mock_solar_cls, \
             patch.object(settings, "LLM_MODE", "local_only"):
            mock_instance = MagicMock()
            mock_ollama_cls.return_value = mock_instance

            factory_module.get_structured_llm(
                schema=IntentResult,
                model="test-model",
                temperature=0.1,
            )

            mock_instance.with_structured_output.assert_called_once_with(
                IntentResult, method="json_schema",
            )

    def test_structured_cache_hit(self):
        """동일 schema+model+temp로 호출하면 캐시에서 반환한다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_ollama_cls, \
             patch("monglepick.llm.factory.ChatOpenAI") as mock_solar_cls, \
             patch.object(settings, "LLM_MODE", "local_only"):
            mock_instance = MagicMock()
            mock_ollama_cls.return_value = mock_instance

            r1 = factory_module.get_structured_llm(IntentResult, "test", 0.1)
            r2 = factory_module.get_structured_llm(IntentResult, "test", 0.1)

            assert mock_instance.with_structured_output.call_count == 1
            assert r1 is r2


class TestConvenienceFunctions:
    """용도별 편의 함수 테스트 (local_only 모드)."""

    def setup_method(self):
        _clear_all_caches()

    def test_get_intent_llm_uses_intent_model(self):
        """get_intent_llm()이 local_only 모드에서 settings.INTENT_MODEL을 사용한다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls, \
             patch("monglepick.llm.factory.ChatOpenAI"), \
             patch.object(settings, "LLM_MODE", "local_only"):
            mock_instance = MagicMock()
            mock_instance.with_structured_output = MagicMock(return_value=MagicMock())
            mock_cls.return_value = mock_instance

            factory_module.get_intent_llm()

            # INTENT_MODEL이 사용되었는지 확인
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == settings.INTENT_MODEL

    def test_get_conversation_llm_uses_conversation_model(self):
        """get_conversation_llm()이 local_only 모드에서 settings.CONVERSATION_MODEL을 사용한다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls, \
             patch("monglepick.llm.factory.ChatOpenAI"), \
             patch.object(settings, "LLM_MODE", "local_only"), \
             patch.object(settings, "VLLM_ENABLED", False):
            mock_cls.return_value = MagicMock()

            factory_module.get_conversation_llm()

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == settings.CONVERSATION_MODEL

    def test_get_explanation_llm_uses_explanation_model(self):
        """get_explanation_llm()이 local_only 모드에서 settings.EXPLANATION_MODEL을 사용한다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls, \
             patch("monglepick.llm.factory.ChatOpenAI"), \
             patch.object(settings, "LLM_MODE", "local_only"):
            mock_cls.return_value = MagicMock()

            factory_module.get_explanation_llm()

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == settings.EXPLANATION_MODEL
