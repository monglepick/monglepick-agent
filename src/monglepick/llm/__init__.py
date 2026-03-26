"""
LLM 팩토리 + 동시성 제어 모듈 — 하이브리드 라우팅 (몽글이 + Solar API).

LLM_MODE 설정에 따라 체인별로 Ollama(몽글이) 또는 Solar API를 선택한다.
동일 파라미터에 대해 싱글턴 캐싱을 적용한다.
guarded_ainvoke로 모델별 동시 호출 수를 제한한다.
"""

from monglepick.llm.concurrency import (
    acquire_model_slot,
    release_model_slot,
    reset_semaphores,
)
from monglepick.llm.factory import (
    get_conversation_llm,
    get_emotion_llm,
    get_explanation_llm,
    get_intent_emotion_llm,
    get_intent_llm,
    get_llm,
    get_ollama_llm,
    get_preference_llm,
    get_question_llm,
    get_solar_api_llm,
    get_structured_llm,
    get_vision_llm,
    guarded_ainvoke,
)

__all__ = [
    # 하이브리드 LLM 생성
    "get_ollama_llm",
    "get_solar_api_llm",
    # 하위 호환
    "get_llm",
    "get_structured_llm",
    # 용도별 편의 함수 (하이브리드 라우팅)
    "get_intent_llm",
    "get_emotion_llm",
    "get_intent_emotion_llm",
    "get_preference_llm",
    "get_conversation_llm",
    "get_question_llm",
    "get_explanation_llm",
    "get_vision_llm",
    # 동시성 제어
    "guarded_ainvoke",
    "acquire_model_slot",
    "release_model_slot",
    "reset_semaphores",
]
