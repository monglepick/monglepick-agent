"""
LangSmith Evaluation 스크립트 — 골든 테스트셋 기반 자동 품질 평가.

3가지 평가 카테고리:
1. 의도+감정 분류 (Intent+Emotion Classification)
2. 선호 추출 (Preference Extraction)
3. 추천 품질 (End-to-End Recommendation Quality)

사용법:
    # 1. Dataset 생성 (최초 1회)
    PYTHONPATH=src uv run python scripts/run_evaluation.py --create-datasets

    # 2. 의도+감정 분류 평가 실행
    PYTHONPATH=src uv run python scripts/run_evaluation.py --eval intent

    # 3. 선호 추출 평가 실행
    PYTHONPATH=src uv run python scripts/run_evaluation.py --eval preference

    # 4. 추천 품질 종합 평가 실행
    PYTHONPATH=src uv run python scripts/run_evaluation.py --eval recommendation

    # 5. 전체 평가 실행
    PYTHONPATH=src uv run python scripts/run_evaluation.py --eval all

전제 조건:
- LANGCHAIN_API_KEY 환경변수 설정 필요
- Ollama 서버 실행 중 (의도+감정/선호 추출 평가 시)
- Docker 인프라 실행 중 (추천 품질 평가 시)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

# LangSmith 환경변수 설정 (import 전에 설정)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

from langsmith import Client  # noqa: E402
from langsmith.evaluation import evaluate  # noqa: E402


# ============================================================
# 골든 테스트 데이터셋 정의
# ============================================================

# --- 1. 의도+감정 분류 골든 데이터 (30개) ---
INTENT_EMOTION_GOLDEN: list[dict[str, Any]] = [
    # recommend 의도
    {"input": {"message": "우울한데 영화 추천해줘"}, "expected": {"intent": "recommend", "emotion": "sad"}},
    {"input": {"message": "신나는 액션 영화 뭐 없어?"}, "expected": {"intent": "recommend", "emotion": "excited"}},
    {"input": {"message": "여자친구랑 볼만한 로맨스 추천해줘"}, "expected": {"intent": "recommend", "emotion": None}},
    {"input": {"message": "혼자 볼 힐링 영화 추천 부탁해"}, "expected": {"intent": "recommend", "emotion": "calm"}},
    {"input": {"message": "스릴러 좋아하는데 뭐 볼만한 거 있어?"}, "expected": {"intent": "recommend", "emotion": None}},
    {"input": {"message": "오늘 기분 좋은데 뭐 재밌는 거 없나"}, "expected": {"intent": "recommend", "emotion": "happy"}},
    {"input": {"message": "화가 나서 뭔가 시원하게 복수하는 영화 보고 싶어"}, "expected": {"intent": "recommend", "emotion": "angry"}},
    {"input": {"message": "넷플릭스에서 볼만한 거 추천해줘"}, "expected": {"intent": "recommend", "emotion": None}},
    {"input": {"message": "봉준호 감독 영화 같은 거 추천"}, "expected": {"intent": "recommend", "emotion": None}},
    {"input": {"message": "인셉션 같은 영화 추천해줘"}, "expected": {"intent": "recommend", "emotion": None}},
    # search 의도
    {"input": {"message": "인셉션 검색해줘"}, "expected": {"intent": "search", "emotion": None}},
    {"input": {"message": "기생충이라는 영화 찾아줘"}, "expected": {"intent": "search", "emotion": None}},
    {"input": {"message": "최근 개봉한 마블 영화 뭐 있어?"}, "expected": {"intent": "search", "emotion": None}},
    # general 의도
    {"input": {"message": "안녕하세요!"}, "expected": {"intent": "general", "emotion": None}},
    {"input": {"message": "너는 누구야?"}, "expected": {"intent": "general", "emotion": None}},
    {"input": {"message": "오늘 날씨 좋다"}, "expected": {"intent": "general", "emotion": "happy"}},
    {"input": {"message": "고마워!"}, "expected": {"intent": "general", "emotion": "happy"}},
    {"input": {"message": "뭐 할 수 있어?"}, "expected": {"intent": "general", "emotion": None}},
    # info 의도
    {"input": {"message": "인셉션 줄거리 알려줘"}, "expected": {"intent": "info", "emotion": None}},
    {"input": {"message": "기생충 감독이 누구야?"}, "expected": {"intent": "info", "emotion": None}},
    {"input": {"message": "올드보이 평점 어때?"}, "expected": {"intent": "info", "emotion": None}},
    # 경계 케이스
    {"input": {"message": "슬픈 영화"}, "expected": {"intent": "recommend", "emotion": "sad"}},
    {"input": {"message": "무서운 거"}, "expected": {"intent": "recommend", "emotion": None}},
    {"input": {"message": "뭐 볼까"}, "expected": {"intent": "recommend", "emotion": None}},
    {"input": {"message": "ㅎㅎ"}, "expected": {"intent": "general", "emotion": "happy"}},
    # 복합 의도
    {"input": {"message": "요즘 우울한데 인터스텔라 같은 웅장한 SF 영화 추천해줘"}, "expected": {"intent": "recommend", "emotion": "sad"}},
    {"input": {"message": "가족이랑 같이 볼 수 있는 따뜻한 영화 없을까?"}, "expected": {"intent": "recommend", "emotion": None}},
    {"input": {"message": "지금 극장에서 뭐 하고 있어?"}, "expected": {"intent": "search", "emotion": None}},
    {"input": {"message": "아 진짜 스트레스 받는다 뭐 좀 재밌는 거 틀어줘"}, "expected": {"intent": "recommend", "emotion": "angry"}},
    {"input": {"message": "디즈니플러스에 새로 올라온 거 있어?"}, "expected": {"intent": "search", "emotion": None}},
]

# --- 2. 선호 추출 골든 데이터 (20개) ---
PREFERENCE_GOLDEN: list[dict[str, Any]] = [
    {
        "input": {"message": "SF 영화 추천해줘"},
        "expected": {"genre_preference": "SF", "mood": None},
    },
    {
        "input": {"message": "넷플릭스에서 볼 수 있는 로맨스 영화"},
        "expected": {"genre_preference": "로맨스", "platform": "넷플릭스"},
    },
    {
        "input": {"message": "인셉션 같은 영화 보고 싶어"},
        "expected": {"reference_movies": ["인셉션"]},
    },
    {
        "input": {"message": "2020년대 한국 영화 추천해줘"},
        "expected": {"era": "2020년대"},
    },
    {
        "input": {"message": "공포는 빼고 추천해줘"},
        "expected": {"exclude": "공포"},
    },
    {
        "input": {"message": "혼자 볼 잔잔한 힐링 영화"},
        "expected": {"mood": "힐링", "viewing_context": "혼자"},
    },
    {
        "input": {"message": "여자친구랑 극장에서 볼 로맨스 추천"},
        "expected": {"genre_preference": "로맨스", "viewing_context": "연인", "platform": "극장"},
    },
    {
        "input": {"message": "인터스텔라, 그래비티 같은 우주 영화 좋아해"},
        "expected": {"reference_movies": ["인터스텔라", "그래비티"]},
    },
    {
        "input": {"message": "90년대 느와르 영화 추천해줘"},
        "expected": {"genre_preference": "느와르", "era": "90년대"},
    },
    {
        "input": {"message": "디즈니+에서 아이랑 볼 수 있는 애니메이션"},
        "expected": {"genre_preference": "애니메이션", "platform": "디즈니+", "viewing_context": "가족"},
    },
    {
        "input": {"message": "왓챠에서 볼 수 있는 따뜻한 영화 추천해줘"},
        "expected": {"mood": "따뜻", "platform": "왓챠"},
    },
    {
        "input": {"message": "범죄 스릴러 좋아해"},
        "expected": {"genre_preference": "범죄 스릴러"},
    },
    {
        "input": {"message": "친구들이랑 볼 코미디 영화"},
        "expected": {"genre_preference": "코미디", "viewing_context": "친구"},
    },
    {
        "input": {"message": "기생충, 올드보이, 마더 같은 봉준호 영화"},
        "expected": {"reference_movies": ["기생충", "올드보이", "마더"]},
    },
    {
        "input": {"message": "19금 말고 가족이 볼 수 있는 영화"},
        "expected": {"exclude": "19금", "viewing_context": "가족"},
    },
    {
        "input": {"message": "틱톡에서 본 영화인데 이름이 기억 안 나"},
        "expected": {"genre_preference": None},
    },
    {
        "input": {"message": "액션 코미디 장르 좋아하는데 최근 영화로"},
        "expected": {"genre_preference": "액션 코미디", "era": "최근"},
    },
    {
        "input": {"message": "웅장하고 몰입감 있는 전쟁 영화"},
        "expected": {"genre_preference": "전쟁", "mood": "웅장"},
    },
    {
        "input": {"message": "일본 애니메이션 추천해줘"},
        "expected": {"genre_preference": "애니메이션"},
    },
    {
        "input": {"message": "티빙에서 독점 공개한 한국 드라마 영화"},
        "expected": {"genre_preference": "드라마", "platform": "티빙"},
    },
]

# --- 3. 추천 품질 E2E 골든 데이터 (10개) ---
RECOMMENDATION_E2E_GOLDEN: list[dict[str, Any]] = [
    {
        "input": {"user_id": "eval_user_1", "message": "우울한데 따뜻한 힐링 영화 추천해줘"},
        "expected": {"has_response": True, "intent": "recommend", "min_movies": 1},
    },
    {
        "input": {"user_id": "eval_user_2", "message": "인셉션 같은 SF 스릴러 추천해줘"},
        "expected": {"has_response": True, "intent": "recommend", "min_movies": 1},
    },
    {
        "input": {"user_id": "eval_user_3", "message": "넷플릭스에서 볼만한 액션 영화"},
        "expected": {"has_response": True, "intent": "recommend", "min_movies": 0},
    },
    {
        "input": {"user_id": "eval_user_4", "message": "가족과 함께 볼 수 있는 따뜻한 영화"},
        "expected": {"has_response": True, "intent": "recommend", "min_movies": 0},
    },
    {
        "input": {"user_id": "eval_user_5", "message": "봉준호 감독 영화 같은 거 추천해줘"},
        "expected": {"has_response": True, "intent": "recommend", "min_movies": 0},
    },
    {
        "input": {"user_id": "eval_user_6", "message": "안녕하세요!"},
        "expected": {"has_response": True, "intent": "general", "min_movies": 0},
    },
    {
        "input": {"user_id": "eval_user_7", "message": "무서운 거 말고 재밌는 영화 추천"},
        "expected": {"has_response": True, "intent": "recommend", "min_movies": 0},
    },
    {
        "input": {"user_id": "eval_user_8", "message": "2024년 개봉 한국 영화 중에 볼만한 거"},
        "expected": {"has_response": True, "intent": "recommend", "min_movies": 0},
    },
    {
        "input": {"user_id": "eval_user_9", "message": "혼자 볼 잔잔한 다큐멘터리 영화"},
        "expected": {"has_response": True, "intent": "recommend", "min_movies": 0},
    },
    {
        "input": {"user_id": "eval_user_10", "message": "기생충 줄거리 알려줘"},
        "expected": {"has_response": True, "intent": "info", "min_movies": 0},
    },
]


# ============================================================
# Evaluator 함수 정의
# ============================================================

def eval_intent_accuracy(run, example) -> dict:
    """
    의도 분류 정확도 평가.

    LLM 출력의 intent가 골든 데이터의 expected intent와 일치하는지 확인한다.
    정확히 일치하면 1.0, 불일치하면 0.0.

    Returns:
        {"key": "intent_accuracy", "score": 0.0 | 1.0}
    """
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    score = 1.0 if predicted == expected else 0.0
    return {"key": "intent_accuracy", "score": score}


def eval_emotion_accuracy(run, example) -> dict:
    """
    감정 분류 정확도 평가.

    LLM 출력의 emotion이 골든 데이터와 일치하는지 확인한다.
    둘 다 None이면 정확, 하나만 None이면 부정확.

    Returns:
        {"key": "emotion_accuracy", "score": 0.0 | 1.0}
    """
    predicted = run.outputs.get("emotion")
    expected = example.outputs.get("emotion")

    # 둘 다 None이면 정확
    if predicted is None and expected is None:
        return {"key": "emotion_accuracy", "score": 1.0}
    # 하나만 None이면 부정확
    if predicted is None or expected is None:
        return {"key": "emotion_accuracy", "score": 0.0}
    # 감정 일치 여부
    score = 1.0 if predicted == expected else 0.0
    return {"key": "emotion_accuracy", "score": score}


def eval_intent_confidence(run, example) -> dict:
    """
    의도 분류 신뢰도 평가.

    신뢰도 값 자체를 점수로 사용한다 (0.0~1.0).
    높을수록 모델이 확신한 것.

    Returns:
        {"key": "intent_confidence", "score": float}
    """
    confidence = run.outputs.get("confidence", 0.0)
    return {"key": "intent_confidence", "score": float(confidence)}


def eval_preference_genre_match(run, example) -> dict:
    """
    선호 추출 — 장르 일치 평가.

    골든 데이터에 genre_preference가 있으면 LLM 출력에도 있는지 확인.
    부분 일치(포함)도 인정한다.

    Returns:
        {"key": "genre_match", "score": 0.0 | 0.5 | 1.0}
    """
    expected_genre = example.outputs.get("genre_preference")
    if expected_genre is None:
        return {"key": "genre_match", "score": 1.0}  # 기대값 없으면 평가 스킵

    predicted_genre = run.outputs.get("genre_preference", "")
    if not predicted_genre:
        return {"key": "genre_match", "score": 0.0}

    # 정확 일치
    if expected_genre.lower() in predicted_genre.lower():
        return {"key": "genre_match", "score": 1.0}
    # 부분 일치 (첫 단어)
    if expected_genre.split()[0] in predicted_genre:
        return {"key": "genre_match", "score": 0.5}
    return {"key": "genre_match", "score": 0.0}


def eval_preference_reference_match(run, example) -> dict:
    """
    선호 추출 — 참조 영화 일치 평가.

    골든 데이터의 reference_movies가 LLM 출력에 포함되는지 확인.
    재현율(recall) 기반: 기대 영화 중 몇 개가 추출되었는지.

    Returns:
        {"key": "reference_recall", "score": float (0.0~1.0)}
    """
    expected_refs = example.outputs.get("reference_movies", [])
    if not expected_refs:
        return {"key": "reference_recall", "score": 1.0}  # 기대값 없으면 스킵

    predicted_refs = run.outputs.get("reference_movies", [])
    if not predicted_refs:
        return {"key": "reference_recall", "score": 0.0}

    # 각 기대 영화가 추출 결과에 포함되는지 확인 (부분 문자열 매칭)
    matched = 0
    for expected in expected_refs:
        for predicted in predicted_refs:
            if expected in predicted or predicted in expected:
                matched += 1
                break

    recall = matched / len(expected_refs)
    return {"key": "reference_recall", "score": recall}


def eval_preference_completeness(run, example) -> dict:
    """
    선호 추출 — 필드 완성도 평가.

    골든 데이터에서 기대하는 필드들이 LLM 출력에 존재하는지 확인.
    존재하는 필드 비율을 점수로 반환.

    Returns:
        {"key": "field_completeness", "score": float (0.0~1.0)}
    """
    check_fields = ["genre_preference", "mood", "viewing_context", "platform", "era", "exclude"]
    expected_count = 0
    matched_count = 0

    for field in check_fields:
        expected_val = example.outputs.get(field)
        if expected_val is not None:
            expected_count += 1
            predicted_val = run.outputs.get(field)
            if predicted_val is not None and predicted_val != "":
                matched_count += 1

    if expected_count == 0:
        return {"key": "field_completeness", "score": 1.0}

    return {"key": "field_completeness", "score": matched_count / expected_count}


def eval_recommendation_has_response(run, example) -> dict:
    """
    E2E 추천 — 응답 존재 평가.

    그래프 실행 결과에 비어있지 않은 응답이 있는지 확인.

    Returns:
        {"key": "has_response", "score": 0.0 | 1.0}
    """
    response = run.outputs.get("response", "")
    score = 1.0 if response and len(response.strip()) > 0 else 0.0
    return {"key": "has_response", "score": score}


def eval_recommendation_intent_correct(run, example) -> dict:
    """
    E2E 추천 — 의도 분류 정확도.

    Returns:
        {"key": "e2e_intent_accuracy", "score": 0.0 | 1.0}
    """
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    score = 1.0 if predicted == expected else 0.0
    return {"key": "e2e_intent_accuracy", "score": score}


def eval_recommendation_movie_count(run, example) -> dict:
    """
    E2E 추천 — 추천 영화 수 평가.

    min_movies 이상의 영화가 추천되었는지 확인.

    Returns:
        {"key": "movie_count_pass", "score": 0.0 | 1.0}
    """
    movie_count = run.outputs.get("movie_count", 0)
    min_movies = example.outputs.get("min_movies", 0)
    score = 1.0 if movie_count >= min_movies else 0.0
    return {"key": "movie_count_pass", "score": score}


def eval_recommendation_latency(run, example) -> dict:
    """
    E2E 추천 — 응답 지연시간 평가.

    60초 이내이면 1.0, 120초 이내이면 0.5, 그 이상이면 0.0.

    Returns:
        {"key": "latency_score", "score": float}
    """
    latency_ms = run.outputs.get("latency_ms", 999999)
    if latency_ms <= 60000:
        score = 1.0
    elif latency_ms <= 120000:
        score = 0.5
    else:
        score = 0.0
    return {"key": "latency_score", "score": score}


def eval_response_length(run, example) -> dict:
    """
    E2E 추천 — 응답 길이 평가.

    너무 짧거나(< 20자) 너무 긴(> 2000자) 응답은 감점.

    Returns:
        {"key": "response_length_score", "score": float}
    """
    response = run.outputs.get("response", "")
    length = len(response)
    if length < 20:
        score = 0.0
    elif length > 2000:
        score = 0.5
    else:
        score = 1.0
    return {"key": "response_length_score", "score": score}


# ============================================================
# Target 함수 (LangSmith evaluate가 호출하는 함수)
# ============================================================

def target_intent_emotion(inputs: dict) -> dict:
    """의도+감정 분류 체인을 실행하고 결과를 반환한다."""
    from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

    result = asyncio.run(classify_intent_and_emotion(
        current_input=inputs["message"],
    ))
    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "emotion": result.emotion,
        "mood_tags": result.mood_tags,
    }


def target_preference(inputs: dict) -> dict:
    """선호 추출 체인을 실행하고 결과를 반환한다."""
    from monglepick.chains.preference_chain import extract_preferences

    result = asyncio.run(extract_preferences(
        current_input=inputs["message"],
    ))
    return {
        "genre_preference": result.genre_preference,
        "mood": result.mood,
        "viewing_context": result.viewing_context,
        "platform": result.platform,
        "reference_movies": result.reference_movies,
        "era": result.era,
        "exclude": result.exclude,
    }


def target_recommendation(inputs: dict) -> dict:
    """Chat Agent 그래프를 동기 실행하고 결과를 반환한다."""
    from monglepick.agents.chat.graph import run_chat_agent_sync

    start = time.perf_counter()
    state = asyncio.run(run_chat_agent_sync(
        user_id=inputs.get("user_id", ""),
        session_id="",  # 매번 새 세션
        message=inputs["message"],
    ))
    latency_ms = (time.perf_counter() - start) * 1000

    intent = state.get("intent")
    ranked = state.get("ranked_movies", [])

    return {
        "response": state.get("response", ""),
        "intent": intent.intent if intent else "",
        "movie_count": len(ranked),
        "latency_ms": round(latency_ms, 1),
    }


# ============================================================
# Dataset 생성
# ============================================================

def create_datasets(client: Client) -> None:
    """LangSmith에 3개 골든 데이터셋을 생성한다."""

    datasets = [
        ("monglepick-intent-emotion", INTENT_EMOTION_GOLDEN),
        ("monglepick-preference", PREFERENCE_GOLDEN),
        ("monglepick-recommendation-e2e", RECOMMENDATION_E2E_GOLDEN),
    ]

    for dataset_name, golden_data in datasets:
        # 기존 데이터셋이 있으면 삭제 후 재생성
        try:
            existing = client.read_dataset(dataset_name=dataset_name)
            client.delete_dataset(dataset_id=existing.id)
            print(f"  기존 데이터셋 삭제: {dataset_name}")
        except Exception:
            pass  # 존재하지 않으면 무시

        # 데이터셋 생성
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=f"몽글픽 {dataset_name} 골든 테스트셋",
        )

        # Example 추가
        for item in golden_data:
            client.create_example(
                inputs=item["input"],
                outputs=item["expected"],
                dataset_id=dataset.id,
            )

        print(f"  ✅ {dataset_name}: {len(golden_data)}개 Example 생성")

    print("\n모든 데이터셋 생성 완료!")
    print("LangSmith 콘솔에서 확인: https://smith.langchain.com/datasets")


# ============================================================
# 평가 실행
# ============================================================

def run_intent_evaluation(client: Client) -> None:
    """의도+감정 분류 평가를 실행한다."""
    print("\n🎯 의도+감정 분류 평가 시작...")
    print("=" * 60)

    results = evaluate(
        target_intent_emotion,
        data="monglepick-intent-emotion",
        evaluators=[
            eval_intent_accuracy,
            eval_emotion_accuracy,
            eval_intent_confidence,
        ],
        experiment_prefix="intent-emotion",
        description="의도+감정 분류 정확도 평가 (qwen3.5:35b-a3b)",
        max_concurrency=1,  # Ollama 직렬 처리
    )

    # 결과 요약 출력
    print("\n📊 의도+감정 분류 평가 결과:")
    print(f"  실험 URL: {results.experiment_url}")
    _print_summary(results)


def run_preference_evaluation(client: Client) -> None:
    """선호 추출 평가를 실행한다."""
    print("\n🎯 선호 추출 평가 시작...")
    print("=" * 60)

    results = evaluate(
        target_preference,
        data="monglepick-preference",
        evaluators=[
            eval_preference_genre_match,
            eval_preference_reference_match,
            eval_preference_completeness,
        ],
        experiment_prefix="preference",
        description="선호 추출 정확도 평가 (exaone-32b)",
        max_concurrency=1,
    )

    print("\n📊 선호 추출 평가 결과:")
    print(f"  실험 URL: {results.experiment_url}")
    _print_summary(results)


def run_recommendation_evaluation(client: Client) -> None:
    """추천 품질 E2E 평가를 실행한다."""
    print("\n🎯 추천 품질 E2E 평가 시작...")
    print("=" * 60)
    print("⚠️  Docker 인프라 + Ollama 서버가 실행 중이어야 합니다.")

    results = evaluate(
        target_recommendation,
        data="monglepick-recommendation-e2e",
        evaluators=[
            eval_recommendation_has_response,
            eval_recommendation_intent_correct,
            eval_recommendation_movie_count,
            eval_recommendation_latency,
            eval_response_length,
        ],
        experiment_prefix="recommendation-e2e",
        description="추천 품질 종합 E2E 평가 (전체 그래프 14노드)",
        max_concurrency=1,
    )

    print("\n📊 추천 품질 E2E 평가 결과:")
    print(f"  실험 URL: {results.experiment_url}")
    _print_summary(results)


def _print_summary(results) -> None:
    """평가 결과 요약을 출력한다."""
    try:
        # LangSmith evaluate 결과에서 집계 데이터 추출
        summary = results.to_pandas()
        print(f"  총 케이스: {len(summary)}개")

        # 각 evaluator 컬럼의 평균 점수 출력
        score_cols = [col for col in summary.columns if col.startswith("feedback.")]
        for col in score_cols:
            metric_name = col.replace("feedback.", "")
            mean_score = summary[col].mean()
            print(f"  {metric_name}: {mean_score:.3f}")
    except Exception as e:
        print(f"  (요약 출력 실패: {e})")
        print("  → LangSmith 콘솔에서 상세 결과를 확인하세요.")


# ============================================================
# CLI 엔트리포인트
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="몽글픽 LangSmith Evaluation")
    parser.add_argument(
        "--create-datasets",
        action="store_true",
        help="LangSmith에 골든 데이터셋 생성 (최초 1회)",
    )
    parser.add_argument(
        "--eval",
        choices=["intent", "preference", "recommendation", "all"],
        help="실행할 평가 카테고리",
    )

    args = parser.parse_args()

    # API 키 확인
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if not api_key:
        print("❌ LANGCHAIN_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 LANGCHAIN_API_KEY=lsv2_pt_xxx 를 추가하세요.")
        sys.exit(1)

    client = Client()

    if args.create_datasets:
        print("📦 LangSmith 골든 데이터셋 생성 중...")
        create_datasets(client)
        return

    if args.eval:
        if args.eval in ("intent", "all"):
            run_intent_evaluation(client)
        if args.eval in ("preference", "all"):
            run_preference_evaluation(client)
        if args.eval in ("recommendation", "all"):
            run_recommendation_evaluation(client)
        return

    # 인수 없으면 도움말 표시
    parser.print_help()


if __name__ == "__main__":
    main()
