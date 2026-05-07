"""
고객센터 챗봇 — FAQ 캐시 & 의미 기반(임베딩) 검색 모듈.

### 설계 (2026-04-23 v2 — 하드코드 동의어/용언어미 방식 폐기)

기존 키워드 LIKE + 조사/동의어 하드코딩 접근은 케이스가 추가될 때마다 코드를
고쳐야 해서 확장성이 없었다 (사용자 실측 피드백). v2 는 Solar 임베딩으로
**문장 의미** 를 매칭한다.

1) `load_faqs()`
    - Backend `GET /api/v1/support/faqs` 로 공개 FAQ 전체를 가져와 캐시.
    - 각 FAQ 는 `question + " " + answer` 를 `embed_texts()` 로 배치 임베딩.
    - 결과는 normalized 4096-d 벡터로 메모리에 보관 (TTL 300s).

2) `search(user_message, extra_keywords=None, top_k=3)`
    - 사용자 발화(+ intent_classifier 가 뽑은 키워드) 를 `embed_query_async()`
      로 단일 쿼리 벡터화.
    - 캐시된 FAQ 벡터들과 코사인 유사도(= 내적, 정규화되어 있음) 계산.
    - **3-tier 분류**:
         HIGH (≥ 0.75) → 강한 매칭, `match_tier="high"`
         MID  (0.55 ~ 0.75) → 유사하지만 100% 아님, `match_tier="mid"`
         LOW  (< 0.55) → 탈락
    - top_k 반환 시 HIGH 우선, 동점이면 helpful_count/sort_order tiebreaker.

### 3-tier 활용 (노드 레이어에서 결정)
- HIGH 1건 이상 → `answer_generator` 정규 프롬프트, `needs_human=False`
- MID 만 있음    → `answer_generator` "augment" 프롬프트 (유사 FAQ 요약 +
                    "정확한 답변은 1:1 문의" 유도), `needs_human=True`
- 매칭 0건       → `fallback_responder`, `needs_human=True`

### 실패 대응
- 임베딩 API (Upstage Solar) 장애 시: 캐시가 비면 `search()` 는 빈 리스트 반환
  → 상위 그래프는 자연스럽게 `fallback_responder` 로 유도 (1:1 문의 안내).
- 단건 FAQ 임베딩 실패는 전체를 막지 않는다 (해당 FAQ 만 매칭 후보에서 제외).
"""

from __future__ import annotations

import asyncio
import time

import httpx
import numpy as np
import structlog

from monglepick.agents.support_assistant.models import FaqDoc, MatchedFaq
from monglepick.config import settings
from monglepick.data_pipeline.embedder import embed_query_async, embed_texts

logger = structlog.get_logger(__name__)


# =============================================================================
# 캐시 파라미터
# =============================================================================

# FAQ 목록 재조회 주기. 관리자 CRUD 반영까지 최대 5분 지연 허용.
_CACHE_TTL_SECONDS = 300.0

# Backend 호출 타임아웃 (FAQ 목록 조회). 챗봇이 통째로 지연되면 안 되므로 짧게.
_BACKEND_TIMEOUT_SECONDS = 3.0

# 기본 top-K — answer_generator 컨텍스트에 넣을 FAQ 수.
DEFAULT_TOP_K = 3

# 3-tier 임계값 (코사인 유사도 기준). 오프라인 평가로 후속 조정 가능.
SCORE_HIGH_THRESHOLD = 0.75   # 이 이상이면 "정답 FAQ"
SCORE_MID_THRESHOLD = 0.55    # 이 이상이면 "유사한 FAQ, 보강용"
# MID 미만은 매칭 실패로 간주.

# FAQ 임베딩 입력 텍스트 길이 상한. Upstage passage 모델은 긴 입력도 처리 가능하지만
# 단일 FAQ 당 과도하게 길면 배치 비용이 커진다.
_FAQ_EMBED_TEXT_LIMIT = 2000


# =============================================================================
# 내부 상태
# =============================================================================

# 모듈 전역 캐시. asyncio.Lock 으로 동시 갱신 보호.
_cache_lock = asyncio.Lock()
_cached_faqs: list[FaqDoc] = []
# faq_id → normalized 4096-d numpy 벡터
_cached_faq_vectors: dict[int, np.ndarray] = {}
_cache_loaded_at: float = 0.0

# FAQ 목록 호출 전용 httpx 싱글턴.
_cached_http_client: httpx.AsyncClient | None = None


# =============================================================================
# Backend FAQ 목록 조회
# =============================================================================


async def _get_http_client() -> httpx.AsyncClient:
    """FAQ 조회 전용 httpx.AsyncClient 싱글턴."""
    global _cached_http_client
    if _cached_http_client is None or _cached_http_client.is_closed:
        _cached_http_client = httpx.AsyncClient(
            base_url=settings.BACKEND_BASE_URL,
            timeout=_BACKEND_TIMEOUT_SECONDS,
        )
    return _cached_http_client


async def _fetch_faqs_from_backend() -> list[FaqDoc]:
    """
    Backend 에서 FAQ 전체 목록을 한 번 가져와 FaqDoc 리스트로 변환한다.

    실패 시 빈 리스트 반환 — 에러 전파 금지 (챗봇은 폴백으로 계속 돌아야 한다).
    """
    client = await _get_http_client()
    try:
        response = await client.get("/api/v1/support/faqs")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning(
            "support_faq_cache_fetch_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return []

    raw = response.json()
    if not isinstance(raw, list):
        logger.warning("support_faq_cache_unexpected_shape", type=type(raw).__name__)
        return []

    faqs: list[FaqDoc] = []
    for row in raw:
        try:
            faqs.append(
                FaqDoc(
                    faq_id=int(row.get("faqId", 0)),
                    category=str(row.get("category", "GENERAL")),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    helpful_count=int(row.get("helpfulCount", 0) or 0),
                    sort_order=row.get("sortOrder"),
                )
            )
        except Exception as exc:  # noqa: BLE001 — 한 건 실패가 전체를 막지 않도록
            logger.debug("support_faq_cache_row_skip", error=str(exc), row=row)
    return faqs


# =============================================================================
# 임베딩 헬퍼
# =============================================================================


def _compose_faq_text(faq: FaqDoc) -> str:
    """
    FAQ 한 건을 임베딩 입력 문자열로 직렬화한다.

    질문을 앞에 둬서 질문 의미가 우세하게 반영되도록 구성한다.
    너무 긴 답변은 잘라서 배치 비용을 제한한다.
    """
    text = f"{faq.question} {faq.answer}".strip()
    if len(text) > _FAQ_EMBED_TEXT_LIMIT:
        text = text[:_FAQ_EMBED_TEXT_LIMIT]
    return text


async def _compute_faq_vectors(faqs: list[FaqDoc]) -> dict[int, np.ndarray]:
    """
    FAQ 목록에 대해 배치 임베딩을 계산한다. 실패하면 빈 dict 반환.

    embed_texts 는 동기 HTTP 호출이라 asyncio.to_thread 로 감싸 event loop 를
    막지 않도록 한다.
    """
    if not faqs:
        return {}
    texts = [_compose_faq_text(f) for f in faqs]
    try:
        matrix = await asyncio.to_thread(embed_texts, texts)
    except Exception as exc:  # noqa: BLE001 — 임베딩 API 장애는 챗봇 폴백 경로로
        logger.warning(
            "support_faq_embed_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            faq_count=len(faqs),
        )
        return {}

    if matrix is None or len(matrix) != len(faqs):
        logger.warning(
            "support_faq_embed_shape_mismatch",
            returned=len(matrix) if matrix is not None else -1,
            expected=len(faqs),
        )
        return {}

    # Solar embedding 은 이미 magnitude=1 로 정규화되어 있으나 방어적으로 재정규화.
    vectors: dict[int, np.ndarray] = {}
    for faq, vec in zip(faqs, matrix):
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            arr = arr / norm
        vectors[faq.faq_id] = arr
    return vectors


# =============================================================================
# 퍼블릭 — 캐시 로딩 / 검색
# =============================================================================


async def load_faqs(force_reload: bool = False) -> list[FaqDoc]:
    """
    FAQ 캐시를 반환한다. TTL 만료 또는 force_reload=True 면 재조회 + 재임베딩.

    다중 요청이 TTL 만료 순간에 동시에 들어와도 Backend/Upstage 를 한 번만 호출하도록
    `_cache_lock` 으로 보호한다.
    """
    global _cached_faqs, _cached_faq_vectors, _cache_loaded_at

    now = time.time()
    stale = (now - _cache_loaded_at) > _CACHE_TTL_SECONDS
    if not force_reload and not stale and _cached_faqs:
        return _cached_faqs

    async with _cache_lock:
        # lock 획득 후 재검사 — 이미 다른 코루틴이 갱신했을 수 있다.
        now = time.time()
        stale = (now - _cache_loaded_at) > _CACHE_TTL_SECONDS
        if not force_reload and not stale and _cached_faqs:
            return _cached_faqs

        fetched = await _fetch_faqs_from_backend()
        if not fetched:
            # Backend 실패했지만 이전 캐시가 있으면 그대로 서빙 — 챗봇 중단 방지.
            if _cached_faqs:
                logger.warning(
                    "support_faq_cache_keep_stale",
                    count=len(_cached_faqs),
                    age_seconds=round(now - _cache_loaded_at, 1),
                )
            return _cached_faqs

        # 임베딩 계산 — 실패해도 FAQ 목록은 교체한다 (벡터 없으면 search() 가 빈 결과).
        vectors = await _compute_faq_vectors(fetched)

        _cached_faqs = fetched
        _cached_faq_vectors = vectors
        _cache_loaded_at = now
        logger.info(
            "support_faq_cache_reloaded",
            faq_count=len(fetched),
            vector_count=len(vectors),
            embed_ok=bool(vectors),
        )
        return _cached_faqs


async def search(
    user_message: str,
    extra_keywords: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[MatchedFaq]:
    """
    사용자 발화를 임베딩 유사도로 FAQ top-K 에 매칭시켜 MatchedFaq 리스트를 돌려준다.

    Args:
        user_message: 원문 사용자 발화.
        extra_keywords: `SupportIntent.search_keywords` 등 보조 키워드.
            쿼리 임베딩 입력 텍스트에 덧붙여 의미적 단서를 강화한다.
        top_k: 반환 최대 건수. 기본 DEFAULT_TOP_K.

    Returns:
        MatchedFaq 리스트. score 내림차순. MID 임계값 미만은 제외.
        각 MatchedFaq 의 match_tier 는 "high" 또는 "mid".
        캐시/임베딩 불가 → 빈 리스트.
    """
    query_text = (user_message or "").strip()
    if extra_keywords:
        # 중복 제거 후 텍스트 뒤에 이어 붙여 의미 단서 강화.
        unique = []
        seen: set[str] = set()
        for kw in extra_keywords:
            kw_clean = (kw or "").strip()
            if kw_clean and kw_clean not in seen:
                seen.add(kw_clean)
                unique.append(kw_clean)
        if unique:
            query_text = f"{query_text} {' '.join(unique)}".strip()

    if not query_text:
        return []

    faqs = await load_faqs()
    if not faqs or not _cached_faq_vectors:
        return []

    # 쿼리 임베딩 — 실패 시 빈 결과로 fallback.
    try:
        qvec_raw = await embed_query_async(query_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "support_query_embed_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return []

    qvec = np.asarray(qvec_raw, dtype=np.float32)
    qnorm = float(np.linalg.norm(qvec))
    if qnorm <= 0.0:
        return []
    qvec = qvec / qnorm

    # 벡터화 dot product — normalized 이므로 cosine 과 동일.
    scored: list[tuple[float, FaqDoc]] = []
    for faq in faqs:
        fvec = _cached_faq_vectors.get(faq.faq_id)
        if fvec is None:
            continue
        sim = float(np.dot(qvec, fvec))
        if sim >= SCORE_MID_THRESHOLD:
            scored.append((sim, faq))

    # 정렬: score desc → helpful_count desc → sort_order asc
    scored.sort(
        key=lambda sf: (
            -sf[0],
            -sf[1].helpful_count,
            sf[1].sort_order if sf[1].sort_order is not None else 9_999,
        )
    )

    out: list[MatchedFaq] = []
    for score, faq in scored[:top_k]:
        tier = "high" if score >= SCORE_HIGH_THRESHOLD else "mid"
        out.append(
            MatchedFaq(
                faq_id=faq.faq_id,
                category=faq.category,
                question=faq.question,
                answer=faq.answer,
                score=round(score, 3),
                match_tier=tier,
            )
        )
    return out


# =============================================================================
# 테스트/디버그 유틸 — 외부에서 캐시 조작이 필요할 때
# =============================================================================


def _reset_cache_for_tests() -> None:
    """테스트에서 모듈 캐시를 직접 초기화할 때 사용."""
    global _cached_faqs, _cached_faq_vectors, _cache_loaded_at
    _cached_faqs = []
    _cached_faq_vectors = {}
    _cache_loaded_at = 0.0


def _install_test_cache(faqs: list[FaqDoc], vectors: dict[int, np.ndarray]) -> None:
    """테스트에서 임베딩을 pre-compute 해 직접 주입할 때 사용."""
    global _cached_faqs, _cached_faq_vectors, _cache_loaded_at
    _cached_faqs = list(faqs)
    _cached_faq_vectors = dict(vectors)
    _cache_loaded_at = time.time()
