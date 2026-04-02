"""
Qdrant 공통 헬퍼 함수.

여러 모듈(data_pipeline.qdrant_loader, agents.match.nodes 등)에서
동일한 Qdrant point ID 변환 로직을 사용하므로, 단일 소스로 통합한다.
"""

from __future__ import annotations

import uuid


def to_point_id(doc_id: str) -> int | str:
    """
    movie_id를 Qdrant PointStruct ID로 변환한다.

    - 순수 숫자 ID (TMDB/Kaggle): int() 변환 (기존 호환)
    - 알파벳 포함 ID (KOBIS 코드 '2026A342' 등): uuid5() 변환

    Qdrant PointStruct ID는 int 또는 UUID(str) 형식만 허용한다.
    KOBIS 코드에 영문자가 포함될 수 있어 int() 변환이 실패하므로,
    uuid5(NAMESPACE_URL, "kobis:{doc_id}")를 사용하여 결정적 UUID를 생성한다.

    Args:
        doc_id: 영화 ID 문자열

    Returns:
        Qdrant 호환 point ID (int 또는 str UUID)
    """
    if doc_id.isdigit():
        return int(doc_id)
    # 알파벳 포함 ID → 결정적 UUID5 변환 (동일 입력 → 동일 UUID)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"kobis:{doc_id}"))
