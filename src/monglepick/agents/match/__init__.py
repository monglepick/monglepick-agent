"""
Movie Match Agent 모듈 (§21).

두 영화의 교집합 특성을 분석하여 "함께 볼 영화"를 추천하는 전용 에이전트.

주요 구성 요소:
- models.py  : MovieMatchState(TypedDict), SharedFeatures, MatchScoreDetail,
               MatchedMovie, MovieMatchRequest, MovieMatchResponse + 유틸 함수
- nodes.py   : 6개 노드 (movie_loader, feature_extractor, query_builder,
               rag_retriever, match_scorer, explanation_generator)
- graph.py   : StateGraph 조립, SSE 스트리밍 인터페이스

핵심 알고리즘:
- 개별 유사도: 0.35*genre_jaccard + 0.25*mood_jaccard + 0.15*keyword_jaccard + 0.25*cosine
- 최종 스코어: min(sim_to_movie_1, sim_to_movie_2) — 양쪽 모두에 유사해야 높은 점수
- MMR 리랭킹: λ=0.7 (점수 70% + 다양성 30%)
"""
