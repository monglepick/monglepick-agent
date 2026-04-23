"""
고객센터 전용 '몽글이' 챗봇 에이전트.

엔드포인트: POST /api/v1/support/chat (SSE)
모델:       EXAONE 1.2B 몽글이 (vLLM) + Solar intent classifier
노드:       context_loader → intent_classifier → {smalltalk | faq_retriever + answer_generator | fallback} → response_formatter
설계:       docs/고객센터_AI챗봇_설계서.md (후속 작성 예정)
"""
