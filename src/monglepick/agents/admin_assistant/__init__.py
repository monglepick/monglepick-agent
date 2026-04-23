"""
관리자 AI 에이전트 (Admin Assistant).

자연어 → Intent 분류 → Tool RAG → Tool-call loop → Narration 구조의
LangGraph StateGraph 기반 관리자 전용 에이전트.

설계서: docs/관리자_AI에이전트_설계서.md (v2.0)

Step 1 범위 (2026-04-23):
- 뼈대 + Intent 분류 + SSE 엔드포인트만 구현한다.
- 실제 Tool 실행(Tier 0~3), HITL 승인, 감사 로그는 후속 Step 에서 추가.
- 현재는 smalltalk intent 만 실응답 생성, 나머지는 "구현 예정" placeholder 를 반환한다.
"""
