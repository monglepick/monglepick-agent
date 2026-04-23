# 관리자 AI 에이전트 v3 재설계서 (2026-04-23)

> **핵심 원칙 — AI 는 쓰지 않는다.**
>
> AI 는 데이터를 조회(Read)하고, 사용자가 원하는 폼을 자동으로 채워주거나(Draft),
> 특정 관리 화면으로 안내(Navigate) 할 뿐이다. 실제 생성/수정/삭제는 **반드시
> 관리자가 해당 관리 페이지의 저장/실행 버튼을 직접 눌러야만** 반영된다.

v2 (2026-04-23 초) 대비 변경점:
- Tier 2/3 쓰기 tool 전면 제거 → Draft + Navigate 로 재분류
- HITL `interrupt()` + `confirm_keyword` 메커니즘 제거 (실제 쓰기 없음)
- `risk_gate` 노드 제거
- ReAct 루프 도입 (여러 read 를 순차 호출한 뒤 draft/navigate 로 종결)
- SSE 이벤트 2종 신규: `form_prefill`, `navigation` (기존 `confirmation_required` 는 예비 보관)

---

## 1. 3종 Tool 분류

### 1.1 Read Tool (Tier 0)
- Backend GET EP 를 직접 호출해 데이터를 가져온다.
- 자동 실행, 승인 없음.
- 결과는 `AdminApiResult.data` 로 반환되어 narrator 가 자연어 서술에 사용.
- Dashboard KPI, Users 목록/상세, Orders 목록/상세, Reports, Stats, Notices, FAQ 등.

### 1.2 Draft Tool (Tier 0)
- Backend 를 호출하지 **않는다.** LLM 이 사용자 발화 + 이전 read 결과를 바탕으로
  관리자 폼 필드(제목/내용/카테고리 등)를 구조화해 반환한다.
- SSE `form_prefill` 이벤트로 Client 에 전달 → Client 가 "해당 화면으로 열기" 버튼
  제공 → 클릭 시 `navigate(target_path, {state: {draft: ...}})`.
- 대상 페이지(예: NoticeCreateModal) 가 `location.state.draft` 를 초기 폼 값으로
  세팅 + 상단에 "AI 가 채운 내용이에요. 검토 후 저장하세요" 배너 노출.
- **저장은 관리자가 직접.** Backend 의 기존 POST/PUT/DELETE EP 는 관리자 페이지가
  호출하는 그대로 유지.

### 1.3 Navigate Tool (Tier 0)
- Backend 쓰기를 호출하지 **않는다.** 대상 엔티티를 (필요 시 read tool 로) 검색해
  특정 화면으로 이동할 URL 을 생성해 반환한다.
- SSE `navigation` 이벤트로 Client 에 전달 → "이동" 버튼 제공.
- 환불·계정 제재·포인트 조정 같은 위험한 작업은 **반드시 관리자가 해당 관리
  페이지에서 직접** 수행. AI 는 "찾아주고 링크 거는" 역할까지.
- 후보가 여러 개면 목록으로 제시 + 각각의 "이동" 버튼.

> 모든 tool 을 Tier 0 으로 통일한다. Backend 쓰기는 아예 일어나지 않으므로 위험도
> 개념이 없다. `ToolSpec.tier` 필드는 호환성 위해 남기되 전부 0.

---

## 2. ReAct 그래프 (v3)

```
START
  → context_loader
  → intent_classifier ──┐
                        │
    smalltalk ──────────┤
    ▼                   │
    smalltalk_responder │ stats / query / action
    │                   ▼
    │           tool_selector ◀────────┐
    │                   │              │ continue
    │         ┌─────────┼─────────┐    │
    │   pending=None  pending_tool_call│
    │         ▼         ▼              │
    │  smart_fallback  tool_executor ──┤
    │         │         ▼              │
    │         │      observation ──────┘
    │         │         │
    │         │         ├─ draft     → draft_emitter
    │         │         ├─ navigate  → navigator
    │         │         └─ finish    → narrator
    │         │         │
    │         └─────────┼─→ response_formatter → END
    ▼                   ▼
    response_formatter  ┘
        ▼
       END
```

### 2.1 State 추가 필드

```python
class AdminAssistantState(TypedDict, total=False):
    # ... 기존 필드 ...

    # ── ReAct 루프 상태 ──
    tool_call_history: list[ToolCall]          # 호출 순서 그대로 누적
    tool_results_history: list[dict]           # 각 호출의 결과 (축약본)
    hop_count: int                             # 현재 hop 수 (0부터 시작)

    # ── 종결 출력 (draft_emitter / navigator 가 세팅) ──
    form_prefill: dict | None                  # {target_path, draft_fields, action_label, summary}
    navigation: dict | None                    # {target_path, label, context_summary}

    # ── 제거된 필드 ──
    # awaiting_confirmation, confirmation_payload, confirmation_decision → 제거
```

### 2.2 노드

| 노드 | 역할 |
|------|------|
| `context_loader` | (기존 유지) JWT/role 복원 |
| `intent_classifier` | (기존 유지) smalltalk vs stats/query/action |
| `smalltalk_responder` | (기존 유지) |
| **`tool_selector`** | Solar `bind_tools` 로 단일 tool 선택. bind 목록에 `finish_task` 가상 tool 추가. 이전 `tool_call_history`/`tool_results_history` 를 프롬프트에 주입 |
| **`tool_executor`** | Read tool 만 실제 Backend 호출. Draft/Navigate tool 은 handler 가 Backend 호출 없이 payload 만 리턴 |
| **`observation`** (신규) | `tool_executor` 결과를 `tool_call_history` + `tool_results_history` 에 append. hop_count++. Draft/Navigate tool 결과면 다음 라우터에서 종결 경로로 분기 |
| **`draft_emitter`** (신규) | Draft tool 실행 직후 결과를 `state.form_prefill` 로 확정. SSE `form_prefill` 이벤트 발행 |
| **`navigator`** (신규) | Navigate tool 실행 직후 결과를 `state.navigation` 으로 확정. SSE `navigation` 이벤트 발행 |
| `narrator` | (기존 유지) 최종 자연어 요약 생성. Draft/Navigate 로 종결된 경우 "폼을 채워 두었어요" / "해당 화면으로 이동하실 수 있어요" 식 안내 |
| `response_formatter` | (기존 유지) |
| ~~`risk_gate`~~ | **제거** |
| ~~`smart_fallback_responder`~~ | 유지 (tool_selector 가 아무것도 못 고를 때) |

### 2.3 조건부 분기

```python
def route_after_tool_select(state):
    call = state.get("pending_tool_call")
    if call is None:
        return "smart_fallback_responder"
    return "tool_executor"                  # risk_gate 경유 X


def route_after_observation(state):
    # hop 상한 도달 → 강제 종료
    if state.get("hop_count", 0) >= MAX_HOPS:
        return "narrator"

    # 마지막 tool 이 finish_task(가상) → 종결
    last_call = state["tool_call_history"][-1] if state.get("tool_call_history") else None
    if last_call and last_call.tool_name == "finish_task":
        return "narrator"

    # 마지막 tool 이 draft_* → draft_emitter
    if last_call and last_call.tool_name.endswith("_draft"):
        return "draft_emitter"

    # 마지막 tool 이 goto_* → navigator
    if last_call and last_call.tool_name.startswith("goto_"):
        return "navigator"

    # 그 외 read tool → tool_selector 로 돌아가서 다음 hop
    return "tool_selector"
```

### 2.4 MAX_HOPS

- 기본값 **5** (토큰 비용 + 무한 루프 방어).
- 환경변수 `ADMIN_ASSISTANT_MAX_HOPS` 로 override 가능.

---

## 3. SSE 이벤트 (v3)

기존 10종 유지 + 2종 신규:

| 이벤트 | v2 | v3 | 페이로드 |
|--------|----|----|---------|
| session | O | O | `{session_id}` |
| status | O | O | `{phase, message, keepalive?}` |
| tool_call | O | O | `{tool_name, arguments, tier}` (매 hop 마다 발행) |
| tool_result | O | O | `{tool_name, ok, status_code, latency_ms, row_count, ref_id}` |
| token | O | O | `{delta}` |
| confirmation_required | O | (미사용) | — |
| **form_prefill** | — | **신규** | `{target_path, draft_fields: dict, action_label, summary, tool_name}` |
| **navigation** | — | **신규** | `{target_path, label, context_summary, candidates?: list}` |
| chart_data | O | O | (기존 예비) |
| table_data | O | O | (기존 예비) |
| report_chunk | O | O | (기존 예비) |
| done | O | O | `{}` |
| error | O | O | `{message}` |

### 3.1 form_prefill 예시

```json
{
  "event": "form_prefill",
  "data": {
    "target_path": "/admin/support?tab=notice&modal=create",
    "draft_fields": {
      "title": "오늘의 신작 업데이트 안내",
      "type": "NEWS",
      "pinned": true,
      "content": "오늘 몽글픽에 새로 추가된 영화 3편을 안내드려요..."
    },
    "action_label": "공지사항 작성 화면 열기",
    "summary": "최근 24시간 신규 입고 영화 3편 기준으로 공지 초안을 만들었어요.",
    "tool_name": "notice_draft"
  }
}
```

### 3.2 navigation 예시 (단건)

```json
{
  "event": "navigation",
  "data": {
    "target_path": "/admin/payment?tab=orders&orderId=ord_a1b2&action=refund",
    "label": "환불 화면으로 이동",
    "context_summary": "chulsoo@test.com 님의 최근 결제 주문 1건을 찾았어요.",
    "tool_name": "goto_order_refund"
  }
}
```

### 3.3 navigation 예시 (후보 여러 개)

```json
{
  "event": "navigation",
  "data": {
    "target_path": null,
    "label": "사용자를 선택하세요",
    "context_summary": "'chulsoo' 로 검색된 계정이 2개 있어요. 이동할 계정을 골라주세요.",
    "candidates": [
      {"target_path": "/admin/users?userId=u_aaa", "label": "chulsoo@test.com (2024-03 가입)"},
      {"target_path": "/admin/users?userId=u_bbb", "label": "chulsoo.park@gmail.com (2025-08 가입)"}
    ],
    "tool_name": "goto_user_detail"
  }
}
```

---

## 4. Tool 인벤토리 (총 ~65개)

### 4.1 Read Tool (~43개, Tier 0)

| 파일 | Tool | Backend EP |
|------|------|------------|
| `stats.py` (기존) | stats_overview / stats_trends / stats_revenue / stats_ai_service_overview / stats_community_overview | GET /api/v1/admin/stats/* |
| `dashboard.py` (신규) | dashboard_kpi / dashboard_trends / dashboard_recent | GET /api/v1/admin/dashboard/* |
| `users_read.py` (기존) | users_list / user_detail / user_activity / user_points_history / user_payments | GET /api/v1/admin/users/** |
| `users_read.py` (확장) | user_rewards / user_suspension_history | GET /api/v1/admin/users/{id}/rewards, /suspension-history |
| `content_read.py` (기존) | reports_list / toxicity_list / posts_list / reviews_list | GET /api/v1/admin/{reports,toxicity,posts,reviews} |
| `payment_read.py` (기존) | orders_list / order_detail / subscriptions_list | GET /api/v1/admin/payment/** |
| `payment_read.py` (확장) | point_histories / point_items | GET /api/v1/admin/point/** |
| `support_read.py` (기존) | faqs_list / help_articles_list / tickets_list | GET /api/v1/admin/support/** |
| `support_read.py` (확장) | faq_detail / help_article_detail / ticket_detail / notices_list / notice_detail | GET /api/v1/admin/support/**, /notices/** |
| `stats_extended.py` (신규) | stats_recommendation / stats_search_popular / stats_behavior / stats_retention / stats_subscription / stats_point_economy / stats_engagement / stats_content_performance / stats_funnel / stats_churn_risk (대표 10개만 노출) | GET /api/v1/admin/stats/** |
| `ai_ops_read.py` (신규) | quizzes_list / quiz_detail / chatbot_sessions / review_verifications_list / review_verification_detail | GET /api/v1/admin/ai/** |
| `system_read.py` (신규) | system_health / system_settings / system_logs_recent | GET /api/v1/admin/system/** |
| `settings_read.py` (신규) | audit_logs_list / admin_accounts_list / terms_list / banners_list | GET /api/v1/admin/{audit-logs,admins,terms,banners} |
| `chat_suggestions_read.py` (신규) | chat_suggestions_list | GET /api/v1/admin/chat-suggestions |

### 4.2 Draft Tool (~10개, Tier 0, Backend 호출 없음)

`admin_tools/drafts.py` 단일 파일에 집약.

| Tool | target_path | 주요 필드 |
|------|-------------|-----------|
| `notice_draft` | `/admin/support?tab=notice&modal=create` | title, type, pinned, content, startAt, endAt |
| `faq_draft` | `/admin/support?tab=faq&modal=create` | category, question, answer, tags |
| `help_article_draft` | `/admin/support?tab=help&modal=create` | title, category, content |
| `banner_draft` | `/admin/content-events?tab=banner&modal=create` | title, imageUrl, link, position, priority |
| `quiz_draft` | `/admin/ai?tab=quiz&modal=create` | movieId, question, choices, answerIndex, explanation |
| `chat_suggestion_draft` | `/admin/settings?tab=chat-sugg&modal=create` | surface, text, reason, tags |
| `term_draft` | `/admin/settings?tab=terms&modal=create` | type, version, content |
| `worldcup_candidate_draft` | `/admin/content-events?tab=worldcup&modal=create` | movieId, tier |
| `reward_policy_draft` | `/admin/settings?tab=reward-policy&modal=create` | code, pointAmount, condition |
| `point_pack_draft` | `/admin/payment?tab=point-packs&modal=create` | packCode, points, priceKrw |

각 handler 는 Backend 호출 없이 `AdminApiResult(ok=True, data={target_path, draft_fields, action_label, summary})` 를 돌려준다.

### 4.3 Navigate Tool (~12개, Tier 0, Backend 호출 없음)

`admin_tools/navigation.py` 단일 파일.

| Tool | 동작 | target_path |
|------|------|-------------|
| `goto_user_detail` | user 검색 후 상세 화면 | `/admin/users?userId=...` |
| `goto_user_suspend` | user 검색 후 정지 폼 prefill | `/admin/users?userId=...&action=suspend` |
| `goto_user_activate` | user 검색 후 복구 폼 | `/admin/users?userId=...&action=activate` |
| `goto_user_role_change` | user 검색 후 역할 변경 폼 | `/admin/users?userId=...&action=role` |
| `goto_order_detail` | order 검색 후 상세 | `/admin/payment?tab=orders&orderId=...` |
| `goto_order_refund` | order 검색 후 환불 폼 | `/admin/payment?tab=orders&orderId=...&action=refund` |
| `goto_subscription_manage` | user/subscription 검색 후 관리 화면 | `/admin/payment?tab=subscriptions&...` |
| `goto_points_adjust` | user 검색 후 포인트 조정 폼 | `/admin/users?userId=...&action=points-adjust` |
| `goto_token_grant` | user 검색 후 이용권 발급 폼 | `/admin/users?userId=...&action=tokens-grant` |
| `goto_report_detail` | report 검색 후 처리 화면 | `/admin/board?tab=reports&reportId=...` |
| `goto_ticket_detail` | ticket 검색 후 상세 | `/admin/support?tab=tickets&ticketId=...` |
| `goto_audit_log` | 조건부 감사로그 검색 화면 | `/admin/settings?tab=audit&q=...` |

각 handler 는 **내부적으로 필요한 read tool 을 한 번 호출해** 대상을 찾고 그 결과를
`navigation` payload 로 돌려준다. 후보 여러 개면 `candidates` 배열.

### 4.4 가상 Tool `finish_task`

- 실제 handler 없음. `tool_selector` 가 bind_tools 에 끼워 놓고 LLM 이 "더 이상 tool 이
  필요 없다" 고 판단하면 이 이름으로 호출.
- `observation` 라우터가 이 이름을 보고 narrator 로 직행.

---

## 5. Role × Tool 매트릭스 (재정의)

쓰기/위험 tool 이 없어져 매트릭스가 크게 단순화된다.

| Role | Read Tool | Draft Tool | Navigate Tool |
|------|-----------|-----------|---------------|
| SUPER_ADMIN | 전체 | 전체 | 전체 |
| ADMIN | 전체 | 전체 | 전체 |
| MODERATOR | users/content/reports/toxicity/posts/reviews read, support read | notice/faq/help draft | goto_user_*, goto_report/post/review/ticket_* |
| FINANCE_ADMIN | stats (전체), payment read | point_pack/reward_policy draft | goto_order_*, goto_subscription_*, goto_points_*, goto_token_* |
| SUPPORT_ADMIN | users read, support read | faq/help/notice draft | goto_user_*, goto_ticket_*, goto_audit_* |
| DATA_ADMIN | stats (전체), system read | (없음) | (없음) |
| AI_OPS_ADMIN | stats ai-service, ai_ops read | quiz/chat_suggestion draft | (없음) |
| STATS_ADMIN | stats 전체 | (없음) | (없음) |

Role enforcement 는 기존 `list_tools_for_role()` 유틸 그대로 사용. SUPER_ADMIN 은
필터 무시하고 전체 접근.

---

## 6. Prompt 변경점

### 6.1 tool_selector 프롬프트 (핵심 문구)

```
당신은 관리자 데이터 조회와 안내를 돕는 어시스턴트예요.

**중요한 제약**:
- 당신은 데이터를 **생성·수정·삭제할 수 없어요.**
- 생성/수정/삭제가 필요한 요청은 `*_draft` 도구로 폼 내용을 채워주거나,
  `goto_*` 도구로 해당 관리 화면으로 안내해 주세요.
- 금전·회원 제재 같은 위험한 요청은 반드시 `goto_*` 로 화면 링크만 제공해요.
  실제 실행은 관리자가 직접 버튼을 눌러야 해요.

**여러 단계가 필요하면**:
- 먼저 필요한 데이터를 `read` 도구로 수집하세요.
- 정보가 충분하면 `*_draft` 또는 `goto_*` 로 종결하세요.
- 더 이상 도구가 필요 없으면 `finish_task` 를 호출해 자연어로 답변하세요.

이전 호출 결과 요약: {tool_history_summary}
현재 hop: {hop_count} / {max_hops}
```

### 6.2 narrator 프롬프트 추가

- Draft 종결 시: "폼을 채워 두었어요. 검토 후 '저장' 버튼을 눌러주세요."
- Navigate 종결 시: "해당 화면으로 바로 이동하실 수 있어요. 실제 처리는 관리자 페이지에서 진행해 주세요."
- Read 종결 시: 기존과 동일.

---

## 7. Admin Client 연동

### 7.1 assistantApi.js dispatcher 확장

```js
case 'form_prefill':
  onFormPrefill?.(data);
  break;
case 'navigation':
  onNavigation?.(data);
  break;
```

### 7.2 AssistantChatPanel 카드 타입

- `FormPrefillCard`
  - 상단: `summary` + draft 주요 필드 요약
  - 하단: "[action_label]" 버튼 → `navigate(target_path, {state: {draft: draft_fields}})`
- `NavigationCard` (단건)
  - 상단: `context_summary`
  - 하단: "[label]" 버튼 → `navigate(target_path)`
- `NavigationCard` (다건)
  - 후보 리스트 + 각각의 "[label]" 버튼

### 7.3 대상 페이지 수정

각 대상 모달/폼이 `location.state.draft` 를 읽어 초기 form state 세팅 + 상단에
"AI 가 채운 내용이에요. 검토 후 저장하세요" 배너 노출. query param `action=refund/suspend/...`
은 해당 버튼/모달 자동 오픈 트리거로 사용.

### 7.4 ConfirmationDialog 제거

v2 의 Tier 2/3 승인 모달은 더 이상 호출되지 않음. 컴포넌트 자체는 당장 삭제하지
말고 v3 안정화까지 예비 보관.

---

## 8. 삭제되는 v2 자산

| 파일/노드 | 처리 |
|-----------|------|
| `admin_tools/users_write.py` (user_suspend) | 삭제. goto_user_suspend 로 대체 |
| `admin_tools/points_write.py` (points_manual_adjust) | 삭제. goto_points_adjust 로 대체 |
| `admin_tools/support_write.py` (faq_create) | 삭제. faq_draft 로 대체 |
| `admin_tools/settings_write.py` (banner_create) | 삭제. banner_draft 로 대체 |
| `agents/admin_assistant/nodes.py::risk_gate` | 삭제 |
| `agents/admin_assistant/models.py::ConfirmationPayload/ConfirmationDecision` | 삭제 또는 deprecated 마킹 |
| `api/admin_assistant.py::/resume` 엔드포인트 | 삭제 (더 이상 HITL 재개 없음) |
| `agents/admin_assistant/state::awaiting_confirmation/confirmation_*` | 삭제 |

Backend 쪽은 변경 없음.

---

## 9. 테스트 시나리오

| # | 시나리오 | 예상 SSE 흐름 |
|---|---------|---------------|
| 1 | "지난 7일 매출 얼마?" | tool_call(stats_revenue) → tool_result → token → done |
| 2 | "chulsoo 환불해줘" (1명) | tool_call(users_list) → tool_result → tool_call(orders_list) → tool_result → tool_call(goto_order_refund) → tool_result → **navigation** → token → done |
| 3 | "chulsoo 환불해줘" (2명) | users_list 후보 2 → **navigation(candidates)** → token → done |
| 4 | "오늘 최신 영화 3편 공지 초안" | tool_call(stats_recent or movie read) → tool_result → tool_call(notice_draft) → tool_result → **form_prefill** → token → done |
| 5 | "신작 공지 등록하고 알려줘" | notice_draft 실행 후 form_prefill — **Agent 는 저장 안 함** (narrator 가 고지) |
| 6 | 5-hop 초과 | max_hops 도달 → narrator fallback → token → done |
| 7 | Role=MODERATOR 가 stats_revenue 요청 | list_tools_for_role 필터로 해당 tool 미노출 → smart_fallback |

---

## 10. 마이그레이션 순서 (본 계획)

1. **Phase A** — Read tool 43 개로 확장
2. **Phase B** — Draft tool 10 개 (`drafts.py`) 추가
3. **Phase C** — Navigate tool 12 개 (`navigation.py`) 추가
4. **Phase D** — ReAct 그래프 개편 + SSE 2 이벤트 추가 + risk_gate 제거
5. **Phase E** — 프롬프트/Role 매트릭스/테스트 업데이트
6. **Phase F** — Admin Client 카드 + 대상 페이지 prefill 처리
7. **문서화** — CLAUDE.md 요약 갱신, docs/PROGRESS.md 이력 추가

---

## 11. 결정 이력

- 2026-04-23 · v2 출시 (HITL + Tier 3). 실제 쓰기 수행.
- 2026-04-23 · v3 재설계 (본 문서). **AI 쓰기 금지** 원칙으로 전환. Draft/Navigate 개념
  도입. 안전성과 UX 모두 개선.
