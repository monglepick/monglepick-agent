"""
관리자 AI 에이전트 LangGraph 노드.

설계서: docs/관리자_AI에이전트_설계서.md §3.2 LangGraph 노드

Step 1 (2026-04-23):
- context_loader / intent_classifier / smalltalk_responder / response_formatter

Step 2 (2026-04-23, 추가):
- tool_selector   : Solar bind_tools 로 단일 tool_call 결정 (admin_role matrix 필터)
- tool_executor   : 레지스트리 handler 실행 → tool_results_cache 저장
- narrator        : Solar 가 tool_result 축약본을 한국어로 서술 (수치 생성 금지)

후속 Step 에서 추가될 노드:
- risk_gate       (Tier ≥ 2 → LangGraph interrupt)
- data_analyzer   (루프 종료 판단, pandas aggregate 필요 여부)
"""

from __future__ import annotations

import json
import time
import traceback
import uuid

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.types import interrupt

from monglepick.agents.admin_assistant.models import (
    AdminAssistantState,
    AdminIntent,
    ConfirmationPayload,
    ToolCall,
    normalize_admin_role,
)
from monglepick.api.admin_backend_client import AdminApiResult, summarize_for_llm
from monglepick.chains.admin_intent_chain import classify_admin_intent
from monglepick.chains.admin_tool_selector_chain import select_admin_tool
from monglepick.llm.factory import (
    get_conversation_llm,
    get_solar_api_llm,
    guarded_ainvoke,
)
from monglepick.prompts.admin_assistant import (
    NARRATOR_HUMAN_PROMPT,
    NARRATOR_SYSTEM_PROMPT,
    SMALLTALK_SYSTEM_PROMPT,
)
from monglepick.tools.admin_tools import (
    ADMIN_TOOL_REGISTRY,
    ToolContext,
    list_tools_for_role,
)

logger = structlog.get_logger()


# ============================================================
# 관리자 아닌 사용자 진입 차단 메시지
# ============================================================

_NOT_ADMIN_MESSAGE = (
    "관리자 권한이 필요한 기능이에요. 관리자 계정으로 로그인해주세요."
)


# ============================================================
# intent 별 placeholder 응답 (Step 1: tool 실행 미구현)
# ============================================================

# ============================================================
# Audit target 추론 (Step 6b)
# ============================================================

# tool 이름 → (Backend AdminAuditService 의 TARGET_* 상수 문자열, arguments key 후보 순서)
# Agent 쪽에서는 Backend 의 TARGET_USER / TARGET_PAYMENT / TARGET_SUBSCRIPTION 등을
# 하드코딩 상수로 맞춰둔다(AdminAuditService.java 의 값과 동일해야 한다).
_AUDIT_TARGET_HINTS: dict[str, tuple[str, tuple[str, ...]]] = {
    # users_write (Step 6b)
    "user_suspend": ("USER", ("userId",)),
    "user_unsuspend": ("USER", ("userId",)),  # Step 6c 예정
    "user_role_update": ("USER", ("userId",)),
    # points_write (Step 6b)
    "points_manual_adjust": ("USER", ("userId",)),
    # payment_write (Step 6c 예정)
    "payment_refund": ("PAYMENT", ("orderId",)),
    "ai_token_grant": ("USER", ("userId",)),
    # support_write / settings_write (Step 5a)
    "faq_create": ("FAQ", ("faqId",)),
    "banner_create": ("BANNER", ("bannerId",)),
}


def _infer_audit_target(tool_name: str, arguments: dict) -> tuple[str | None, str | None]:
    """
    tool 이름 + arguments 에서 감사 로그의 (targetType, targetId) 를 추론한다.

    매핑 규칙(§7.2):
    - 레지스트리에 hint 가 있으면 그걸 사용. arguments 에서 후보 키를 순서대로 검사해 첫
      non-empty 문자열을 targetId 로 반환.
    - hint 가 없으면 둘 다 None — Backend 가 targetType/targetId null 로 저장.
    """
    hint = _AUDIT_TARGET_HINTS.get(tool_name)
    if not hint:
        return (None, None)
    target_type, keys = hint
    for k in keys:
        value = arguments.get(k) if isinstance(arguments, dict) else None
        if isinstance(value, (str, int)) and str(value).strip():
            return (target_type, str(value))
    return (target_type, None)


_PLACEHOLDER_MESSAGES: dict[str, str] = {
    # Step 4 부터 query 도 tool_selector 경로를 경유한다. 이 메시지는 tool_selector 가
    # "적합한 도구를 못 찾은" 경우의 fallback 으로만 쓰인다. "개발 중" 문구는 제거.
    "query": (
        "요청하신 조회에 적합한 도구를 찾지 못했어요. "
        "사용자/결제/게시글/리뷰/티켓 등의 조회는 지원되지만, 구체 대상(userId·orderId 등)이 "
        "발화에 포함되어야 하는 경우가 많아요. 질문을 조금 더 구체적으로 말씀해 주세요."
    ),
    "action": (
        "🛠️ 쓰기 작업(공지·배너·FAQ·퀴즈 CRUD, 계정 정지/환불 등) 은 "
        "아직 구현되지 않았어요. 관리자 승인 플로우(HITL) 와 함께 다음 단계에서 추가될 예정입니다."
    ),
    # Step 2 에서 stats 는 실제 tool 경로로 분기. 이 placeholder 는
    # "stats intent 이지만 적합한 tool 이 없거나 tool 실행 실패" fallback 으로 쓰인다.
    "stats": (
        "요청하신 통계에 적합한 도구를 찾지 못했어요. "
        "다음 Step 에서 더 많은 통계 도구(추천 성능·포인트 경제·참여도 등) 가 추가됩니다."
    ),
    "report": (
        "🛠️ 보고서 생성은 Phase 4 예정 기능이에요. "
        "지금은 통계 조회가 연결된 후 주간/월간 템플릿으로 확장됩니다."
    ),
    "sql": (
        "이 에이전트는 자유 SQL 실행을 지원하지 않아요. "
        "기존 통계 화면이나 미리 준비된 조회를 이용해 주세요."
    ),
}


# ============================================================
# Node 1 — context_loader
# ============================================================

async def context_loader(state: AdminAssistantState) -> dict:
    """
    요청 수명 시작점. admin_id / admin_role 검증 + 빈 필드 기본값 채움.

    - admin_role 이 비어있으면 (정규화 실패) 이후 노드가 placeholder 로 우회해
      안내 메시지만 내려가도록 한다.
    - history 는 Step 1 에서 매 요청 빈 배열로 초기화 (세션 저장소 미도입).
    - tool_results_cache / analysis_outputs / chart_payloads / table_payloads
      모두 빈 기본값으로 초기화 해 downstream 노드가 접근해도 KeyError 가 나지 않게.
    """
    start = time.perf_counter()
    admin_id = state.get("admin_id", "") or ""
    admin_role = normalize_admin_role(state.get("admin_role", ""))

    if not admin_id:
        logger.warning("admin_assistant_missing_admin_id")
    if not admin_role:
        logger.warning(
            "admin_assistant_missing_or_invalid_role",
            raw_role=state.get("admin_role"),
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "admin_context_loaded",
        admin_id=admin_id or "(unknown)",
        admin_role=admin_role or "(blank)",
        session_id=state.get("session_id", ""),
        elapsed_ms=round(elapsed_ms, 1),
    )
    return {
        "admin_id": admin_id,
        "admin_role": admin_role,
        "history": state.get("history", []) or [],
        "candidate_tools": state.get("candidate_tools", []) or [],
        "tool_results_cache": state.get("tool_results_cache", {}) or {},
        "analysis_outputs": state.get("analysis_outputs", []) or [],
        "chart_payloads": state.get("chart_payloads", []) or [],
        "table_payloads": state.get("table_payloads", []) or [],
        "awaiting_confirmation": False,
        "iteration_count": 0,
        "error": None,
    }


# ============================================================
# Node 2 — intent_classifier
# ============================================================

async def intent_classifier(state: AdminAssistantState) -> dict:
    """
    발화를 AdminIntent(kind/confidence/reason) 로 분류한다.

    관리자 권한이 없는 상태(admin_role="") 로 진입하면 LLM 호출을 건너뛰고
    smalltalk 로 고정해 불필요한 Solar API 비용을 절감한다.
    """
    admin_role = state.get("admin_role", "") or ""
    admin_id = state.get("admin_id", "") or ""
    user_message = state.get("user_message", "") or ""

    # 비관리자 → 분류 생략, 이후 response_formatter 가 _NOT_ADMIN_MESSAGE 로 응답
    if not admin_role:
        return {
            "intent": AdminIntent(
                kind="smalltalk",
                confidence=0.0,
                reason="no_admin_role",
            ),
        }

    # 빈 발화 방어 — 체인을 태우지 않고 smalltalk 로 처리
    if not user_message.strip():
        return {
            "intent": AdminIntent(
                kind="smalltalk",
                confidence=0.0,
                reason="empty_user_message",
            ),
        }

    intent = await classify_admin_intent(
        user_message=user_message,
        admin_role=admin_role,
        request_id=f"admin:{admin_id[:8]}" if admin_id else "admin:anon",
    )
    return {"intent": intent}


# ============================================================
# Node 3 — smalltalk_responder
# ============================================================

async def smalltalk_responder(state: AdminAssistantState) -> dict:
    """
    smalltalk intent 응답 생성.

    hybrid 모드: vLLM EXAONE 1.2B 또는 Ollama 몽글이(빠른 응답 체인).
    api_only: Solar API.
    local_only: Ollama EXAONE 32B.

    get_conversation_llm() 이 LLM_MODE 를 내부 분기한다.
    수치/유저 정보는 이 응답에서 만들지 않도록 프롬프트로 강제한다.
    """
    start = time.perf_counter()
    user_message = state.get("user_message", "") or ""
    admin_id = state.get("admin_id", "")

    try:
        llm = get_conversation_llm()
        messages = [
            SystemMessage(content=SMALLTALK_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response_obj = await guarded_ainvoke(
            llm,
            messages,
            model="admin_smalltalk",
            request_id=f"admin:{admin_id[:8]}" if admin_id else "admin:anon",
        )
        text = getattr(response_obj, "content", None) or str(response_obj)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "admin_smalltalk_generated",
            length=len(text),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"response_text": text.strip()}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "admin_smalltalk_failed_fallback",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        # 에러 전파 금지 — 고정 안내로 폴백
        return {
            "response_text": (
                "안녕하세요! 관리자 어시스턴트예요. "
                "통계 조회·리소스 조회·배너/공지/FAQ 등록 같은 걸 자연어로 요청해주시면 도와드릴게요."
            ),
        }


# ============================================================
# Node 4 — response_formatter
# ============================================================

async def response_formatter(state: AdminAssistantState) -> dict:
    """
    최종 응답 조립.

    우선순위:
    1) admin_role 이 비어있으면 → _NOT_ADMIN_MESSAGE
    2) smalltalk_responder 가 채운 response_text 가 있으면 → 그대로 사용
    3) 그 외 intent 는 Step 1 에서 placeholder 안내

    이 단계는 수치를 생성/가공하지 않는다 (§6.1 "LLM 은 숫자를 만들지 않는다").
    """
    admin_role = state.get("admin_role", "") or ""
    intent = state.get("intent")
    already_composed = state.get("response_text", "") or ""

    # 1) 비관리자 차단
    if not admin_role:
        return {"response_text": _NOT_ADMIN_MESSAGE}

    # 2) smalltalk 응답이 이미 채워진 경우
    if already_composed:
        return {"response_text": already_composed}

    # 3) 나머지 intent 는 Step 1 placeholder
    intent_kind = intent.kind if isinstance(intent, AdminIntent) else "smalltalk"
    placeholder = _PLACEHOLDER_MESSAGES.get(
        intent_kind,
        "요청을 처리하지 못했어요. 다시 한 번 말씀해주시겠어요?",
    )
    return {"response_text": placeholder}


# ============================================================
# Step 2 Node 5 — tool_selector
# ============================================================

async def tool_selector(state: AdminAssistantState) -> dict:
    """
    stats/query intent 에서 Solar bind_tools 로 단일 tool-call 을 선택한다.

    - admin_role 이 없거나 레지스트리 필터 결과가 비어있으면 pending_tool_call=None.
    - LLM 응답에 tool_call 이 없으면 pending_tool_call=None → response_formatter 로 직행.
    - 성공 시 ToolCall 객체를 state.pending_tool_call 에 저장. tier 는 레지스트리에서 주입.
    """
    admin_role = state.get("admin_role", "") or ""
    admin_id = state.get("admin_id", "") or ""
    user_message = state.get("user_message", "") or ""
    intent = state.get("intent")
    intent_kind = intent.kind if isinstance(intent, AdminIntent) else "unknown"

    if not admin_role or not user_message.strip():
        return {"pending_tool_call": None}

    selected = await select_admin_tool(
        user_message=user_message,
        admin_role=admin_role,
        intent_kind=intent_kind,
        request_id=f"admin:{admin_id[:8]}" if admin_id else "admin:anon",
    )

    if selected is None:
        return {"pending_tool_call": None}

    # 레지스트리에서 tier 주입 (selector 체인은 tier 를 반환하지 않음)
    spec = ADMIN_TOOL_REGISTRY.get(selected.name)
    tier = spec.tier if spec is not None else 0

    call = ToolCall(
        tool_name=selected.name,
        arguments=selected.arguments,
        tier=tier,
        rationale=selected.rationale,
    )
    return {"pending_tool_call": call}


# ============================================================
# Step 2 Node 6 — tool_executor
# ============================================================

async def tool_executor(state: AdminAssistantState) -> dict:
    """
    pending_tool_call 을 실제로 실행한다.

    - 레지스트리에서 ToolSpec 조회 후 args_schema 로 arguments 검증.
    - admin_role 이 허용 목록에 포함되지 않으면 실행 거부 (설계서 §4.2 Role matrix).
    - Tier 2/3 는 Step 2 범위 밖 — 현재는 Tier 0/1 만 실행되도록 가드.
    - 결과는 tool_results_cache[ref_id] 에 저장하고, ref_id 를 state.latest_tool_ref_id 에 기록.
    - 실패(ok=False) 결과도 그대로 캐시한다 — narrator 가 error 메시지를 정확히 서술하도록.
    """
    call: ToolCall | None = state.get("pending_tool_call")
    admin_role = state.get("admin_role", "") or ""
    cache = dict(state.get("tool_results_cache", {}) or {})

    if call is None:
        return {"tool_results_cache": cache, "latest_tool_ref_id": ""}

    spec = ADMIN_TOOL_REGISTRY.get(call.tool_name)
    if spec is None:
        logger.warning("admin_tool_executor_unknown_tool", tool_name=call.tool_name)
        return {"tool_results_cache": cache, "latest_tool_ref_id": ""}

    # Step 5a: Tier 2/3 하드 가드는 제거됐다 (risk_gate 에서 사용자 승인을 받은 뒤에만
    # tool_executor 가 호출되는 구조). 다만 Tier 4(SQL 샌드박스) 는 설계상 영구 미지원이라
    # 여전히 차단한다.
    if spec.tier >= 4:
        logger.info(
            "admin_tool_executor_tier4_blocked",
            tool_name=call.tool_name,
            tier=spec.tier,
            reason="tier=4 (SQL 샌드박스) 는 v2 영구 미지원",
        )
        return {"tool_results_cache": cache, "latest_tool_ref_id": ""}

    # Role matrix 재검증 (selector 에서 한 번 걸러도 실행 직전 이중 방어)
    allowed = {s.name for s in list_tools_for_role(admin_role)}
    if call.tool_name not in allowed:
        logger.warning(
            "admin_tool_executor_role_denied",
            tool_name=call.tool_name,
            admin_role=admin_role,
        )
        return {"tool_results_cache": cache, "latest_tool_ref_id": ""}

    # args 검증 — Pydantic 으로 타입/기본값 적용
    try:
        validated = spec.args_schema.model_validate(call.arguments)
        args_dict = validated.model_dump()
    except Exception as e:
        logger.warning(
            "admin_tool_executor_args_validation_failed",
            tool_name=call.tool_name,
            arguments=call.arguments,
            error=str(e),
        )
        failed = AdminApiResult(
            ok=False, status_code=0, error=f"args_validation:{type(e).__name__}",
        )
        ref_id = f"tr_{uuid.uuid4().hex[:10]}"
        cache[ref_id] = failed
        return {"tool_results_cache": cache, "latest_tool_ref_id": ref_id}

    # 실행
    ctx = ToolContext(
        admin_jwt=state.get("admin_jwt", "") or "",
        admin_role=admin_role,
        admin_id=state.get("admin_id", "") or "",
        session_id=state.get("session_id", "") or "",
        invocation_id=f"admin:{state.get('session_id', '')[:12]}",
    )
    start = time.perf_counter()
    try:
        result = await spec.handler(ctx=ctx, **args_dict)
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.warning(
            "admin_tool_executor_handler_crashed",
            tool_name=call.tool_name,
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=elapsed_ms,
            stack_trace=traceback.format_exc(),
        )
        result = AdminApiResult(
            ok=False, status_code=0,
            error=f"handler_crash:{type(e).__name__}",
            latency_ms=elapsed_ms,
        )

    ref_id = f"tr_{uuid.uuid4().hex[:10]}"
    cache[ref_id] = result
    logger.info(
        "admin_tool_executed",
        tool_name=call.tool_name,
        tier=spec.tier,
        ok=result.ok,
        status_code=result.status_code,
        latency_ms=result.latency_ms,
        ref_id=ref_id,
    )

    # ── Step 6a/6b: Tier≥2 쓰기 실행은 감사 로그에 자동 기록 ──
    # Tier 0/1 읽기는 볼륨 폭증 방지를 위해 감사 미기록(§7.2). Tier 2/3 은 성공/실패 모두
    # 한 건 기록 — "AGENT_EXECUTED" actionType 으로, 실제 쓰기가 발생시킨 도메인 감사 로그
    # (POINT_MANUAL 등) 와 별도 레코드로 남아 양방향 추적 가능(§7.1). 감사 기록 실패는
    # graceful — 원 작업 응답은 그대로 narrator 로 흘러간다.
    #
    # Step 6b 추가: Tier 3 tool 이 `AdminApiResult.before_data`/`after_data` 에 담아 올린
    # 리소스 스냅샷을 audit 의 beforeData/afterData 필드로 그대로 전달한다. 또한
    # arguments 의 `userId`/`orderId` 를 target_id 로 유추해 감사 조회 시 특정 리소스로
    # 필터링하기 쉽게 한다(targetType 은 tool 이름에서 간단 매핑).
    if spec.tier >= 2:
        from monglepick.api.admin_audit_client import log_agent_action
        target_type, target_id = _infer_audit_target(call.tool_name, call.arguments)
        try:
            await log_agent_action(
                admin_jwt=state.get("admin_jwt", "") or "",
                tool_name=call.tool_name,
                arguments=call.arguments,
                ok=result.ok,
                user_prompt=state.get("user_message", "") or "",
                target_type=target_type,
                target_id=target_id,
                before_data=result.before_data,
                after_data=result.after_data,
                error=result.error if not result.ok else "",
                invocation_id=ctx.invocation_id,
            )
        except Exception as audit_err:
            # log_agent_action 내부에서 이미 graceful 처리하지만 이중 안전망.
            logger.warning(
                "admin_tool_audit_outer_error",
                tool_name=call.tool_name,
                error=str(audit_err),
            )

    return {"tool_results_cache": cache, "latest_tool_ref_id": ref_id}


# ============================================================
# Step 2 Node 7 — narrator
# ============================================================

async def narrator(state: AdminAssistantState) -> dict:
    """
    tool_result 축약본을 Solar Pro 로 한국어 서술.

    - latest_tool_ref_id 가 없으면 아무것도 하지 않음 (response_formatter 가 placeholder 사용).
    - 축약본은 summarize_for_llm 으로 생성 — raw rows 가 LLM 컨텍스트에 들어가지 않음 (§6.1).
    - 수치 생성 금지 규칙은 프롬프트로 강제.
    - LLM 실패 시 "조회는 됐지만 해석이 실패했다" 로 안내 + tool 원시값을 축약해 인용.
    """
    ref_id = state.get("latest_tool_ref_id", "") or ""
    cache = state.get("tool_results_cache", {}) or {}
    call: ToolCall | None = state.get("pending_tool_call")

    if not ref_id or ref_id not in cache:
        # tool_selector 가 None 을 낸 경우 — response_formatter 에서 placeholder 처리
        return {}

    result = cache[ref_id]
    # result 는 AdminApiResult 인스턴스 또는 dict (테스트용). 둘 다 허용.
    if isinstance(result, AdminApiResult):
        summarized = summarize_for_llm(result)
    elif isinstance(result, dict):
        summarized = result
    else:
        summarized = {"ok": False, "error": f"unexpected_result_type:{type(result).__name__}"}

    tool_name = call.tool_name if call else "(unknown)"
    args_repr = json.dumps(call.arguments, ensure_ascii=False) if call else "{}"
    result_json = json.dumps(summarized, ensure_ascii=False, default=str)

    start = time.perf_counter()
    try:
        llm = get_solar_api_llm(temperature=0.2)
        system = SystemMessage(content=NARRATOR_SYSTEM_PROMPT)
        human = HumanMessage(content=NARRATOR_HUMAN_PROMPT.format(
            user_message=state.get("user_message", ""),
            tool_name=tool_name,
            tool_args=args_repr,
            tool_result_json=result_json,
        ))
        response = await guarded_ainvoke(
            llm, [system, human],
            model="solar_api",
            request_id=f"admin_narrator:{ref_id}",
        )
        text = getattr(response, "content", None) or str(response)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "admin_narrator_generated",
            tool_name=tool_name,
            length=len(text),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"response_text": text.strip()}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "admin_narrator_failed",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
        )
        # narrator 실패 fallback — 축약본 원문을 그대로 인용
        if isinstance(result, AdminApiResult) and result.ok:
            return {
                "response_text": (
                    f"{tool_name} 호출은 성공했지만 결과 해석에 실패했어요. "
                    f"원시 데이터를 그대로 전달드려요:\n\n```\n{result_json[:1500]}\n```\n\n"
                    f"[출처: {tool_name}]"
                ),
            }
        return {
            "response_text": (
                f"요청을 처리하는 중 문제가 발생했어요 ({type(e).__name__}). 잠시 후 다시 시도해주세요."
            ),
        }


# ============================================================
# Step 5a Node — risk_gate (HITL 승인 게이트)
# ============================================================

async def risk_gate(state: AdminAssistantState) -> dict:
    """
    Tier≥2 쓰기 작업이 실행되기 전 사용자 승인을 받는 관문.

    흐름:
    - pending_tool_call.tier < 2 → 통과 (아무 것도 안 하고 return {}). tool_executor 로 직행.
    - pending_tool_call.tier >= 2 → LangGraph `interrupt(payload)` 호출.
      · 최초 실행 시: 그래프가 여기서 멈추고, astream 루프가 종료된다. SSE 레이어가
        state.confirmation_payload 를 보고 `confirmation_required` 이벤트를 발행한다.
      · Client 가 `/resume` 으로 ConfirmationDecision 을 보내면 LangGraph 가 이 지점부터
        재개하는데, interrupt() 는 이번엔 decision 값을 반환한다.
    - decision.decision == 'approve' → tool_executor 진행 (state 그대로 유지).
    - decision.decision == 'reject' → pending_tool_call=None, response_text 에 거절 안내를
      직접 써서 response_formatter 가 그대로 내보내게 한다.

    주의: Tier 4(SQL) 는 레지스트리에 등록이 금지되어 있지만, 혹시 등록돼도 tool_executor 가
    재차 차단한다.
    """
    call: ToolCall | None = state.get("pending_tool_call")
    if call is None:
        return {}

    # Tier 0/1 은 HITL 불필요 — 바로 통과
    if call.tier < 2:
        return {}

    # 레지스트리에서 rationale/설명/confirm_keyword 를 끌어와 plan_summary 를 구성
    spec = ADMIN_TOOL_REGISTRY.get(call.tool_name)
    tool_desc = spec.description[:120] if spec else call.tool_name
    # Step 6b: Tier 3 tool 은 spec.confirm_keyword 를 payload.required_keyword 로 전달.
    # Admin UI 가 이 키워드를 사용자에게 타이핑시켜 오조작 2중 방어.
    keyword = (spec.confirm_keyword if spec and spec.confirm_keyword else "") or ""
    payload = ConfirmationPayload(
        tool_name=call.tool_name,
        arguments=call.arguments,
        tier=call.tier,
        plan_summary=tool_desc,
        rationale=call.rationale,
        required_keyword=keyword,
    )

    logger.info(
        "admin_risk_gate_interrupt",
        tool_name=call.tool_name,
        tier=call.tier,
        session_id=state.get("session_id", ""),
    )

    # interrupt() 는 최초 호출 시 그래프를 여기서 멈추고, resume 이 들어오면 그 값을 반환한다.
    decision_raw = interrupt(payload.model_dump())

    # ── 재개 경로: decision 값 파싱 ──
    # decision_raw 는 ConfirmationDecision.model_dump() 형태로 오는 걸 기대하지만,
    # 방어적으로 dict / 문자열 / None 모든 경우를 처리한다.
    if isinstance(decision_raw, dict):
        decision = str(decision_raw.get("decision", "")).lower()
        comment = str(decision_raw.get("comment", ""))
    elif isinstance(decision_raw, str):
        decision = decision_raw.lower()
        comment = ""
    else:
        decision = ""
        comment = ""

    if decision == "approve":
        logger.info(
            "admin_risk_gate_approved",
            tool_name=call.tool_name,
            session_id=state.get("session_id", ""),
        )
        return {
            "awaiting_confirmation": False,
            "confirmation_decision": {"decision": "approve", "comment": comment},
        }

    # reject 또는 알 수 없는 응답 → 안전하게 거절 처리
    logger.info(
        "admin_risk_gate_rejected",
        tool_name=call.tool_name,
        raw_decision=str(decision_raw)[:100],
        session_id=state.get("session_id", ""),
    )
    reject_msg = (
        "요청하신 쓰기 작업을 실행하지 않았어요. "
        f"(거부된 작업: `{call.tool_name}`)"
    )
    if comment:
        reject_msg += f" 메모: {comment}"
    return {
        "awaiting_confirmation": False,
        "confirmation_decision": {"decision": "reject", "comment": comment},
        # pending_tool_call 을 None 으로 비워 route_after_risk_gate 가 실행 경로 차단 판단
        "pending_tool_call": None,
        "response_text": reject_msg,
    }
