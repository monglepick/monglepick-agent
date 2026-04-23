# =========================================
# 몽글픽 AI Agent Docker 이미지
# =========================================
# 멀티스테이지 빌드: uv 의존성 설치 → 런타임 이미지
# Python 3.12 slim 기반, 비루트 사용자로 실행
#
# 빌드: docker build -t monglepick-agent .
# 실행: docker run -p 8000:8000 monglepick-agent

# --- 1단계: 의존성 설치 ---
FROM python:3.12-slim AS builder

WORKDIR /app

# uv 설치 (빠른 패키지 매니저)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 의존성 파일 복사 (캐시 레이어 활용)
COPY pyproject.toml uv.lock ./

# Linux 환경용 의존성 설치 (macOS 전용 환경 설정 무시)
# --frozen 대신 --no-dev로 설치하여 Linux 플랫폼에 맞게 해석
RUN uv sync --no-dev --no-editable

# --- 2단계: 런타임 이미지 ---
FROM python:3.12-slim

WORKDIR /app

# curl: 헬스체크용
# tzdata: Asia/Seoul 시간대 지원 (QA #162/#177 근본 해결). debian-slim 은 tzdata 미포함 상태로
#   `datetime.now()` 가 UTC 로 떨어진다. tzdata 설치 + /etc/localtime 심볼릭 링크로 확실히 고정.
RUN apt-get update && apt-get install -y --no-install-recommends curl tzdata \
    && ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
    && echo "Asia/Seoul" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 기본 타임존 — docker-compose 에서 TZ 로 오버라이드 가능.
ENV TZ=Asia/Seoul

# 빌드 단계에서 생성된 가상환경 복사
COPY --from=builder /app/.venv /app/.venv

# 소스 코드 복사
COPY src/ src/

# PATH에 venv 추가, PYTHONPATH 설정
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# 비루트 사용자 생성 (보안)
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# 헬스체크용 포트 노출
EXPOSE 8000

# uvicorn으로 FastAPI 앱 실행 (workers=2)
CMD ["uvicorn", "monglepick.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
