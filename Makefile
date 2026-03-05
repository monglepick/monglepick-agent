.PHONY: dev infra infra-down test lint clean

# 개발 서버 실행
# OLLAMA_MAX_LOADED_MODELS=2: qwen3.5 + exaone-32b 두 모델을 GPU에 동시 유지하여 모델 스왑 방지
dev:
	OLLAMA_MAX_LOADED_MODELS=2 uv run uvicorn monglepick.main:app --reload --host 0.0.0.0 --port 8000

# Docker 인프라 실행
infra:
	docker compose up -d

# Docker 인프라 중지
infra-down:
	docker compose down

# 인프라 상태 확인
infra-status:
	docker compose ps

# 테스트
test:
	uv run pytest tests/ -v

# 린트
lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

# 린트 자동 수정
lint-fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

# 의존성 설치
install:
	uv sync

# 개발 의존성 포함 설치
install-dev:
	uv sync --extra dev

# 캐시/빌드 정리
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/

# 헬스체크
health:
	curl -s http://localhost:8000/health | python -m json.tool
