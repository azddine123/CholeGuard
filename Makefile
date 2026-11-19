.PHONY: test api ui demo
test:
	pytest -q
api:
	uvicorn backend.app:app --reload --port 8000
ui:
	cd frontend && npm run dev
demo:
	docker compose up --build
