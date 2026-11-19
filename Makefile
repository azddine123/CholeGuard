.PHONY: test models samples api ui demo
test:
	pytest -q && cd frontend && npm test
models:
	bash scripts/download_models.sh
samples:  ## requires the Endoscapes test frames: make samples FRAMES=/path/to/endoscapes/test
	python scripts/make_sample.py --frames-dir $(FRAMES) --video 170 --name cas_170
api:
	uvicorn backend.app:app --reload --port 8000
ui:
	cd frontend && npm run dev
demo: models
	docker compose up --build
