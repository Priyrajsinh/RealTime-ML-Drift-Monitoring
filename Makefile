.PHONY: install lint test serve train simulate dashboard docker-up docker-down

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	black src/ tests/ utils/ && isort src/ tests/ utils/ --profile black && flake8 src/ tests/ utils/ && bandit -r src/ -ll -ii

test:
	python -m pytest tests/ -v --tb=short --cov=src --cov-fail-under=70

serve:
	python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

train:
	python -m src.model.train

simulate:
	python -m src.monitoring.drift_simulator

dashboard:
	streamlit run src/dashboard/streamlit_app.py

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
