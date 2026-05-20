.PHONY: install dataset baseline lora qlora eval deploy clean lint test

install:
	pip install -r requirements.txt
	pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

dataset:
	python scripts/run_pipeline.py --step dataset

baseline:
	python scripts/run_pipeline.py --step baseline

lora:
	python scripts/run_pipeline.py --step lora

qlora:
	python scripts/run_pipeline.py --step qlora

eval:
	python scripts/run_pipeline.py --step eval

deploy:
	python scripts/run_pipeline.py --step deploy

all:
	python scripts/run_pipeline.py --step all

api:
	uvicorn src.inference.api:app --reload --port 8000

demo:
	streamlit run src/inference/demo.py

lint:
	ruff check src/ scripts/ tests/
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	rm -rf outputs/ mlruns/ __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
