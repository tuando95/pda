.PHONY: install test clean train evaluate

install:
	pip install -e .
	pip install -r requirements.txt

test:
	pytest tests/

test-coverage:
	pytest tests/ --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

train-baseline:
	python train.py --config config.yml

train-pda:
	python train.py --config config.yml --override pda.enable=true

evaluate-all:
	python evaluate.py --checkpoint logs/*/model_best.pth --all

run-ablations:
	python experiments/ablation_timestep.py
	python experiments/ablation_reverse_steps.py
	python experiments/ablation_components.py

docker-build:
	docker build -t pda:latest .

docker-run:
	docker run --gpus all -v $(PWD):/workspace pda:latest