.PHONY: setup ingest features train optimize evaluate clean lint test

# Environment Setup
setup:
	poetry install
	cp .env.example .env

# Data Pipeline
ingest:
	poetry run python main.py ingest

features:
	poetry run python main.py features

# ML Workflow
train:
	poetry run python main.py train --config model=base_lstm

optimize:
	poetry run python main.py optimize --swarm_size 20 --iterations 50

evaluate:
	poetry run python main.py evaluate --model_path models/checkpoints/best_model.pt

req-freeze:
	@echo "Updating requirements.txt (clean)..."
	pip3 freeze | grep -vE '^(pip|setuptools|wheel)' | sort > requirements.txt

req-no-versions:
	@echo "Updating requirements_no_versions.txt..."
	pip3 freeze | grep -vE '^(pip|setuptools|wheel)' | sed 's/==.*//' | sort | uniq > requirements_no_versions.txt

# Quality Control
lint:
	poetry run ruff check src/
	poetry run black --check src/

test:
	poetry run pytest tests/

clean:
	rm -rf logs/* data/interim/* data/processed/*