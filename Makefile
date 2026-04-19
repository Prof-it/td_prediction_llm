.PHONY: install test reproduce mine label train xai analysis clean-artifacts

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=src pytest -q tests/

reproduce: label train xai analysis

mine:
	python scripts/run_pipeline.py --stage mine

label:
	python scripts/run_pipeline.py --stage label

train:
	python scripts/run_pipeline.py --stage train --lopo

xai:
	python scripts/run_pipeline.py --stage xai --model lgbm

analysis:
	python scripts/run_pipeline.py --stage analysis

clean-artifacts:
	rm -rf artifacts/figures/* artifacts/results/* artifacts/models/*
