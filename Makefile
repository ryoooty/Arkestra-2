.PHONY: run api migrate sleep seed test

run:
python -m app.io.cli

api:
python -m scripts.serve

migrate:
python -m scripts.migrate

sleep:
python -m scripts.consolidate_sleep

seed:
python -m scripts.seed_rag

test:
pytest -q
