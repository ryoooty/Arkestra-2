.PHONY: run migrate sleep seed test docker-build docker-run

run:
	python -m app.io.cli

migrate:
	python -m scripts.migrate

sleep:
	python -m scripts.consolidate_sleep

seed:
	python -m scripts.seed_rag

test:
	pytest -q

docker-build:
	docker build -t arkestra:latest .

docker-run:
	docker run --rm -p 8080:8080 arkestra:latest
