.PHONY: all setup run-scrapy compress clean start-postgres stop-postgres delete-postgres create-db start-mlflow test help

include .env
export $(shell sed 's/=.*//' .env)

all: setup run-scrapy compress

setup:
	@echo "Setting up the environment..."
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r $(REQUIREMENTS) && deactivate

run-scrapy: setup
	@echo "Running Scrapy spider..."
	if [ -f $(DATA_DIR)/raw_output.json ]; then \
		echo "Deleting existing raw_output.json..."; \
		rm $(DATA_DIR)/raw_output.json; \
	fi
	. $(VENV)/bin/activate && cd $(PROJECT_DIR) && $(SCRAPY) crawl furet -o ../../$(DATA_DIR)/raw_output.json && deactivate

compress: setup
	@echo "Compressing data..."
	. $(VENV)/bin/activate && $(PYTHON) -m $(SCRIPTS_DIR).compress && deactivate

prepare: compress
	@echo "Preparing data..."
	. $(VENV)/bin/activate && $(PYTHON) -m $(SCRIPTS_DIR).prepare && deactivate

load: prepare
	@echo "Loading data into PostgreSQL asynchronously..."
	. $(VENV)/bin/activate && $(PYTHON) -m $(SCRIPTS_DIR).loader && deactivate

clean:
	@echo "Cleaning up the data..."
	rm -f $(DATA_DIR)/*.parquet
	rm -rf $(DATA_DIR)/postgres

clean-venv:
	rm -rf $(VENV)

start-postgres:
	@echo "Starting PostgreSQL container..."
	if [ ! -d $(POSTGRES_VOLUME) ]; then \
		mkdir -p $(POSTGRES_VOLUME); \
	fi
	if [ "$$(docker ps -aq -f name=$(POSTGRES_CONTAINER_NAME))" ]; then \
		docker start $(POSTGRES_CONTAINER_NAME); \
	else \
		docker run --name $(POSTGRES_CONTAINER_NAME) \
			-e POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
			-v $(POSTGRES_VOLUME):/var/lib/postgresql/data \
			-p $(POSTGRES_PORT):5432 \
			-d pgvector/pgvector:pg17; \
		sleep 5; \
		docker exec -it $(POSTGRES_CONTAINER_NAME) psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;" \
		|| echo "pgvector extension setup failed. Ensure pgvector is installed or retry."; \
	fi

create-db: start-postgres
	@echo "Creating database and enabling pgvector extension..."
	docker exec -it $(POSTGRES_CONTAINER_NAME) psql -U postgres -c "CREATE DATABASE $(POSTGRES_DB);" || true
	docker exec -it $(POSTGRES_CONTAINER_NAME) psql -U postgres -d $(POSTGRES_DB) -c "CREATE EXTENSION IF NOT EXISTS vector;"

stop-postgres:
	@echo "Stopping PostgreSQL container..."
	if [ "$$(docker ps -aq -f name=$(POSTGRES_CONTAINER_NAME))" ]; then \
		docker stop $(POSTGRES_CONTAINER_NAME); \
	else \
		echo "PostgreSQL container is not running."; \
	fi

delete-postgres: stop-postgres
	@echo "Deleting PostgreSQL container..."
	if [ "$$(docker ps -aq -f name=$(POSTGRES_CONTAINER_NAME))" ]; then \
		docker rm $(POSTGRES_CONTAINER_NAME); \
	else \
		echo "PostgreSQL container does not exist."; \
	fi

test:
	@echo "Running end-to-end test..."
	./make-test.sh --enable-scraping --enable-venv-setup

start-mlflow: setup
	@echo "Starting MLflow server..."
	. $(VENV)/bin/activate && \
	mlflow server \
		--default-artifact-root $(MLFLOW_ARTIFACT_ROOT) \
		--host $(MLFLOW_HOST) --port $(MLFLOW_PORT) && \
	deactivate

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  all                  Run all tasks (setup, run-scrapy, compress)"
	@echo "  setup                Set up the environment"
	@echo "  run-scrapy           Run the Scrapy spider"
	@echo "  compress             Compress the data"
	@echo "  prepare              Prepare the data after compression"
	@echo "  load                 Load the prepared data into PostgreSQL asynchronously"
	@echo "  clean                Clean up the environment"
	@echo "  clean-venv           Delete the virtual environment"
	@echo "  start-postgres       Start the PostgreSQL container"
	@echo "  stop-postgres        Stop the PostgreSQL container"
	@echo "  delete-postgres      Delete the PostgreSQL container"
	@echo "  create-db            Create the PostgreSQL database if it does not exist"
	@echo "  start-mlflow         Start the MLflow server"
	@echo "  help                 Display this help message"
	@echo "  test                 Run end-to-end test for the entire data pipeline"
