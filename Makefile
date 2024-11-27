.PHONY: postgres mlflow pipeline all clean help

include .env
export

all: postgres mlflow pipeline

postgres:
	@echo "Starting PostgreSQL container..."
	docker-compose up -d postgres

mlflow:
	@echo "Starting MLflow container..."
	docker-compose up -d mlflow

pipeline:
	@echo "Running data pipeline..."
	docker-compose run --rm data-pipeline

pipeline-with-scraping:
	@echo "Running data pipeline with data refresh..."
	REFRESH_DATA=true docker-compose run --rm data-pipeline

stop:
	@echo "Stopping all containers..."
	docker-compose down

clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f

help:
	@echo "Available targets:"
	@echo "  postgres    - Start PostgreSQL container"
	@echo "  mlflow      - Start MLflow container"
	@echo "  pipeline    - Run data pipeline"
	@echo "  pipeline-with-scraping - Run data pipeline and refresh scraped data"
	@echo "  stop        - Stop all containers"
	@echo "  clean       - Remove all containers and resources"