.PHONY: postgres mlflow pipeline pipeline-with-scraping pipeline-with-drop all clean help

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
	REFRESH_DATA=false DROP_TABLE=false docker-compose run --rm data-pipeline

pipeline-with-scraping:
	@echo "Running data pipeline with data refresh..."
	REFRESH_DATA=true DROP_TABLE=false docker-compose run --rm data-pipeline

pipeline-with-drop:
	@echo "Running data pipeline with table drop..."
	REFRESH_DATA=true DROP_TABLE=false docker-compose run --rm data-pipeline


pipeline-with-drop-and-scraping:
	@echo "Running data pipeline with table drop..."
	REFRESH_DATA=true DROP_TABLE=true docker-compose run --rm data-pipeline

stop:
	@echo "Stopping all containers..."
	docker-compose down

rebuild-postgres:
	@echo "Rebuilding PostgreSQL container..."
	docker-compose up -d --build postgres

rebuild-mlflow:
	@echo "Rebuilding MLflow container..."
	docker-compose up -d --build mlflow

rebuild-data-pipeline:
	@echo "Rebuilding data-pipeline container..."
	REFRESH_DATA=false DROP_TABLE=false docker-compose up -d --build data-pipeline

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
	@echo "  pipeline-with-drop - Run data pipeline with table drop"
	@echo "  pipeline-with-drop-and-scraping - Run data pipeline with table drop and refresh scraped data"
	@echo "  stop        - Stop all containers"
	@echo "  clean       - Remove all containers and resources"
	@echo "  rebuild-postgres    - Rebuild PostgreSQL container"
	@echo "  rebuild-mlflow      - Rebuild MLflow container"
	@echo "  rebuild-data-pipeline - Rebuild data-pipeline container"