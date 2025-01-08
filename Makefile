.PHONY: pipeline pipeline-with-scraping api microservices stop-pipeline stop-services clean help

pipeline:
	@echo "Running data pipeline..."
	sudo docker compose -f docker-compose-pipeline.yml up -d

pipeline-with-scraping:
	@echo "Running data pipeline with data refresh..."
	sudo -E REFRESH_DATA=true docker compose -f docker-compose-pipeline.yml up -d

api:
	@echo "Starting API service..."
	sudo docker compose -f docker-compose-microservices.yml up -d api

microservices:
	@echo "Starting microservices..."
	sudo docker compose -f docker-compose-microservices.yml up -d clustering-service vectorizer-service images-service

stop-services:
	@echo "Stopping services..."
	sudo docker compose -f docker-compose-microservices.yml down

clean:
	@echo "Cleaning up Docker resources..."
	sudo docker compose -f docker-compose-pipeline.yml down -v
	sudo docker compose -f docker-compose-microservices.yml down -v

help:
	@echo "Available targets:"
	@echo " pipeline              - Run data pipeline"
	@echo " pipeline-with-scraping - Run data pipeline and refresh scraped data"
	@echo " api                 - Start only the API service"
	@echo " microservices       - Start only the microservices"
	@echo " stop-services       - Stop all services (API and microservices)"
	@echo " clean               - Remove all containers and resources"
