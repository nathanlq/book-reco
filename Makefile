.PHONY: pipeline pipeline-with-scraping clean help

pipeline:
	@echo "Running data pipeline..."
	sudo docker compose run --rm data-pipeline

pipeline-with-scraping:
	@echo "Running data pipeline with data refresh..."
	REFRESH_DATA=true sudo docker compose run --rm data-pipeline

clean:
	@echo "Cleaning up Docker resources..."
	sudo docker compose down -v

help:
	@echo "Available targets:"
	@echo "  pipeline               - Run data pipeline"
	@echo "  pipeline-with-scraping - Run data pipeline and refresh scraped data"
	@echo "  clean                  - Remove all containers and resources"
