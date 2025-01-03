# Makefile Documentation

This Makefile manages two main components:
1. Data Pipeline: Collects and processes book data
2. Services: Runs the API and microservices (clustering, vectorization, image processing)

## Environment Setup

Required environment variables in `.env`:
- Database configuration (POSTGRES_*)
- MLflow settings (MLFLOW_*)
- Network configuration (NETWORK_NAME)
- Optional: REFRESH_DATA for pipeline

## Makefile Targets

### Data Pipeline

- `pipeline`: Runs the data pipeline
  ```bash
  make pipeline
  ```
- `pipeline-with-scraping`: Runs pipeline with fresh data collection
  ```bash
  make pipeline-with-scraping
  ```
- `stop-pipeline`: Stops the pipeline containers
  ```bash
  make stop-pipeline
  ```

### Services

- `api`: Starts the FastAPI service
  ```bash
  make api
  ```
- `microservices`: Starts all microservices (clustering, vectorizer, images)
  ```bash
  make microservices
  ```
- `stop-services`: Stops all services
  ```bash
  make stop-services
  ```

### Maintenance

- `clean`: Removes all containers and volumes
  ```bash
  make clean
  ```
- `help`: Displays available commands

## Architecture

### Pipeline Components
- Data collection (Scrapy crawler)
- Data processing (compression, preparation)
- Database loading

### Microservices
- **API**: FastAPI service (port 9999)
- **Clustering**: K-means clustering service
- **Vectorizer**: Text vectorization service
- **Images**: Image processing service

## Docker Integration

Uses two separate compose files:
- `docker-compose-pipeline.yml`: Data pipeline
- `docker-compose-microservices.yml`: API and microservices

### Networking
- External network 'book-network' required
- Services connect to shared PostgreSQL and MLflow instances

### Volumes
- Data persistence through ./data volume mount
- Shared access to models and processed data

## Running the Full Stack

1. Start database and MLflow (external setup required)
2. Run pipeline: `make pipeline`
3. Start API: `make api`
4. Start microservices: `make microservices`

Use `make clean` for full cleanup when needed.