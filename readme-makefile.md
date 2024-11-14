# Makefile Documentation

The Makefile automates setting up the project environment, running data collection, preparing data, loading it into PostgreSQL, and managing additional services like MLflow for model tracking and PostgreSQL for database storage.

## Setup and Usage

Before running the Makefile, ensure that all environment variables in .env are set up. Variables used include:

- `PYTHON`: The path to the Python interpreter (e.g., python3).
- `VENV`: The path for the Python virtual environment directory.
- `REQUIREMENTS`: The requirements file path.
- `DATA_DIR`: Directory for data files.
- `PROJECT_DIR`: The main project directory.
- `SCRIPTS_DIR`: Directory for Python scripts.
- `POSTGRES_VOLUME`: Directory for PostgreSQL data.
- `POSTGRES_CONTAINER_NAME`, `POSTGRES_DB`, `POSTGRES_PORT`, `POSTGRES_PASSWORD`: PostgreSQL setup details.
- `MLFLOW_ARTIFACT_ROOT`: Root directory for MLflow artifacts.
- `MLFLOW_HOST`, `MLFLOW_PORT`: Host and port for the MLflow server.

### Makefile Targets

Here is a summary of the Makefile targets. Each command can be run individually using `make <target>`.

- `all`: Run all tasks (setup, run-scrapy, compress).
- `setup`: Set up the environment by creating a virtual environment and installing dependencies.
- `run-scrapy`: Run the Scrapy spider to collect data.
- `compress`: Compress the collected data.
- `prepare`: Prepare the data after compression.
- `load`: Load the prepared data into PostgreSQL asynchronously.
- `clean`: Clean up the environment by removing the virtual environment and data files.
- `start-postgres`: Start the PostgreSQL container.
- `stop-postgres`: Stop the PostgreSQL container.
- `delete-postgres`: Delete the PostgreSQL container.
- `create-db`: Create the PostgreSQL database if it does not exist.
- `start-mlflow`: Starts the MLflow server for model tracking.
- `help`: Display the help message with available targets.
- `test`: Run end-to-end test for the entire data pipeline.


### Usage

To use the Makefile, run the following commands in the terminal:

- `make all`: Run all tasks.
- `make setup`: Set up the environment.
- `make run-scrapy`: Run the Scrapy spider.
- `make compress`: Compress the data.
- `make prepare`: Prepare the data.
- `make load`: Load the data into PostgreSQL.
- `make clean`: Clean up the environment.
- `make start-postgres`: Start the PostgreSQL container.
- `make stop-postgres`: Stop the PostgreSQL container.
- `make delete-postgres`: Delete the PostgreSQL container.
- `make create-db`: Create the PostgreSQL database.
- `make help`: Display the help message.
- `make test`: Run end-to-end test for the entire data pipeline.

## Running Tests

The project includes an automated end-to-end test script, **make-test.sh**, which tests the full data pipeline, including the environment setup, data scraping, processing, and loading into PostgreSQL.

### Running make-test.sh

To run the end-to-end test, use the command `make test`. This command initiates make-test.sh with two optional flags:

- `--enable-scraping`: Enables the Scrapy data scraping step.
- `--enable-venv-setup`: Enables environment setup to create and activate a virtual environment before running tests.

### Test Workflow

The `make-test.sh` script performs the following steps:

- **Setup**: Activates the virtual environment if specified by `--enable-venv-setup`.
- **Start PostgreSQL**: Checks if the PostgreSQL container is running, and starts or creates it if needed.
- **Database Verification**: Tests PostgreSQL connectivity and ensures the database is correctly initialized.
- **Scraping**: Runs the Scrapy spider if `--enable-scraping` is specified.
- **Data Processing and Loading**: Executes the compression, preparation, and loading steps.
- **Data Verification**: Confirms data has been successfully loaded into the database by verifying table counts and content.
- **Cleanup**: Stops and removes the PostgreSQL container and cleans up temporary data

### Notes

- Ensure Docker is installed and running as make-test.sh depends on Docker to manage the PostgreSQL container.
- If using MLflow, ensure the necessary MLflow configuration is specified in `.env`. The server has to be started.

This test script is intended for development and validation purposes, enabling you to quickly check the end-to-end functionality of the data pipeline.