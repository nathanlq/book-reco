# README

This repository contains a Makefile and a test script (make-test.sh) to automate the setup, execution, and testing of a data pipeline.

## Makefile

The Makefile defines various targets to manage the environment, run tasks, and handle PostgreSQL operations.

### Targets

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
- `run-microservice`: Run the microservice.
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
- `make run-microservice`: Run the microservice.
- `make help`: Display the help message.
- `make test`: Run end-to-end test for the entire data pipeline.

## make-test.sh

The `make-test.sh` script automates the end-to-end testing of the data pipeline. It performs the following steps:

1. Set up the environment (optional).
2. Start the PostgreSQL container.
3. Check if the PostgreSQL container is running.
4. Test the PostgreSQL connection.
5. Create the PostgreSQL database.
6. Run the Scrapy spider (optional).
7. Compress the data.
8. Prepare the data.
9. Load the data into PostgreSQL.
10. Verify the data in PostgreSQL.
11. Clean up the environment.

### Flags

- `--enable-scraping`: Enable the Scrapy spider step.
- `--enable-venv-setup`: Enable the virtual environment setup step.

### Usage

To run the test script, use the following commands in the terminal:

- `./make-test.sh`: Run the test script without scraping and virtual environment setup.
- `./make-test.sh --enable-scraping`: Run the test script with scraping enabled.
- `./make-test.sh --enable-venv-setup`: Run the test script with virtual environment setup enabled.
- `./make-test.sh --enable-scraping --enable-venv-setup`: Run the test script with both scraping and virtual environment setup enabled.

### Cleanup

The script includes a cleanup function that is triggered on exit. It stops and deletes the PostgreSQL container and cleans up the environment.

## Environment Variables

The Makefile and test script rely on environment variables defined in the `.env` file. Ensure that the `.env` file is properly configured with the required variables.

## Dependencies

- Python
- Docker
- PostgreSQL
- Scrapy
- Other dependencies specified in the `requirements.txt` file.

## Ressources

- Documentation [pgvector](https://github.com/pgvector/pgvector)