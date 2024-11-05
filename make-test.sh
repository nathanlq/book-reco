#!/bin/bash

set -e

source .env

ENABLE_SCRAPING=false
ENABLE_VENV_SETUP=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --enable-scraping) ENABLE_SCRAPING=true ;;
        --enable-venv-setup) ENABLE_VENV_SETUP=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cleanup() {
    echo "An error occurred. Performing cleanup..."
    run_make_command "stop-postgres" || true
    run_make_command "delete-postgres" || true
    run_make_command "clean" || true
    echo "Cleanup completed."
}

trap cleanup EXIT

run_make_command() {
    local command=$1
    echo "Running make $command..."
    make $command
    echo "Finished make $command."
    echo ""
}

check_postgres_container() {
    echo "Checking if PostgreSQL container is running..."
    if [ "$(docker ps -q -f name=${POSTGRES_CONTAINER_NAME})" ]; then
        echo "PostgreSQL container is running."
    else
        echo "PostgreSQL container is not running."
        exit 1
    fi
    echo ""
}

test_postgres_connection() {
    echo "Testing PostgreSQL connection..."
    local retries=5
    local count=0
    local success=false

    while [ $count -lt $retries ]; do
        if docker exec -it ${POSTGRES_CONTAINER_NAME} psql -U postgres -c "\conninfo" >/dev/null 2>&1; then
            echo "PostgreSQL connection successful."
            success=true
            break
        else
            echo "PostgreSQL connection failed. Retrying in 5 seconds..."
            sleep 5
            count=$((count + 1))
        fi
    done

    if [ "$success" = false ]; then
        echo "PostgreSQL connection failed after $retries attempts."
        exit 1
    fi
    echo "Finished testing PostgreSQL connection."
    echo ""
}

verify_postgres_data() {
    echo "Verifying data in PostgreSQL..."
    docker exec -it ${POSTGRES_CONTAINER_NAME} psql -U postgres -d ${POSTGRES_DB} -c "\dt" || exit 1
    docker exec -it ${POSTGRES_CONTAINER_NAME} psql -U postgres -d ${POSTGRES_DB} -c "SELECT COUNT(*) FROM books;" || exit 1
    docker exec -it ${POSTGRES_CONTAINER_NAME} psql -U postgres -d ${POSTGRES_DB} -c "SELECT * FROM books LIMIT 5;" || exit 1
    echo "Finished verifying data in PostgreSQL."
    echo ""
}

echo "Starting end-to-end test for the entire data pipeline..."

if [ "$ENABLE_VENV_SETUP" = true ]; then
    run_make_command "setup"
fi

run_make_command "start-postgres"
check_postgres_container
test_postgres_connection
run_make_command "create-db"

if [ "$ENABLE_SCRAPING" = true ]; then
    run_make_command "run-scrapy"
fi

run_make_command "compress"
run_make_command "prepare"
run_make_command "load"

verify_postgres_data

trap - EXIT
echo "All steps completed successfully. Performing final cleanup..."

run_make_command "stop-postgres"
run_make_command "delete-postgres"
run_make_command "clean"

echo "End-to-end test and cleanup completed successfully."
