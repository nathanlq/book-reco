# Store Module Documentation

The `store` module is responsible for processing, transforming, and storing the book data scraped by the `FuretSpider`. The module includes scripts for data cleaning and preparation, compressing data into efficient formats, loading data into a database, and performing basic database testing.

Additionally, MLflow is integrated throughout the module for tracking experiments, parameters, metrics, and artifacts associated with each data processing step. This helps in monitoring data transformations and database operations, facilitating easier debugging and reproducibility.

## Code Overview

The `store` module contains the following scripts:

1. `compress.py`
2. `prepare.py`
3. `loader.py`

Each script has a specific role in the data pipeline, from initial data compression to loading into the database.

## MLflow Integration

MLflow is used in the following ways across scripts in the `store` module:

- **Experiment Tracking**: All scripts use the same experiment name, `compress_prepare_load`, which organizes the runs into a single experiment. This makes it easy to view and compare runs across the different stages.
- **Run Logging**: Each script initializes an MLflow run with a unique `run_name`, indicating the script and process it is logging (e.g., `compress_run`, `prepare_run`, `loader_run`).
- **Parameter Logging**: Script parameters, such as the source and destination file paths, are logged. For `loader.py`, additional parameters related to database configurations are also logged.
- **Metric Logging**: Metrics such as the number of rows processed are logged, helping to monitor the volume of data handled at each step.
- Artifact Logging: Key output files (e.g., `data/raw_data.parquet` and `data/cleaned_data.parquet`) are saved as MLflow artifacts, allowing access to the processed data at each stage in the pipeline.

### `compress.py`

This script reads the raw JSON data output from the `collect` step and converts it into the more efficient Parquet format.

- **Input**: `data/raw_output.json`
- **Output**: `data/raw_data.parquet`
- **Process**:
  - Uses `pandas` to load the JSON file and saves it as a Parquet file, which optimizes storage and read/write speeds for subsequent steps.
  - **MLflow Logging**: Logs the input and output file paths as parameters, the row count as a metric, and the Parquet file as an artifact.

### `prepare.py`

This script performs extensive data cleaning, transformation, and normalization. It reads the raw Parquet file, processes various columns, and outputs a cleaned Parquet file.

- **Input**: `data/raw_data.parquet`
- **Output**: `data/cleaned_data.parquet`
- **Process**:
    - **Label Cleaning**: Explodes the `labels` column to handle each label individually, filters out blacklisted labels, and normalizes them to a standard format.
    - **Stop Words Removal**: Loads French stop words from a file (`data/stop_words_french.txt`) to aid in label cleaning.
    - **Metadata Extraction**: Extracts and flattens fields from the `information` column (e.g., ISBN, dimensions, publication date).
    - **Data Type Adjustments**: Converts columns like publication date and page numbers to appropriate data types.
    - **Categorical Encoding**: Converts columns like `author` and `editor` into categorical types and generates label-encoded versions for machine learning compatibility.
    - **Dimensions Parsing**: Extracts dimensions (width, height, depth) from a text column.
    - **MLflow Logging**: Logs input and output file paths, the shape of the cleaned data as metrics, and the cleaned Parquet file as an artifact. Additionally, an input schema, output schema, and a sample of the prepared data are logged to facilitate model signature verification.

### `loader.py`

This script loads the cleaned data into a PostgreSQL database, with a focus on handling JSON fields and creating a unique ID for each record.

- **Input**: `data/cleaned_data.parquet`
- **Database**: PostgreSQL
- **Process**:
    - **Schema Loading**: Loads a predefined schema from data/schemes/books.json.
    - **Record ID Generation**: Creates a unique SHA-256 hash ID for each record using key fields (e.g., title, author, editor).
    - **Database Connection**: Connects to PostgreSQL using `asyncpg`.
    - **Table Creation**: Creates a new table with the specified schema, using the `vector` extension for vector-based queries.
    - **Data Insertion**: Inserts records, skipping duplicates using the `ON CONFLICT DO NOTHING` clause.
    - **MLflow Logging**: Logs information about the database (e.g., table name, number of records) and whether the table was dropped before insertion. The cleaned data file is also logged as an artifact.
  
## Summary

The `store` module systematically prepares and loads book data for storage in PostgreSQL, making it ready for efficient retrieval and analysis. It includes optimized storage with Parquet (`compress.py`), comprehensive data processing (`prepare.py`), and reliable loading (`loader.py`) forming a robust data pipeline.

The integration of MLflow enhances the module’s functionality by tracking parameters, metrics, and artifacts throughout the data pipeline, making it easy to review and audit the data processing steps.