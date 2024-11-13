# Store Module Documentation

The `store` module is responsible for processing, transforming, and storing the book data scraped by the `FuretSpider`. The module includes scripts for data cleaning and preparation, compressing data into efficient formats, loading data into a database, and performing basic database testing.

## Code Overview

The `store` module contains the following scripts:

1. `compress.py`
2. `prepare.py`
3. `loader.py`

Each script has a specific role in the data pipeline, from initial data compression to loading into the database.

### `compress.py`

This script reads the raw JSON data output from the `collect` step and converts it into the more efficient Parquet format.

- **Input**: `data/raw_output.json`
- **Output**: `data/raw_data.parquet`
- **Process**:
  - Uses `pandas` to load the JSON file and saves it as a Parquet file, which optimizes storage and read/write speeds for subsequent steps.

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
  
## Summary

The `store` module systematically prepares and loads book data for storage in PostgreSQL, making it ready for efficient retrieval and analysis. It includes optimized storage with Parquet (`compress.py`), comprehensive data processing (`prepare.py`), and reliable loading (`loader.py`) forming a robust data pipeline.