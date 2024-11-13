## Project Documentation

### 1. Project Overview
The project is a data processing pipeline designed to scrape book data from an online bookstore, transform and load it into a PostgreSQL database, and expose it via a FastAPI-based API. The pipeline includes:

- **Data Collection**: Scrapes book details with Scrapy.
- **Data Processing**: Cleans and formats the scraped data, compresses it, and loads it into a database.
- **API Exposure**: Provides endpoints to retrieve book information and similar books based on embeddings calculated using BERT and TF-IDF.

### 2. System Architecture
This section includes the flow and structure of the pipeline as visualized in the architecture diagram (refer to `BigData.png`):

- **Crawler & Scrapers**: Collect data from the target website.
- **Data Processing**: Cleans and formats the data into the appropriate schema.
- **Vectorization**: Converts text data to embeddings using CamemBERT and calculates TF-IDF vectors.
- **Database Storage**: Stores books in a PostgreSQL database using the `pgvector` extension for efficient similarity queries.
- **API & Microservices**: Provides API endpoints for book retrieval and similarity search, along with daily and dynamic vector recalculations.

### 3. Modules Overview

#### `microservices`
A microservice module responsible for calculating text embeddings and TF-IDF vectors for each book in the database. This module performs:

- **Daily Vector Recalculation**: Runs every day at 1 a.m., updating embeddings and TF-IDF vectors for all books.
- **Embedding Calculations**: Uses BERT-based embeddings stored as vectors for similarity queries.
  
#### `collect`
Uses Scrapy to scrape book details from the website. The scraped data is output in JSON format for further processing.

#### `store`
Handles data transformation and loading tasks, with three main scripts:

- **`compress.py`**: Converts the JSON output from Scrapy into Parquet format, a compressed, columnar storage format.
- **`prepare.py`**: Cleans and processes data to ensure consistent data types, handling missing values, and performing any necessary formatting.
- **`loader.py`**: Loads cleaned data into a PostgreSQL database, adhering to the schema specified in `book.json` (described below).

#### `expose`
A FastAPI-based module that provides RESTful API endpoints. Key endpoints include:

- **GET /books**: Retrieve books with various filtering options (title, author, date, etc.).
- **GET /books/{book_id}/similar**: Retrieve books similar to a given book based on its embeddings, with optional filters on categorical fields (author, collection, editor, etc.).

### 4. Database Schema
The database schema, defined in `book.json`, structures book data to support filtering and vector-based similarity search:

```json
{
    "columns": [
        {"name": "id", "type": "TEXT PRIMARY KEY"},
        {"name": "product_title", "type": "TEXT"},
        {"name": "author", "type": "TEXT"},
        {"name": "resume", "type": "TEXT"},
        {"name": "labels", "type": "JSONB"},
        {"name": "image_url", "type": "TEXT"},
        {"name": "collection", "type": "TEXT"},
        {"name": "date_de_parution", "type": "DATE"},
        {"name": "ean", "type": "BIGINT"},
        {"name": "editeur", "type": "TEXT"},
        {"name": "format", "type": "TEXT"},
        {"name": "isbn", "type": "TEXT"},
        {"name": "nb_de_pages", "type": "INT"},
        {"name": "poids", "type": "DECIMAL(5, 3)"},
        {"name": "presentation", "type": "TEXT"},
        {"name": "width", "type": "DECIMAL(5, 2)"},
        {"name": "height", "type": "DECIMAL(5, 2)"},
        {"name": "depth", "type": "DECIMAL(5, 2)"},
        {"name": "embedding", "type": "VECTOR(128)"},
        {"name": "tfidf", "type": "VECTOR(2048)"}
    ]
}
```

### 5. API Endpoints

`GET /books`

Retrieve books from the database with filtering options:

```python
@router.get("/books", response_model=List[Book])
async def get_books(...):
    # Filters by fields like `product_title`, `author`, `date_de_parution`, etc.
    # Pagination supported through `page` and `page_size` parameters.
```

Parameters:

- `id`, `product_title`, `author`, etc.: Optional filters on book fields.
- `page` and `page_size`: Pagination controls.

`GET /books/{book_id}/similar`

Retrieve similar books based on the embedding vector of a specified `book_id`.

```python
@router.get("/books/{book_id}/similar", response_model=List[Book])
async def get_similar_books(...):
    # Retrieves books similar to the given `book_id` using specified distance methods.
```
Parameters:

- `method`: Similarity metric, e.g., "cosine," "euclidean," or "taxicab."
- Optional filters to narrow down by author, collection, editor, etc.

### 6. Running the Pipeline

To operate the entire pipeline without duplicating the Makefile documentation, follow these steps:

- Data Collection: Run the Scrapy crawler to gather raw book data in JSON format.
- Data Transformation and Storage: Use the `store` module scripts to process the JSON data into a clean, compressed format and load it into the PostgreSQL database.
- Daily Embedding Update: Scheduled recalculation of embeddings and TF-IDF vectors at 1 a.m., managed within the `microservices` module.
- API Exposure: Run the FastAPI server to provide RESTful access to the database.

Refer to the Makefile readme for specific command details.

### 7. Environment Setup

To set up the environment, cf `readme-makefile`. Environment variables are located in `.env`.


### 8. Testing

For testing, utilize the make-test.sh script. This script automates testing across the different modules and verifies end-to-end functionality. For now, only the module `collect` and `store` can be tested. See the Makefile readme for available testing flags.