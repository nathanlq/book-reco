# Expose Module

The `expose` module provides a REST API for managing and retrieving book data stored in PostgreSQL. Built with FastAPI, this module offers endpoints to search and filter books and find similar books based on various criteria. It consists of configuration, database connection management, data models, routing, and API logic.

## File Structure

- **config.py**: Loads environment variables related to PostgreSQL configuration.
- **database.py**: Manages asynchronous PostgreSQL connections using `asyncpg`.
- **main.py**: Initializes the FastAPI app and includes the router for handling API endpoints.
- **models.py**: Defines the `Book` data model using Pydantic, representing the structure of book data.
- **routes.py**: Contains API route definitions for book retrieval and similarity search.

## Environment Variables

The `config.py` file loads the following environment variables:

- `POSTGRES_USER`: Username for PostgreSQL.
- `POSTGRES_PASSWORD`: Password for PostgreSQL.
- `POSTGRES_HOST`: Hostname for the PostgreSQL server.
- `POSTGRES_PORT`: Port on which PostgreSQL is running.
- `POSTGRES_DB`: Name of the PostgreSQL database.
- `TABLE_NAME`: Name of the database table storing book data.

Ensure these variables are set in a `.env` file in the root directory.

## Database Connection

The `database.py` file provides the `get_db_connection()` function, which establishes an asynchronous connection to the PostgreSQL database. This function is used within `routes.py` to access the book data.

## Data Models

The `models.py` file defines a `Book` model using Pydantic, representing the schema of each book entry, which includes fields like:

- `id`: Unique identifier for the book.
- `product_title`, `author`, `resume`, `image_url`: Information about the book.
- `collection`, `date_de_parution`, `ean`, `editeur`, `format`: Metadata fields.
- `isbn`, `nb_de_pages`, `poids`, `presentation`: Additional book details.
- `width`, `height`, `depth`: Physical dimensions of the book.

This model is used for validating and serializing data returned from the API.

## API Routes

The `routes.py` file defines the following API endpoints:

### 1. **GET** `/books`

Retrieves a list of books, with optional filtering and pagination parameters.

**Query Parameters**:
- `id`: Filter by book ID.
- `product_title`, `author`, `resume`, `image_url`: Filter based on book metadata.
- `collection`, `date_de_parution`, `ean`, `editeur`, `format`: Metadata filters.
- `isbn`, `nb_de_pages`, `poids`, `presentation`: Additional filters for book details.
- `width`, `height`, `depth`: Physical dimension filters.
- `page`: Page number for pagination (default is 1).
- `page_size`: Number of items per page (default is 10).

This endpoint supports complex filtering by combining multiple criteria in a single request. Pagination is handled through `page` and `page_size`.

### 2. **GET** `/books/{book_id}/similar`

Finds books similar to the specified book ID based on content embeddings and optional filtering criteria.

**Path Parameter**:
- `book_id`: The unique ID of the reference book for similarity search.

**Query Parameters**:
- `method`: Method for similarity calculation (supports `cosine`, `euclidean`, and `taxicab`).
- `author`, `collection`, `editeur`, `format`: Optional boolean filters to match similar books by specific metadata.

The similarity search uses embeddings stored in the `embedding` column, ranking results based on the selected method. By default, it uses cosine similarity but can also apply Euclidean or taxicab (Manhattan) distance.

## Usage

To start the `expose` module, run the FastAPI server in `main.py`. Make sure that:

- PostgreSQL is running and accessible with the credentials provided in `.env`.
- The database and table structure are prepared to support the module’s queries.
  
This module serves as a microservice for book data retrieval and analysis, supporting a scalable and asynchronous API.
