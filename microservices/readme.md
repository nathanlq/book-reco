# Microservices Module Documentation

This module consists of vectorization and database update services for a text processing pipeline. It includes two main files:
1. **vectors.py** - Contains functions for generating embeddings and TF-IDF vectors and training the PCA and TF-IDF models.
2. **vectorizer.py** - Manages asynchronous tasks for batch updates, recalculations, and scheduled retraining of vectors.

## `vectors.py`

This file is responsible for text vectorization, including creating embeddings and TF-IDF vectors and training or loading PCA and TF-IDF models. It uses the CamemBERT model for embedding and TF-IDF for keyword relevance.

### Global Variables and Initialization

- **MODEL_DIR, PCA_MODEL_PATH, TFIDF_MODEL_PATH, STOP_WORDS_PATH**: Paths to the storage directory and files for PCA and TF-IDF models, and stop words.
- **french_stop_words**: List of stop words read from the `STOP_WORDS_PATH` file.
- **model_name**: The CamemBERT model identifier used for generating embeddings.
- **tokenizer, model**: CamemBERT tokenizer and model objects for embedding.
- **pca, tfidf_vectorizer**: Initialized PCA and TF-IDF vectorizer, loaded from disk if the models already exist.

### Functions

#### `initialize_pca_model(conn, table_name)`
Asynchronously initializes or loads a PCA model from a database.

- **Parameters**:
  - `conn`: Database connection object.
  - `table_name`: Database table name containing text data.
- **Functionality**:
  - Loads the PCA model if saved; otherwise, fetches text data from the database, generates embeddings, trains a PCA model, and saves it.

#### `initialize_tfidf_model(conn, table_name)`
Asynchronously initializes or loads a TF-IDF vectorizer from the database.

- **Parameters**:
  - `conn`: Database connection object.
  - `table_name`: Database table name containing text data.
- **Functionality**:
  - Loads the TF-IDF model if saved; otherwise, fetches data from the database, trains a TF-IDF vectorizer on combined text data, and saves it.

#### `get_embedding(text, max_length=512, apply_pca=True)`
Generates a CamemBERT embedding for a given text.

- **Parameters**:
  - `text`: Input text for embedding.
  - `max_length`: Maximum token length for truncation.
  - `apply_pca`: Flag indicating whether to reduce embedding dimensionality using PCA.
- **Returns**: Reduced or original embedding vector for the input text.

#### `generate_tfidf_vector(column)`
Generates a TF-IDF vector for a text column.

- **Parameters**:
  - `column`: List of text inputs to vectorize.
- **Returns**: TF-IDF vector for the input text.
- **Exceptions**: Raises an error if the TF-IDF vectorizer is not initialized.

#### `generate_vectors_for_row(row)`
Generates embedding and TF-IDF vectors for a database row.

- **Parameters**:
  - `row`: Dictionary containing `resume` and `product_title` text fields.
- **Returns**: Tuple with embedding and TF-IDF vectors for the combined text.

#### `retrain_tfidf_model(conn, table_name)`
Retrains the TF-IDF model using the entire database content.

- **Parameters**:
  - `conn`: Database connection object.
  - `table_name`: Table name containing text data.
- **Functionality**:
  - Fetches text data, retrains the TF-IDF model, and saves the updated model.

#### `retrain_pca_model(conn, table_name)`
Retrains the PCA model using the entire database content.

- **Parameters**:
  - `conn`: Database connection object.
  - `table_name`: Table name containing text data.
- **Functionality**:
  - Fetches text data, generates embeddings, retrains the PCA model, and saves it.

## `vectorizer.py`

This file orchestrates asynchronous tasks for updating vector representations in the database and scheduled model retraining.

### Global Variables and Environment Variables

- **POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, TABLE_NAME**: Database credentials and table name loaded from environment variables.

### Functions

#### `reconnect()`
Re-establishes a database connection using the provided environment variables.

- **Returns**: An `asyncpg` database connection object.

#### `update_combined_vectors(conn, recalculate_all=False)`
Fetches rows from the database and updates vector representations.

- **Parameters**:
  - `conn`: Database connection object.
  - `recalculate_all`: Flag indicating whether to update all rows or only those without vector data.
- **Functionality**:
  - For each row, generates embedding and TF-IDF vectors, then updates the database in batches.

#### `execute_batch_updates(conn, updates)`
Executes batched vector updates in the database with retry logic.

- **Parameters**:
  - `conn`: Database connection object.
  - `updates`: List of (embedding, TF-IDF, ID) tuples for database update.
- **Functionality**:
  - Updates embeddings and TF-IDF vectors for a batch of rows, with retries in case of connection issues.

#### `daily_recalculation_task(conn, lock)`
Schedules a daily recalculation task to update vectors and retrain models at a fixed time.

- **Parameters**:
  - `conn`: Database connection object.
  - `lock`: Asyncio lock to prevent concurrent recalculations.
- **Functionality**:
  - Waits until the scheduled time, then triggers TF-IDF retraining and updates all vectors.

#### `new_vector_watcher_task(conn, lock)`
Continuously checks for new rows needing vector calculations and updates them.

- **Parameters**:
  - `conn`: Database connection object.
  - `lock`: Asyncio lock for task synchronization.
- **Functionality**:
  - Runs in a loop with a 5-minute delay, updating only new rows needing vector calculations.

#### `main()`
The entry point for the vectorizer service.

- **Functionality**:
  - Establishes a database connection and initializes PCA and TF-IDF models if not already loaded.
  - Starts concurrent tasks for daily recalculations and new row monitoring.

---

### Notes

- **Dependencies**: Requires `asyncpg`, `torch`, `transformers`, `joblib`, `scikit-learn`, and `tqdm`.
- **Models**: Uses CamemBERT for text embeddings, and `PCA` and `TF-IDF` for dimensionality reduction and keyword extraction.
- **Concurrency**: Asynchronous tasks handle vector calculations and model updates, allowing efficient database operations.