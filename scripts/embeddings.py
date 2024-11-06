import asyncio
import asyncpg
import os
import numpy as np
import json
from dotenv import load_dotenv
import torch
from transformers import CamembertTokenizer, CamembertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

load_dotenv()

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')
TABLE_NAME = os.getenv('TABLE_NAME')

model_name = 'camembert-base'
tokenizer = CamembertTokenizer.from_pretrained(model_name)
model = CamembertModel.from_pretrained(model_name)

with open('data/stop_words_french.txt', 'r', encoding='utf-8') as file:
    french_stop_words = [line.strip() for line in file]


def get_embedding(text, tokenizer, model, max_length=512):
    """Get the embedding for a given text."""
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding.flatten().tolist()


def calculate_embedding_for_record(row, tokenizer, model, max_length=512):
    """Calculate embedding for resume and product title."""
    resume_embedding = get_embedding(
        row['resume'], tokenizer, model, max_length)
    product_title_embedding = get_embedding(
        row['product_title'], tokenizer, model, max_length)
    return resume_embedding, product_title_embedding


def apply_tfidf_to_column(column, vectorizer):
    """Apply TF-IDF to a column and return the results as a list of vectors."""
    tfidf_matrix = vectorizer.transform(column)
    return tfidf_matrix.toarray().tolist()


def load_weights():
    """Load weights from a JSON file."""
    with open("data/schemes/weights.json", "r") as f:
        weights = json.load(f)
    return weights


def compute_similarity(embeddings):
    """Compute cosine similarity for a given set of embeddings."""
    return cosine_similarity(embeddings)


def get_top_n_similar_items(index, similarity_matrix, n=5):
    """Get the indices of the top n similar items for a given index."""
    similar_indices = np.argsort(similarity_matrix[index])[::-1][1:n+1]
    return similar_indices


async def update_combined_vectors(conn, recalculate_all=False, weights=None):
    print("Fetching rows to update combined vectors...")
    query = f"SELECT * FROM {TABLE_NAME}"
    if not recalculate_all:
        query += " WHERE embedding IS NULL"

    rows = await conn.fetch(query)
    if not rows:
        print("Nothing to calculate.")
        return
    print(f"Fetched {len(rows)} rows for processing.")

    combined_text = [row['resume'] + " " + row['product_title']
                     for row in rows]
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=french_stop_words, max_features=1024)
    tfidf_vectorizer.fit(combined_text)
    print("TF-IDF fitting complete.")

    embedding_weights = weights['embedding_weights']
    categorical_weights = weights['categorical_weights']
    labels_weight = weights['labels_weight']

    print(f"Embedding weights: {embedding_weights}")
    print(f"Categorical weights: {categorical_weights}")
    print(f"Labels weight: {labels_weight}")

    resume_embeddings = []
    product_title_embeddings = []
    resume_tfidf = []
    product_title_tfidf = []

    categorical_columns_with_labels = [
        'author', 'collection', 'editeur']
    categorical_labels = []

    print("Starting the embedding and TF-IDF extraction for each row...")
    for i, row in enumerate(rows):
        print(f"Processing row {i+1}/{len(rows)}: {row['id']}")
        resume_embedding, product_title_embedding = calculate_embedding_for_record(
            row, tokenizer, model)

        resume_tfidf_val = apply_tfidf_to_column(
            [row['resume']], tfidf_vectorizer)[0]
        product_title_tfidf_val = apply_tfidf_to_column(
            [row['product_title']], tfidf_vectorizer)[0]

        resume_embeddings.append(resume_embedding)
        product_title_embeddings.append(product_title_embedding)
        resume_tfidf.append(resume_tfidf_val)
        product_title_tfidf.append(product_title_tfidf_val)

        categorical_labels.append([row[label]
                                  for label in categorical_columns_with_labels])
    print("Embedding and TF-IDF extraction completed.")

    resume_embeddings = np.array(resume_embeddings)
    product_title_embeddings = np.array(product_title_embeddings)
    resume_tfidf = np.array(resume_tfidf)
    product_title_tfidf = np.array(product_title_tfidf)

    encoder = OneHotEncoder(sparse_output=False)
    categorical_labels_one_hot = encoder.fit_transform(categorical_labels)
    print("One-hot encoding complete.")

    num_categories = [len(encoder.categories_[i])
                      for i in range(len(categorical_columns_with_labels))]

    weight_matrix = np.concatenate([np.full(num_cat, weight) for num_cat, weight in zip(
        num_categories, categorical_weights.values())])

    scaled_categorical_labels = categorical_labels_one_hot * weight_matrix

    mlb = MultiLabelBinarizer()
    labels_binary_matrix = mlb.fit_transform([row['labels'] for row in rows])
    scaled_labels_binary_matrix = labels_binary_matrix * labels_weight
    print("Multi-label binarization complete.")

    resume_embeddings *= embedding_weights['resume']
    product_title_embeddings *= embedding_weights['product_title']
    resume_tfidf *= embedding_weights['resume_tfidf']
    product_title_tfidf *= embedding_weights['product_title_tfidf']

    combined_features = np.hstack((resume_embeddings, product_title_embeddings, resume_tfidf,
                                  product_title_tfidf, scaled_categorical_labels, scaled_labels_binary_matrix))
    print("Feature combination complete.")

    for i, row in enumerate(rows):
        combined_vector = combined_features[i]
        combined_vector_str = '[' + ','.join(map(str, combined_vector)) + ']'

        try:
            await conn.execute(
                f"UPDATE {TABLE_NAME} SET embedding = $1 WHERE id = $2",
                combined_vector_str, row['id']
            )
            print(f"Updated combined vector for book ID {row['id']}")
        except Exception as e:
            print(
                f"Error updating combined vector for book ID {row['id']}: {e}")


async def main(recalculate_all=False):
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB
    )

    weights = load_weights()

    try:
        while True:
            await update_combined_vectors(conn, recalculate_all, weights)
            await asyncio.sleep(60)
    finally:
        await conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Update combined vectors for books.")
    parser.add_argument(
        "--recalculate-all",
        action="store_true",
        help="Recalculate combined vectors for all records, not just those with missing vectors."
    )
    args = parser.parse_args()

    asyncio.run(main(args.recalculate_all))
