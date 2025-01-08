import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import CamembertModel, CamembertTokenizer
import joblib
import mlflow
from common.setup_mlflow_autolog import setup_mlflow_autolog
from datetime import datetime

MODEL_DIR = 'data/models'
PCA_MODEL_PATH = os.path.join(MODEL_DIR, 'pca_model.joblib')
TFIDF_MODEL_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
STOP_WORDS_PATH = 'data/stop_words_french.txt'

with open(STOP_WORDS_PATH, 'r', encoding='utf-8') as file:
    french_stop_words = [line.strip() for line in file]

model_name = 'camembert-base'
tokenizer = CamembertTokenizer.from_pretrained(model_name)
model = CamembertModel.from_pretrained(model_name)

pca = PCA(n_components=128)
tfidf_vectorizer = TfidfVectorizer(
    stop_words=french_stop_words, max_features=4096)

if os.path.exists(PCA_MODEL_PATH):
    pca = joblib.load(PCA_MODEL_PATH)
    print("Loaded PCA model from disk.")
else:
    print("PCA model not found; will train when needed.")

if os.path.exists(TFIDF_MODEL_PATH):
    tfidf_vectorizer = joblib.load(TFIDF_MODEL_PATH)
    print("Loaded TF-IDF vectorizer from disk.")
else:
    print("TF-IDF vectorizer not found; will train when needed.")


async def initialize_pca_model(conn, table_name):
    global pca
    setup_mlflow_autolog(experiment_name="vectorizer_monitoring")
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with mlflow.start_run(run_name="initialize_pca_model_run"):
        mlflow.log_param("disk_loaded", os.path.exists(PCA_MODEL_PATH))
        if os.path.exists(PCA_MODEL_PATH):
            pca = joblib.load(PCA_MODEL_PATH)
            print("Loaded PCA model from disk.")
        else:
            print("Initializing PCA model with sufficient samples.")

            query = f"SELECT resume, product_title FROM {table_name}"
            rows = await conn.fetch(query)

            if not rows:
                print("No data found for PCA initialization.")
                return

            combined_texts = [f"{row['resume']} {row['product_title']}".strip()
                              for row in rows]
            embeddings = []
            for text in tqdm(combined_texts, desc="Generating embeddings for PCA", unit="text"):
                embedding = get_embedding(text, apply_pca=False)
                embeddings.append(embedding)

            embeddings_matrix = np.array(embeddings)
            pca = PCA(n_components=128)
            pca.fit(embeddings_matrix)
            joblib.dump(pca, PCA_MODEL_PATH)

            sample_input = embeddings_matrix[:1]
            sample_output = pca.transform(sample_input)
            signature = mlflow.models.signature.infer_signature(
                sample_input, sample_output)
            mlflow.sklearn.log_model(pca, "pca_model", signature=signature)

            print("PCA model trained with a batch of data and saved to disk.")

        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mlflow.log_param("start_time", start_time)
        mlflow.log_param("end_time", end_time)


async def initialize_tfidf_model(conn, table_name):
    global tfidf_vectorizer
    setup_mlflow_autolog(experiment_name="vectorizer_monitoring")
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with mlflow.start_run(run_name="initialize_tfidf_model_run"):
        mlflow.log_param("disk_loaded", os.path.exists(TFIDF_MODEL_PATH))
        if os.path.exists(TFIDF_MODEL_PATH):
            tfidf_vectorizer = joblib.load(TFIDF_MODEL_PATH)
            print("Loaded TF-IDF vectorizer from disk.")
        else:
            print("Initializing TF-IDF model with sufficient samples.")

            query = f"SELECT resume, product_title FROM {table_name}"
            rows = await conn.fetch(query)

            if not rows:
                print("No data found for TF-IDF initialization.")
                return

            combined_texts = [
                f"{row['resume']} {row['product_title']}".strip() for row in rows]

            tfidf_vectorizer = TfidfVectorizer(
                stop_words=french_stop_words, max_features=4096)

            print("Training TF-IDF vectorizer on fetched data...")
            tfidf_vectorizer.fit(combined_texts)

            joblib.dump(tfidf_vectorizer, TFIDF_MODEL_PATH)

            sample_input = combined_texts[:1]
            sample_output = tfidf_vectorizer.transform(sample_input).toarray()
            signature = mlflow.models.signature.infer_signature(
                sample_input, sample_output)
            mlflow.sklearn.log_model(
                tfidf_vectorizer, "tfidf_vectorizer", signature=signature)

            print("TF-IDF vectorizer trained and saved to disk.")

        if len(tfidf_vectorizer.get_feature_names_out()) == 4096:
            print("TF-IDF vectorizer has the correct number of features.")
        else:
            print(
                f"Warning: TF-IDF vectorizer has {len(tfidf_vectorizer.get_feature_names_out())} features, expected 4096.")

        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mlflow.log_param("start_time", start_time)
        mlflow.log_param("end_time", end_time)


def get_embedding(text, max_length=512, apply_pca=True):
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()

    if apply_pca:
        if not hasattr(pca, 'components_'):
            raise RuntimeError(
                "PCA model is not initialized. Please run initialize_pca_model first.")

        if embedding.shape[1] != 768:
            raise ValueError(
                f"Expected embedding size of 768, got {embedding.shape[1]}")

        reduced_embedding = pca.transform(embedding)
        return reduced_embedding.flatten()

    return embedding.flatten()


def generate_tfidf_vector(column):
    if not hasattr(tfidf_vectorizer, 'vocabulary_'):
        raise RuntimeError(
            "TF-IDF vectorizer is not fitted. Please run initialize_tfidf_model first.")

    tfidf_matrix = tfidf_vectorizer.transform(column)
    tfidf_vector = tfidf_matrix.toarray()[0]
    return tfidf_vector


def generate_vectors_for_row(row):
    embedding_text = f"{row['resume']} {row['product_title']}".strip()
    embedding_vector = get_embedding(embedding_text)
    combined_text = [row['resume'], row['product_title']]
    tfidf_vector = generate_tfidf_vector(combined_text)
    return embedding_vector, tfidf_vector


async def retrain_tfidf_model(conn, table_name):
    setup_mlflow_autolog(experiment_name="vectorizer_monitoring")
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with mlflow.start_run(run_name="retrain_tfidf_model_run"):
        print("Starting TF-IDF retraining with full dataset from database.")
        query = f"SELECT resume, product_title FROM {table_name}"
        rows = await conn.fetch(query)
        combined_text = [row['resume']
                         for row in rows] + [row['product_title'] for row in rows]

        tfidf_vectorizer = TfidfVectorizer(
            stop_words=french_stop_words, max_features=4096)
        tfidf_vectorizer.fit(combined_text)
        joblib.dump(tfidf_vectorizer, TFIDF_MODEL_PATH)

        sample_input = combined_text[:1]
        sample_output = tfidf_vectorizer.transform(sample_input).toarray()
        signature = mlflow.models.signature.infer_signature(
            sample_input, sample_output)
        mlflow.sklearn.log_model(
            tfidf_vectorizer, "tfidf_vectorizer", signature=signature)

        print("TF-IDF model retrained with the latest data and saved to disk.")

        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mlflow.log_param("start_time", start_time)
        mlflow.log_param("end_time", end_time)


async def retrain_pca_model(conn, table_name):
    setup_mlflow_autolog(experiment_name="vectorizer_monitoring")
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with mlflow.start_run(run_name="retrain_pca_model_run"):
        print("Starting PCA retraining with full dataset from database.")
        query = f"SELECT resume, product_title FROM {table_name}"
        rows = await conn.fetch(query)
        combined_texts = [f"{row['resume']} {row['product_title']}".strip()
                          for row in rows]

        embeddings = []
        for text in tqdm(combined_texts, desc="Generating embeddings for PCA retraining", unit="text"):
            embedding = get_embedding(text)
            embeddings.append(embedding)

        embeddings_matrix = np.array(embeddings)

        pca = PCA(n_components=128)
        pca.fit(embeddings_matrix)

        joblib.dump(pca, PCA_MODEL_PATH)

        sample_input = embeddings_matrix[:1]
        sample_output = pca.transform(sample_input)
        signature = mlflow.models.signature.infer_signature(
            sample_input, sample_output)
        mlflow.sklearn.log_model(pca, "pca_model", signature=signature)

        print("PCA model retrained with the latest data and saved to disk.")

        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mlflow.log_param("start_time", start_time)
        mlflow.log_param("end_time", end_time)
