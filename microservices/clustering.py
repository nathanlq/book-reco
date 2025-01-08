import asyncio
import numpy as np
import joblib
import json
import mlflow
import os
import asyncpg
import json
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from common.setup_mlflow_autolog import setup_mlflow_autolog
from common.utils import reconnect, execute_batch_updates, TABLE_NAME

KMEANS_MODEL_PATH = 'data/models/kmeans_model.joblib'
try:
    NUM_CLUSTERS = int(os.getenv('NUM_CLUSTERS', 5))
    if NUM_CLUSTERS <= 0:
        raise ValueError("NUM_CLUSTERS must be a positive integer.")
except ValueError as e:
    raise RuntimeError(f"Invalid NUM_CLUSTERS value: {e}")

setup_mlflow_autolog(experiment_name="kmeans_clustering")

def balance_clusters(labels, num_clusters):
    cluster_sizes = np.bincount(labels, minlength=num_clusters)
    target_size = len(labels) // num_clusters
    balanced_labels = labels.copy()

    for i in range(num_clusters):
        if cluster_sizes[i] > target_size:
            excess = cluster_sizes[i] - target_size
            for j in range(num_clusters):
                if cluster_sizes[j] < target_size:
                    needed = target_size - cluster_sizes[j]
                    move_count = min(excess, needed)
                    move_indices = np.where(balanced_labels == i)[0][:move_count]
                    balanced_labels[move_indices] = j
                    cluster_sizes[i] -= move_count
                    cluster_sizes[j] += move_count
                    excess -= move_count
                    if excess == 0:
                        break
    return balanced_labels

async def initialize_kmeans_model(conn, table_name, num_clusters=NUM_CLUSTERS):
    setup_mlflow_autolog(experiment_name="kmeans_clustering")
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with mlflow.start_run(run_name="initialize_kmeans_model_run"):
        query = f"SELECT tfidf FROM {table_name} WHERE tfidf IS NOT NULL"
        rows = await conn.fetch(query)

        if not rows:
            print("No TF-IDF data found for KMeans training.")
            return

        tfidf_vectors = np.array(
            [np.fromstring(row['tfidf'][1:-1], sep=',') for row in rows])
        print(f"Fetched {len(tfidf_vectors)} rows for KMeans training.")

        train_vectors, val_vectors = train_test_split(
            tfidf_vectors, test_size=0.2, random_state=42)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(train_vectors)

        joblib.dump(kmeans, KMEANS_MODEL_PATH)

        train_inertia = kmeans.inertia_
        mlflow.log_metric("train_inertia", train_inertia)
        print(f"Training inertia: {train_inertia}")

        val_labels = kmeans.predict(val_vectors)
        val_labels = balance_clusters(val_labels, num_clusters)
        unique_labels = np.unique(val_labels)

        if len(unique_labels) < 2:
            print(
                f"Warning: Only {len(unique_labels)} unique labels found in validation set. Skipping silhouette score calculation.")
            silhouette_avg = None
        else:
            silhouette_avg = silhouette_score(val_vectors, val_labels)
            mlflow.log_metric("silhouette_score", silhouette_avg)
            print(f"Validation Silhouette Score: {silhouette_avg}")

        sample_input = tfidf_vectors[:1]
        sample_output = kmeans.predict(sample_input)
        signature = mlflow.models.signature.infer_signature(
            sample_input, sample_output)
        mlflow.sklearn.log_model(kmeans, "kmeans_model", signature=signature)

        print("KMeans model trained, evaluated, and saved to disk.")
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mlflow.log_param("start_time", start_time)
        mlflow.log_param("end_time", end_time)

        await labelize_new_rows(conn, recalculate_all=True)

async def labelize_new_rows(conn, recalculate_all=False):
    if not os.path.exists(KMEANS_MODEL_PATH):
        print("KMeans model not found. Please run initialize_kmeans_model first.")
        return

    kmeans = joblib.load(KMEANS_MODEL_PATH)

    if recalculate_all:
        query = f"SELECT id, tfidf, utils FROM {TABLE_NAME} WHERE tfidf IS NOT NULL"
    else:
        query = f"SELECT id, tfidf, utils FROM {TABLE_NAME} WHERE tfidf IS NOT NULL AND (utils->>'dynamic_cluster_number') IS NULL"

    rows = await conn.fetch(query)

    if not rows:
        print("No new rows to labelize.")
        return

    updates = []
    for row in tqdm(rows, desc="Labelizing rows", unit="row"):
        tfidf_vector = np.fromstring(row['tfidf'][1:-1], sep=',')
        cluster_label = int(kmeans.predict([tfidf_vector])[0])

        utils = json.loads(row['utils']) if row['utils'] else {}
        utils['dynamic_cluster_number'] = cluster_label
        updates.append((json.dumps(utils), row['id']))

        if len(updates) >= 100:
            await execute_batch_updates(conn, updates, f"UPDATE {TABLE_NAME} SET utils = $1 WHERE id = $2")
            updates = []

    if updates:
        await execute_batch_updates(conn, updates, f"UPDATE {TABLE_NAME} SET utils = $1 WHERE id = $2")
    print("Finished labeling new rows.")

async def weekly_retrain_task(conn, lock, num_clusters=NUM_CLUSTERS):
    while True:
        now = datetime.now()
        next_run = datetime(now.year, now.month, now.day,
                            0, 0, 0) + timedelta(weeks=1)

        wait_time = (next_run - now).total_seconds()
        print(
            f"KMeans retrain task scheduled to run in {wait_time / 3600:.2f} hours.")
        await asyncio.sleep(wait_time)

        async with lock:
            print("Starting weekly KMeans retraining...")
            await initialize_kmeans_model(conn, TABLE_NAME, num_clusters)
            print("Weekly KMeans retraining complete.")

async def new_row_watcher_task(conn, lock):
    while True:
        async with lock:
            print("Checking for new rows to labelize...")
            await labelize_new_rows(conn)
        await asyncio.sleep(300)

async def main():
    while True:
        try:
            conn = await reconnect()
            lock = asyncio.Lock()

            try:
                await initialize_kmeans_model(conn, TABLE_NAME)
                await asyncio.gather(
                    weekly_retrain_task(conn, lock, num_clusters=NUM_CLUSTERS),
                    new_row_watcher_task(conn, lock)
                )
            finally:
                await conn.close()
        except asyncpg.exceptions.InterfaceError as e:
            print(f"Database connection error: {e}. Retrying in 30 seconds...")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
