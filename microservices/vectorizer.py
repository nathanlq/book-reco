import asyncio
import mlflow
import asyncpg
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm
from common.setup_mlflow_autolog import setup_mlflow_autolog
from common.utils import reconnect, execute_batch_updates, TABLE_NAME
from microservices.utils.vectors import generate_vectors_for_row, retrain_tfidf_model, initialize_pca_model, initialize_tfidf_model

setup_mlflow_autolog(experiment_name="vectorizer_monitoring")


async def update_combined_vectors(conn, recalculate_all=False):
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with mlflow.start_run(run_name="update_combined_vectors_run"):
        print("Fetching rows to update vectors...")
        query = f"SELECT * FROM {TABLE_NAME}"
        if not recalculate_all:
            query += " WHERE embedding IS NULL OR tfidf IS NULL"

        rows = await conn.fetch(query)
        if not rows:
            print("Nothing to calculate.")
            mlflow.log_param("num_rows", 0)

            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            mlflow.log_param("start_time", start_time)
            mlflow.log_param("end_time", end_time)
            mlflow.log_param("recalculate_all", recalculate_all)
            return
        print(f"Fetched {len(rows)} rows for processing.")

        batch_size = 100
        updates = []

        for row in tqdm(rows, desc="Processing rows", unit="row"):
            embedding_vector, tfidf_vector = generate_vectors_for_row(row)

            embedding_vector_str = '[' + \
                ','.join(map(str, embedding_vector)) + ']'
            tfidf_vector_str = '[' + ','.join(map(str, tfidf_vector)) + ']'

            updates.append((embedding_vector_str, tfidf_vector_str, row['id']))

            if len(updates) >= batch_size:
                await execute_batch_updates(conn, updates, f"UPDATE {TABLE_NAME} SET embedding = $1, tfidf = $2 WHERE id = $3")
                updates = []

        if updates:
            await execute_batch_updates(conn, updates, f"UPDATE {TABLE_NAME} SET embedding = $1, tfidf = $2 WHERE id = $3")

        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        mlflow.log_param("start_time", start_time)
        mlflow.log_param("end_time", end_time)
        mlflow.log_param("num_rows", len(rows))
        mlflow.log_param("recalculate_all", recalculate_all)


async def daily_recalculation_task(conn, lock):
    while True:
        now = datetime.now()
        next_run = datetime(now.year, now.month, now.day, 1, 0, 0)
        if now >= next_run:
            next_run += timedelta(days=1)

        wait_time = (next_run - now).total_seconds()
        print(
            f"Daily recalculation scheduled to run in {wait_time / 3600:.2f} hours.")
        await asyncio.sleep(wait_time)

        async with lock:
            print("Starting daily TF-IDF retraining and full recalculation of vectors...")
            await retrain_tfidf_model(conn, TABLE_NAME)
            await update_combined_vectors(conn, recalculate_all=True)
            print("Daily recalculation complete.")


async def new_vector_watcher_task(conn, lock):
    while True:
        async with lock:
            print("Running watcher to check for new rows needing vector calculations...")
            await update_combined_vectors(conn, recalculate_all=False)

        await asyncio.sleep(300)


async def main():
    while True:
        try:
            conn = await reconnect()

            lock = asyncio.Lock()

            try:
                await initialize_pca_model(conn, TABLE_NAME)
                await initialize_tfidf_model(conn, TABLE_NAME)
                await asyncio.gather(
                    daily_recalculation_task(conn, lock),
                    new_vector_watcher_task(conn, lock)
                )
            finally:
                await conn.close()
        except asyncpg.exceptions.InterfaceError as e:
            print(f"Connection error: {e}. Retrying in 30 seconds...")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
