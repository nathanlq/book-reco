import asyncio
import asyncpg
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
import json
import hashlib
from datetime import datetime
import mlflow
import mlflow.sklearn
from common.setup_mlflow_autolog import setup_mlflow_autolog

setup_mlflow_autolog(experiment_name="compress_prepare_load")

load_dotenv()

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')
TABLE_NAME = os.getenv('TABLE_NAME')

with open('data/schemes/books.json', 'r') as file:
    schema = json.load(file)

df = pd.read_parquet('data/cleaned_data.parquet')

data = df.to_dict(orient='records')

for record in data:
    hash_data = {
        'product_title': record['product_title'],
        'author': record['author'],
        'editeur': record['editeur'],
        'format': record['format'],
        'date': str(record['date_de_parution'])
    }

    record_str = json.dumps(hash_data, sort_keys=True).encode('utf-8')
    record['id'] = hashlib.sha256(record_str).hexdigest()

    record['labels'] = json.dumps(record['labels'].tolist())

    if isinstance(record['date_de_parution'], datetime):
        record['date_de_parution'] = record['date_de_parution']

    for field in ['poids', 'collection', 'presentation', 'format']:
        if pd.isna(record[field]):
            record[field] = None
    record['utils'] = json.dumps({'image_downloaded': False})


async def table_exists(conn):
    result = await conn.fetchval(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{TABLE_NAME}')")
    return result

async def create_database_if_not_exists():
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database="postgres"
    )
    
    result = await conn.fetchval(f"SELECT 1 FROM pg_database WHERE datname = '{POSTGRES_DB}'")
    if not result:
        await conn.execute(f"CREATE DATABASE {POSTGRES_DB};")
        print(f"Database {POSTGRES_DB} created.")
    else:
        print(f"Database {POSTGRES_DB} already exists.")
    await conn.close()
    
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB
    )
    
    extension_installed = await conn.fetchval("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
    if not extension_installed:
        print("Extension 'vector' is not available on this server. Please install it on the PostgreSQL server.")
    else:
        extension_active = await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if not extension_active:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("Extension 'vector' has been enabled.")
        else:
            print("Extension 'vector' is already enabled.")
    await conn.close()

async def create_table(conn):
    if not await table_exists(conn):
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        columns = ", ".join(
            [f"{col['name']} {col['type']}" for col in schema['columns']])
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                {columns}
            )
        """)


async def drop_table(conn):
    await conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")


async def insert_data(conn, data):
    async with conn.transaction():
        for record in data:
            try:
                await conn.execute(f"""
                    INSERT INTO {TABLE_NAME} (
                        id, product_title, author, resume, labels, image_url, collection,
                        date_de_parution, ean, editeur, format, isbn, nb_de_pages,
                        poids, presentation, width, height, depth, utils
                    ) VALUES (
                        $1, $2, $3, $4, $5::JSONB, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, $19::JSONB
                    )
                    ON CONFLICT (id) DO NOTHING
                """,
                                   record['id'], record['product_title'], record['author'],
                                   record['resume'], record['labels'], record['image_url'],
                                   record['collection'], record['date_de_parution'], record['ean'],
                                   record['editeur'], record['format'], record['isbn'],
                                   record['nb_de_pages'], record['poids'], record['presentation'],
                                   record['width'], record['height'], record['depth'], record['utils']
                                   )
            except Exception as e:
                print(f"Error inserting record: {record}")
                print(f"Error message: {e}")


async def retrieve_data(conn):
    rows = await conn.fetch(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
    for row in rows:
        print(row)
    print("Retrieve OK.")


async def main(drop_flag=False):
    await create_database_if_not_exists()

    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB
    )

    if drop_flag:
        await drop_table(conn)

    await create_table(conn)
    await insert_data(conn, data)
    await retrieve_data(conn)
    await conn.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Loader script with optional table drop.")
    parser.add_argument("--drop", action="store_true",
                        help="Drop the table before recreating it.")

    args = parser.parse_args()
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with mlflow.start_run(run_name="loader_run"):
        asyncio.run(main(args.drop))

        mlflow.log_param("start_time", start_time)

        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        mlflow.log_param("end_time", end_time)

        mlflow.log_param("drop_table", args.drop)
        mlflow.log_param("table_name", TABLE_NAME)

        mlflow.log_metric("num_records", len(data))

        mlflow.log_artifact('data/cleaned_data.parquet')
