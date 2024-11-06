import asyncio
import asyncpg
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
import json
import hashlib

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
    
    record['date_de_parution'] = int(record['date_de_parution'].timestamp() * 1000)

    for field in ['poids', 'collection', 'presentation', 'format']:
        if pd.isna(record[field]):
            record[field] = None

async def create_table(conn):
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    columns = ", ".join([f"{col['name']} {col['type']}" for col in schema['columns']])
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
                        poids, presentation, width, height, depth
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18
                    )
                    ON CONFLICT (id) DO NOTHING
                """,
                    record['id'], record['product_title'], record['author'],
                    record['resume'], record['labels'], record['image_url'],
                    record['collection'], record['date_de_parution'], record['ean'],
                    record['editeur'], record['format'], record['isbn'],
                    record['nb_de_pages'], record['poids'], record['presentation'],
                    record['width'], record['height'], record['depth']
                )
            except Exception as e:
                print(f"Error inserting record: {record}")
                print(f"Error message: {e}")

async def retrieve_data(conn):
    rows = await conn.fetch(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
    for row in rows:
        print(row)
    print("Retrieve OK.")

async def main():
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB
    )

    await drop_table(conn)
    await create_table(conn)
    await insert_data(conn, data)
    await retrieve_data(conn)
    await conn.close()

asyncio.run(main())
