import asyncio
import asyncpg
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
import json

load_dotenv()

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')
TABLE_NAME = os.getenv('TABLE_NAME')

with open('data/schemes/books.json', 'r') as file:
    schema = json.load(file)

TABLE_NAME = schema['table_name']

df = pd.read_parquet('data/cleaned_data.parquet')

data = df.to_dict(orient='records')

for record in data:
    record['labels'] = json.dumps(record['labels'].tolist())
    record['date_de_parution'] = int(record['date_de_parution'].timestamp() * 1000)
    if pd.isna(record['poids']):
        record['poids'] = None
    if pd.isna(record['collection']):
        record['collection'] = None
    if pd.isna(record['presentation']):
        record['presentation'] = None
    if pd.isna(record['format']):
        record['format'] = None

print("First few records:")
for i, record in enumerate(data[:5]):
    print(f"Record {i}: {record}")
    print(f"Types: {type(record['product_title']), type(record['author']), type(record['resume']), type(record['labels']), type(record['image_url']), type(record['collection']), type(record['date_de_parution']), type(record['ean']), type(record['editeur']), type(record['format']), type(record['isbn']), type(record['nb_de_pages']), type(record['poids']), type(record['presentation']), type(record['width']), type(record['height']), type(record['depth'])}")

async def create_table(conn):
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
                        product_title, author, resume, labels, image_url, collection,
                        date_de_parution, ean, editeur, format, isbn, nb_de_pages,
                        poids, presentation, width,
                        height, depth
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17
                    )
                """,
                    record['product_title'], record['author'], record['resume'],
                    record['labels'], record['image_url'], record['collection'],
                    record['date_de_parution'], record['ean'], record['editeur'],
                    record['format'], record['isbn'], record['nb_de_pages'],
                    record['poids'], record['presentation'],
                    record['width'], record['height'], record['depth']
                )
            except Exception as e:
                print(f"Error inserting record: {record}")
                print(f"Error message: {e}")
                await conn.execute('ROLLBACK')

async def retrieve_data(conn):
    rows = await conn.fetch(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
    for row in rows:
        print(row)

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
