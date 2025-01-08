import asyncio
import asyncpg
import os
from dotenv import load_dotenv
import random

load_dotenv()

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')
TABLE_NAME = os.getenv('TABLE_NAME')

async def check_random_books(conn):
    books = await conn.fetch(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
    for book in books:
        embedding_status = "Exists" if book['embedding'] else "Missing"
        tfidf_status = "Exists" if book['tfidf'] else "Missing"

        utils_status = "Missing"
        if book['utils']:
            utils_keys = book['utils'].keys()
            if 'image_downloaded' in utils_keys:
                utils_status = "Exists"

        print(f"Book ID: {book['id']} - Embedding: {embedding_status} - TF-IDF: {tfidf_status} - Utils: {utils_status}")
        print(book)
    print("Retrieve OK.")

async def main():
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB
    )

    await check_random_books(conn)
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
