import asyncio
import asyncpg
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')
TABLE_NAME = os.getenv('TABLE_NAME')


def calculate_embedding(record):
    return np.ones(6244).tolist()


async def update_embeddings_for_new_books(conn):
    print("Fetching rows with missing embeddings...")
    rows = await conn.fetch(f"SELECT * FROM {TABLE_NAME} WHERE embedding IS NULL")
    if not rows:
        print("Nothing to calculate.")
        return
    for row in rows:
        embedding = calculate_embedding(row)

        embedding_str = '[' + ','.join(map(str, embedding)) + ']'

        try:
            await conn.execute(
                f"UPDATE {TABLE_NAME} SET embedding = $1 WHERE id = $2",
                embedding_str, row['id']
            )
        except Exception as e:
            print(f"Error updating embedding for book ID {row['id']}: {e}")


async def main():
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB
    )

    try:
        while True:
            await update_embeddings_for_new_books(conn)
            await asyncio.sleep(60)
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
