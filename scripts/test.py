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
    total_rows = await conn.fetchval(f"SELECT COUNT(*) FROM {TABLE_NAME}")

    random_offsets = [random.randint(0, total_rows - 1) for _ in range(5)]

    books = []
    for offset in random_offsets:
        book = await conn.fetchrow(f"""
            SELECT id, embedding, tfidf
            FROM {TABLE_NAME}
            OFFSET {offset}
            LIMIT 1
        """)
        if book:
            books.append(book)

    print("Randomly selected books and embedding status:")
    for book in books:
        embedding_status = "Exists" if book['embedding'] else "Missing"
        tfidf_status = "Exists" if book['tfidf'] else "Missing"
        print(f"Book ID: {book['id']} - Embedding: {embedding_status} - TF-IDF: {tfidf_status}")


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
