import asyncio
import asyncpg
import pandas as pd
import os
import hashlib
from dotenv import load_dotenv
from pandas.io.sql import get_sqltype

load_dotenv()

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')
TABLE_NAME = os.getenv('TABLE_NAME')

def hash_row(row):
    """Generate a hash for a row."""
    row_string = row.to_json()
    return hashlib.md5(row_string.encode('utf-8')).hexdigest()

async def load_data_to_postgres():
    df = pd.read_parquet('data/cleaned_data.parquet')

    df['row_hash'] = df.apply(hash_row, axis=1)

    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB
    )

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        row_hash TEXT PRIMARY KEY,
        {', '.join(f"{col} {get_sqltype(df[col].dtype)}" for col in df.columns if col != 'row_hash')}
    );
    """
    await conn.execute(create_table_query)

    existing_hashes = await conn.fetch(f'SELECT row_hash FROM {TABLE_NAME};')
    existing_hashes_set = {row['row_hash'] for row in existing_hashes}

    new_rows = df[~df['row_hash'].isin(existing_hashes_set)]

    if not new_rows.empty:
        async with conn.transaction():
            await conn.executemany(
                f"INSERT INTO {TABLE_NAME} VALUES ({', '.join(['$' + str(i+1) for i in range(len(new_rows.columns))])}) "
                "ON CONFLICT (row_hash) DO NOTHING",
                new_rows.values.tolist()
            )
        print(
            f"Data successfully loaded into table '{TABLE_NAME}' in the '{POSTGRES_DB}' database.")
    else:
        print("No new data to load.")

    await conn.close()

if __name__ == "__main__":
    asyncio.run(load_data_to_postgres())
