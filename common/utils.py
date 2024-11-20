import asyncio
import asyncpg
import os
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')
TABLE_NAME = os.getenv('TABLE_NAME')


async def reconnect():
    return await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB
    )


async def execute_batch_updates(conn, updates, query):
    max_retries = 3
    retry_delay = 30
    retries = 0

    while retries < max_retries:
        try:
            await conn.executemany(query, updates)
            break
        except asyncpg.exceptions.ConnectionDoesNotExistError as e:
            retries += 1
            print(
                f"Error executing updates: {e}. Retrying ({retries}/{max_retries}) in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            conn = await reconnect()
        except asyncpg.exceptions.InterfaceError as e:
            retries += 1
            print(
                f"Error executing updates: {e}. Retrying ({retries}/{max_retries}) in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            print(f"Error executing updates: {e}")
            raise

    if retries == max_retries:
        raise RuntimeError(
            "Max retries reached. Failed to execute batch updates.")
