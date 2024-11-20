# images.py
import asyncio
import aiohttp
import os
import hashlib
import json
import asyncpg
from PIL import Image
from io import BytesIO
from tqdm.asyncio import tqdm
from common.utils import reconnect, execute_batch_updates, TABLE_NAME

IMAGE_DIR = 'data/img'

os.makedirs(IMAGE_DIR, exist_ok=True)

async def fetch_image(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.read()
        else:
            print(f"Failed to fetch image from {url}")
            return None

def standardize_image(image_data):
    image = Image.open(BytesIO(image_data))
    image = image.convert("RGB")
    image.thumbnail((256, 256))
    new_image = Image.new("RGB", (256, 256), (255, 255, 255))
    new_image.paste(image, ((256 - image.width) //
                    2, (256 - image.height) // 2))
    return new_image

def generate_image_path(url):
    reversed_url = url[::-1]
    hash_object = hashlib.sha256(reversed_url.encode())
    hex_dig = hash_object.hexdigest()
    return os.path.join(IMAGE_DIR, f"{hex_dig}.webp")

async def download_and_save_image_webp(session, url, image_path):
    image_data = await fetch_image(session, url)
    if image_data:
        image = standardize_image(image_data)
        image.save(image_path, format="WEBP", quality=85)
        return image_path
    return None

async def process_row(session, conn, row):
    url = row['image_url']
    image_path = generate_image_path(url)
    if not os.path.exists(image_path):
        image_path = await download_and_save_image_webp(session, url, image_path)
        if image_path:
            await execute_batch_updates(conn, [(row['id'],)], f"UPDATE {TABLE_NAME} SET utils = jsonb_set(utils, '{{image_downloaded}}', 'true') WHERE id = $1")
    else:
        await execute_batch_updates(conn, [(row['id'],)], f"UPDATE {TABLE_NAME} SET utils = jsonb_set(utils, '{{image_downloaded}}', 'true') WHERE id = $1")

async def fetch_rows_to_process(conn):
    query = f"SELECT id, image_url, utils FROM {TABLE_NAME} WHERE image_url IS NOT NULL AND (utils->>'image_downloaded' IS NULL OR utils->>'image_downloaded' = 'false')"
    return await conn.fetch(query)

async def hourly_image_download_task(conn, lock):
    while True:
        print("Starting hourly image download task...")
        await process_images(conn)
        print("Hourly image download task complete. Waiting for the next run...")
        await asyncio.sleep(3600)

async def process_images(conn):
    async with aiohttp.ClientSession() as session:
        rows = await fetch_rows_to_process(conn)
        for row in tqdm(rows, desc="Processing rows", unit="row"):
            await process_row(session, conn, row)

async def main():
    while True:
        try:
            conn = await reconnect()

            lock = asyncio.Lock()

            try:
                await process_images(conn)

                await asyncio.gather(
                    hourly_image_download_task(conn, lock)
                )
            finally:
                await conn.close()
        except asyncpg.exceptions.InterfaceError as e:
            print(f"Connection error: {e}. Retrying in 30 seconds...")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
