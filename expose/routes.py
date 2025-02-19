import aiohttp
import os
import json
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from fastapi.responses import FileResponse
from expose.models import Book, SearchRequest
from expose.database import get_db_connection
from expose.config import TABLE_NAME
from microservices.images import generate_image_path, download_and_save_image_webp

router = APIRouter()


@router.get("/books", response_model=List[Book])
async def get_books(
    id: Optional[str] = Query(None, description="Filter by ID"),
    product_title: Optional[str] = Query(
        None, description="Filter by product title"),
    author: Optional[str] = Query(None, description="Filter by author"),
    resume: Optional[str] = Query(None, description="Filter by resume"),
    image_url: Optional[str] = Query(None, description="Filter by image URL"),
    collection: Optional[str] = Query(
        None, description="Filter by collection"),
    date_de_parution: Optional[int] = Query(
        None, description="Filter by date de parution"),
    ean: Optional[int] = Query(None, description="Filter by EAN"),
    editeur: Optional[str] = Query(None, description="Filter by editor"),
    format: Optional[str] = Query(None, description="Filter by format"),
    isbn: Optional[str] = Query(None, description="Filter by ISBN"),
    nb_de_pages: Optional[int] = Query(
        None, description="Filter by number of pages"),
    poids: Optional[float] = Query(None, description="Filter by weight"),
    presentation: Optional[str] = Query(
        None, description="Filter by presentation"),
    width: Optional[float] = Query(None, description="Filter by width"),
    height: Optional[float] = Query(None, description="Filter by height"),
    depth: Optional[float] = Query(None, description="Filter by depth"),
    page: Optional[int] = Query(1, description="Page number"),
    page_size: Optional[int] = Query(
        10, description="Number of items per page")
):
    conn = await get_db_connection()

    query = f"SELECT * FROM {TABLE_NAME}"
    params = []
    conditions = []

    filters = {
        "id": id,
        "product_title": product_title,
        "author": author,
        "resume": resume,
        "image_url": image_url,
        "collection": collection,
        "date_de_parution": date_de_parution,
        "ean": ean,
        "editeur": editeur,
        "format": format,
        "isbn": isbn,
        "nb_de_pages": nb_de_pages,
        "poids": poids,
        "presentation": presentation,
        "width": width,
        "height": height,
        "depth": depth,
    }

    for column, value in filters.items():
        if value is not None:
            conditions.append(f"{column} = ${len(params) + 1}")
            params.append(value)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    offset = (page - 1) * page_size
    query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
    params.append(page_size)
    params.append(offset)

    print(f"Execute query : {query}")

    rows = await conn.fetch(query, *params)

    books = []
    for row in rows:
        book_data = {
            "id": row['id'],
            "product_title": row.get('product_title'),
            "author": row.get('author'),
            "resume": row.get('resume'),
            "image_url": row.get('image_url'),
            "collection": row.get('collection'),
            "date_de_parution": str(row['date_de_parution']) if row.get('date_de_parution') is not None else '',
            "ean": row.get('ean'),
            "editeur": row.get('editeur'),
            "format": row.get('format'),
            "isbn": row.get('isbn'),
            "nb_de_pages": row.get('nb_de_pages'),
            "poids": float(row['poids']) if row.get('poids') is not None and -1e308 < float(row['poids']) < 1e308 else None,
            "presentation": row.get('presentation'),
            "width": float(row['width']) if row.get('width') is not None and -1e308 < float(row['width']) < 1e308 else None,
            "height": float(row['height']) if row.get('height') is not None and -1e308 < float(row['height']) < 1e308 else None,
            "depth": float(row['depth']) if row.get('depth') is not None and -1e308 < float(row['depth']) < 1e308 else None,
        }
        books.append(book_data)

    await conn.close()
    return books


@router.get("/books/{book_id}/similar", response_model=List[Book])
async def get_similar_books(
    book_id: str,
    method: Optional[str] = Query(
        "cosine", description="Method for finding similar books (taxicab, cosine, euclidean)"),
    author: Optional[bool] = Query(
        False, description="Filter by author"),
    collection: Optional[bool] = Query(
        False, description="Filter by collection"),
    editeur: Optional[bool] = Query(
        False, description="Filter by editor"),
    format: Optional[bool] = Query(
        False, description="Filter by format"),
    fast: Optional[bool] = Query(
        False, description="Search only within the same cluster")
):
    conn = await get_db_connection()

    query = f"SELECT *, utils->>'dynamic_cluster_number' as dynamic_cluster_number FROM {TABLE_NAME} WHERE id = $1"
    book_details = await conn.fetchrow(query, book_id)

    if not book_details:
        await conn.close()
        return []

    book_embedding = book_details['embedding']
    cluster_label = book_details.get('dynamic_cluster_number')

    filters = {
        "author": author,
        "collection": collection,
        "editeur": editeur,
        "format": format,
    }

    conditions = []
    params = [book_embedding]

    for column, value in filters.items():
        if value:
            conditions.append(f"{column} = ${len(params) + 1}")
            params.append(book_details[column])

    if fast and cluster_label is not None:
        conditions.append(
            f"utils->>'dynamic_cluster_number' = ${len(params) + 1}")
        params.append(cluster_label)

    base_query = f"SELECT * FROM {TABLE_NAME}"
    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    if method == "euclidean":
        query = f"{base_query} ORDER BY embedding <-> $1 LIMIT 5"
    elif method == "cosine":
        query = f"{base_query} ORDER BY embedding <=> $1 LIMIT 5"
    elif method == "taxicab":
        query = f"{base_query} ORDER BY embedding <+> $1 LIMIT 5"
    else:
        await conn.close()
        return []

    rows = await conn.fetch(query, *params)

    similar_books = []
    for row in rows:
        if row['id'] != book_id:
            book_data = {
                "id": row['id'],
                "product_title": row.get('product_title'),
                "author": row.get('author'),
                "resume": row.get('resume'),
                "image_url": row.get('image_url'),
                "collection": row.get('collection'),
                "date_de_parution": str(row['date_de_parution']) if row.get('date_de_parution') is not None else '',
                "ean": row.get('ean'),
                "editeur": row.get('editeur'),
                "format": row.get('format'),
                "isbn": row.get('isbn'),
                "nb_de_pages": row.get('nb_de_pages'),
                "poids": float(row['poids']) if row.get('poids') is not None and -1e308 < float(row['poids']) < 1e308 else None,
                "presentation": row.get('presentation'),
                "width": float(row['width']) if row.get('width') is not None and -1e308 < float(row['width']) < 1e308 else None,
                "height": float(row['height']) if row.get('height') is not None and -1e308 < float(row['height']) < 1e308 else None,
                "depth": float(row['depth']) if row.get('depth') is not None and -1e308 < float(row['depth']) < 1e308 else None,
            }
            similar_books.append(book_data)

    await conn.close()
    return similar_books


@router.get("/books/{book_id}/image", response_class=FileResponse)
async def get_book_image(book_id: str):
    conn = await get_db_connection()

    query = f"SELECT id, image_url, utils FROM {TABLE_NAME} WHERE id = $1"
    book_details = await conn.fetchrow(query, book_id)

    if not book_details:
        await conn.close()
        raise HTTPException(status_code=404, detail="Book not found")

    image_url = book_details['image_url']
    utils = book_details['utils']
    utils = json.loads(utils)
    image_downloaded = utils.get('image_downloaded', False)

    image_path = generate_image_path(image_url)

    if not os.path.exists(image_path) or not image_downloaded:
        async with aiohttp.ClientSession() as session:
            image_path = await download_and_save_image_webp(session, image_url, image_path)
            if image_path:
                await conn.execute(
                    f"UPDATE {TABLE_NAME} SET utils = jsonb_set(utils, '{{image_downloaded}}', 'true') WHERE id = $1",
                    book_id
                )

    if not os.path.exists(image_path):
        await conn.close()
        raise HTTPException(status_code=404, detail="Image not found")

    await conn.close()
    return FileResponse(image_path)


@router.post("/search", response_model=List[Book])
async def search_books(search_request: SearchRequest):
    conn = await get_db_connection()

    conditions = []
    params = []

    if search_request.nb_de_pages:
        if 'min' in search_request.nb_de_pages:
            conditions.append(f"nb_de_pages >= ${len(params) + 1}")
            params.append(search_request.nb_de_pages['min'])
        if 'max' in search_request.nb_de_pages:
            conditions.append(f"nb_de_pages <= ${len(params) + 1}")
            params.append(search_request.nb_de_pages['max'])

    if search_request.mot_clef:
        keyword = search_request.mot_clef
        keyword_for_author = keyword.replace(" ", "_")

        keyword_conditions = [
            f"product_title LIKE ${len(params) + 1}",
            f"author LIKE ${len(params) + 2}",
            f"resume LIKE ${len(params) + 1}"
        ]
        
        params.append(f"%{keyword}%")
        params.append(f"%{keyword_for_author}%")

        conditions.append(f"({' OR '.join(keyword_conditions)})")

    if search_request.collections:
        conditions.append(f"collection = ANY(${len(params) + 1})")
        params.append(search_request.collections)

    if search_request.date_de_parution:
        if 'après' in search_request.date_de_parution:
            conditions.append(f"EXTRACT(YEAR FROM date_de_parution) >= ${len(params) + 1}")
            params.append(search_request.date_de_parution['après'])
        if 'avant' in search_request.date_de_parution:
            conditions.append(f"EXTRACT(YEAR FROM date_de_parution) <= ${len(params) + 1}")
            params.append(search_request.date_de_parution['avant'])

    query = f"SELECT * FROM {TABLE_NAME}"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " LIMIT 100"
    rows = await conn.fetch(query, *params)

    books = []
    for row in rows:
        book_data = {
            "id": row['id'],
            "product_title": row.get('product_title'),
            "author": row.get('author'),
            "resume": row.get('resume'),
            "image_url": row.get('image_url'),
            "collection": row.get('collection'),
            "date_de_parution": str(row['date_de_parution']) if row.get('date_de_parution') is not None else '',
            "ean": row.get('ean'),
            "editeur": row.get('editeur'),
            "format": row.get('format'),
            "isbn": row.get('isbn'),
            "nb_de_pages": row.get('nb_de_pages'),
            "poids": float(row['poids']) if row.get('poids') is not None and -1e308 < float(row['poids']) < 1e308 else None,
            "presentation": row.get('presentation'),
            "width": float(row['width']) if row.get('width') is not None and -1e308 < float(row['width']) < 1e308 else None,
            "height": float(row['height']) if row.get('height') is not None and -1e308 < float(row['height']) < 1e308 else None,
            "depth": float(row['depth']) if row.get('depth') is not None and -1e308 < float(row['depth']) < 1e308 else None,
        }
        books.append(book_data)

    await conn.close()
    return books


@router.get("/collections", response_model=List[str])
async def get_collections():
    conn = await get_db_connection()

    query = f"SELECT DISTINCT collection FROM {TABLE_NAME} WHERE collection IS NOT NULL"
    rows = await conn.fetch(query)

    collections = [row['collection'] for row in rows]

    await conn.close()
    return collections


@router.get("/book/{book_id}", response_model=Book)
async def get_book(book_id: str):
    conn = await get_db_connection()

    query = f"SELECT * FROM {TABLE_NAME} WHERE id = $1"
    row = await conn.fetchrow(query, book_id)

    if not row:
        raise HTTPException(status_code=404, detail="Book not found")

    book_data = {
        "id": row['id'],
        "product_title": row.get('product_title'),
        "author": row.get('author'),
        "resume": row.get('resume'),
        "image_url": row.get('image_url'),
        "collection": row.get('collection'),
        "date_de_parution": str(row['date_de_parution']) if row.get('date_de_parution') is not None else '',
        "ean": row.get('ean'),
        "editeur": row.get('editeur'),
        "format": row.get('format'),
        "isbn": row.get('isbn'),
        "nb_de_pages": row.get('nb_de_pages'),
        "poids": float(row['poids']) if row.get('poids') is not None and -1e308 < float(row['poids']) < 1e308 else None,
        "presentation": row.get('presentation'),
        "width": float(row['width']) if row.get('width') is not None and -1e308 < float(row['width']) < 1e308 else None,
        "height": float(row['height']) if row.get('height') is not None and -1e308 < float(row['height']) < 1e308 else None,
        "depth": float(row['depth']) if row.get('depth') is not None and -1e308 < float(row['depth']) < 1e308 else None,
    }

    await conn.close()
    return book_data
