from fastapi import APIRouter, Query
from typing import List, Optional
from expose.models import Book
from expose.database import get_db_connection
from expose.config import TABLE_NAME

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
        False, description="Filter by format")
):
    conn = await get_db_connection()

    query = f"SELECT * FROM {TABLE_NAME} WHERE id = $1"
    book_details = await conn.fetchrow(query, book_id)

    if not book_details:
        await conn.close()
        return []

    book_embedding = book_details['embedding']

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

    print(query)
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
