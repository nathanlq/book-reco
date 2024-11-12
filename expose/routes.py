from fastapi import APIRouter, Query
from typing import List, Optional, Any
from expose.models import Book
from expose.database import get_db_connection
from expose.config import TABLE_NAME


router = APIRouter()

@router.get("/books", response_model=List[Any]) # List[Book])
async def get_books(
    id: Optional[str] = Query(None, description="Filter by ID"),
    product_title: Optional[str] = Query(None, description="Filter by product title"),
    author: Optional[str] = Query(None, description="Filter by author"),
    resume: Optional[str] = Query(None, description="Filter by resume"),
    labels: Optional[str] = Query(None, description="Filter by labels (comma-separated)"),
    image_url: Optional[str] = Query(None, description="Filter by image URL"),
    collection: Optional[str] = Query(None, description="Filter by collection"),
    date_de_parution: Optional[int] = Query(None, description="Filter by date de parution"),
    ean: Optional[int] = Query(None, description="Filter by EAN"),
    editeur: Optional[str] = Query(None, description="Filter by editor"),
    format: Optional[str] = Query(None, description="Filter by format"),
    isbn: Optional[str] = Query(None, description="Filter by ISBN"),
    nb_de_pages: Optional[int] = Query(None, description="Filter by number of pages"),
    poids: Optional[float] = Query(None, description="Filter by weight"),
    presentation: Optional[str] = Query(None, description="Filter by presentation"),
    width: Optional[float] = Query(None, description="Filter by width"),
    height: Optional[float] = Query(None, description="Filter by height"),
    depth: Optional[float] = Query(None, description="Filter by depth")
):
    conn = await get_db_connection()

    query = f"SELECT * FROM {TABLE_NAME} WHERE 1=1"
    params = []

    if id:
        query += " AND id = $1"
        params.append(id)

    if product_title:
        query += " AND product_title = $2"
        params.append(product_title)

    if author:
        query += " AND author = $3"
        params.append(author)

    if resume:
        query += " AND resume = $4"
        params.append(resume)

    if labels:
        label_list = labels.split(',')
        for idx, label in enumerate(label_list, start=len(params) + 1):
            query += f" AND labels @> $${idx}"
            params.append({label})

    if image_url:
        query += f" AND image_url = ${len(params) + 1}"
        params.append(image_url)

    if collection:
        query += f" AND collection = ${len(params) + 1}"
        params.append(collection)

    if date_de_parution:
        query += f" AND date_de_parution = ${len(params) + 1}"
        params.append(date_de_parution)

    if ean:
        query += f" AND ean = ${len(params) + 1}"
        params.append(ean)

    if editeur:
        query += f" AND editeur = ${len(params) + 1}"
        params.append(editeur)

    if format:
        query += f" AND format = ${len(params) + 1}"
        params.append(format)

    if isbn:
        query += f" AND isbn = ${len(params) + 1}"
        params.append(isbn)

    if nb_de_pages:
        query += f" AND nb_de_pages = ${len(params) + 1}"
        params.append(nb_de_pages)

    if poids:
        query += f" AND poids = ${len(params) + 1}"
        params.append(poids)

    if presentation:
        query += f" AND presentation = ${len(params) + 1}"
        params.append(presentation)

    if width:
        query += f" AND width = ${len(params) + 1}"
        params.append(width)

    if height:
        query += f" AND height = ${len(params) + 1}"
        params.append(height)

    if depth:
        query += f" AND depth = ${len(params) + 1}"
        params.append(depth)

    rows = await conn.fetch(query, *params)

    books = []
    for row in rows:
        book_data = {
            "id": row['id'],
            "product_title": row['product_title'],
            "author": row['author'],
            "resume": row['resume'],
            "labels": row['labels'],
            "image_url": row['image_url'],
            "collection": row['collection'],
            "date_de_parution": row['date_de_parution'],
            "ean": row['ean'],
            "editeur": row['editeur'],
            "format": row['format'],
            "isbn": row['isbn'],
            "nb_de_pages": row['nb_de_pages'],
            "poids": row['poids'],
            "presentation": row['presentation'],
            "width": row['width'],
            "height": row['height'],
            "depth": row['depth'],
            "embedding": row['embedding'],
            "tfidf": row['tfidf']
        }
        books.append(book_data)

    await conn.close()
    return books
