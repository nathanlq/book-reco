from pydantic import BaseModel
from typing import List, Optional, Dict, Union

class Book(BaseModel):
    id: Optional[str] = None
    product_title: Optional[str] = None
    author: Optional[str] = None
    resume: Optional[str] = None
    image_url: Optional[str] = None
    collection: Optional[str] = None
    date_de_parution: Optional[str] = None
    ean: Optional[int] = None
    editeur: Optional[str] = None
    format: Optional[str] = None
    isbn: Optional[str] = None
    nb_de_pages: Optional[int] = None
    poids: Optional[float] = None
    presentation: Optional[str] = None
    width: Optional[float] = None
    height: Optional[float] = None
    depth: Optional[float] = None

class SearchRequest(BaseModel):
    nb_de_pages: Optional[Dict[str, int]] = None
    mot_clef: Optional[str] = None
    collections: Optional[List[str]] = None
    date_de_parution: Optional[Dict[str, str]] = None