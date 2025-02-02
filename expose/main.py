from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from expose.routes import router

app = FastAPI()

origins = [
    "https://api.bookfarm.spacesheep.ovh",
    "https://app.bookfarm.spacesheep.ovh"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
