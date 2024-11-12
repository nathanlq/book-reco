from fastapi import FastAPI
from expose.routes import router


app = FastAPI()

app.include_router(router)
