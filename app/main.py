from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Vision-Language Model API")
app.include_router(router)
