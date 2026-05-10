from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv(override=True)

from routers import assessment, analysis, content

app = FastAPI(title="EchoFlow API", version="1.0.0")

default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://slai6-afk.github.io",
]
extra_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
allowed_origins = list(dict.fromkeys(default_origins + extra_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assessment.router, prefix="/api/assessment")
app.include_router(analysis.router, prefix="/api/analysis")
app.include_router(content.router, prefix="/api/content")


@app.get("/health")
def health():
    return {"status": "ok"}
