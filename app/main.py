from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
import os
import webbrowser

load_dotenv()

app = FastAPI(
    title="Smart RAG System",
    description="Multimodal document understanding + RAG-powered assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.Routes import upload, query, compare, convert, docs, audio  

app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(compare.router, prefix="/compare", tags=["Compare"])
app.include_router(convert.router, prefix="/convert", tags=["Convert"])
app.include_router(docs.router, prefix="/docs", tags=["Documents"])
app.include_router(audio.router, prefix="/audio", tags=["Audio"])  

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

from app.Services import vector_db

@app.on_event("startup")
def on_startup():
    print("🚀 Initializing RAG system...")
    vector_db.load_index()
    try:
        webbrowser.open("http://127.0.0.1:8000/docs")
    except:
        print("⚠️ Could not open browser automatically.")
