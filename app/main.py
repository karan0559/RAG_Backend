from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
from fastapi.staticfiles import StaticFiles
import os

app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(compare.router, prefix="/compare", tags=["Compare"])
app.include_router(convert.router, prefix="/convert", tags=["Convert"])
app.include_router(docs.router, prefix="/docs", tags=["Documents"])
app.include_router(audio.router, prefix="/audio", tags=["Audio"])

# Ensure static directory exists and mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=FileResponse)
async def root():
    # Serve the main HTML file directly from root
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(html_path)

from app.Services import vector_db

@app.on_event("startup")
def on_startup():
    print("Initializing RAG system...")
    vector_db.load_index()
    try:
        webbrowser.open("http://127.0.0.1:8000/docs")
    except:
        print(" Could not open browser automatically.")





