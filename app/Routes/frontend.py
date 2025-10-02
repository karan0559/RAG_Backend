from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()

# Serve the main HTML file
@router.get("/", response_class=FileResponse)
async def serve_frontend():
    # Get the correct path to the HTML file
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html")
    return FileResponse(html_path)