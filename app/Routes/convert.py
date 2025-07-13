from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def convert_file():
    return {"message": "Convert endpoint is working!"}
