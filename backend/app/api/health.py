from fastapi import APIRouter, HTTPException

from app.database.mongodb import get_database

router = APIRouter(tags=["Health"])


@router.get("/")
def root():
    return {
        "message": "Smart Traffic System API is running",
        "status": "success",
    }


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Smart Traffic Backend",
    }

@router.get("/health/db")
async def db_health_check():
    try:
        await get_database().command("ping")
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"status": "healthy", "database": "connected"}