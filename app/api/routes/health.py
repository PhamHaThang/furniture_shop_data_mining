from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint to verify that the service is running.
    """
    return {"status": "ok"}