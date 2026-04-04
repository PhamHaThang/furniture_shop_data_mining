from fastapi import APIRouter

from .analytics import router as analytics_router
from .health import router as health_router
from .recommendations import router as recommendations_router

router = APIRouter()
router.include_router(health_router)
router.include_router(recommendations_router)
router.include_router(analytics_router)
