from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from app.core.schemas import MLRequest
from app.services.analytics_service import admin_analytics
from app.services.sentiment_service import run_sentiment_analysis

router = APIRouter()

@router.post("/sentiment/reviews")
def sentiment_reviews(payload: MLRequest) -> Dict[str, Any]:
    return run_sentiment_analysis(payload.reviews)


@router.post("/analytics/admin")
def analytics_for_admin(payload: MLRequest) -> Dict[str, Any]:
    return admin_analytics(payload)
