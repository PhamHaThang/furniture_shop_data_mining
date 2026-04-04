from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.core.schemas import MLRequest
from app.services.recommendation_service import (
    collaborative_recommendation,
    content_based_recommendation,
    hybrid_recommendation,
)
from app.services.clustering_service import kmeans_clustering

router = APIRouter()

@router.post("/recommend/content-based")
def recommend_content_based(payload: MLRequest) -> Dict[str, Any]:
    return content_based_recommendation(payload)

@router.post("/recommend/collaborative")
def recommend_collaborative(payload: MLRequest) -> Dict[str, Any]:
    return collaborative_recommendation(payload)

@router.post("/recommend/hybrid")
def recommend_hybrid(payload: MLRequest) -> Dict[str, Any]:
    return hybrid_recommendation(payload)

@router.post("/cluster/kmeans")
def run_kmeans(payload: MLRequest, cluster_type: str = "products") -> Dict[str, Any]:
    if cluster_type not in {"products", "users"}:
        raise HTTPException(status_code=400, detail="Cluster type must be 'products' or 'users'")
    return kmeans_clustering(payload, cluster_type=cluster_type)