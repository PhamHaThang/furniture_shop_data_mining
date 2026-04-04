from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class MLRequest(BaseModel):
    """
    Schema for incoming ML prediction requests.
    """
    users: List[Dict[str, Any]] = Field(default_factory=list)
    products: List[Dict[str, Any]] = Field(default_factory=list)
    reviews: List[Dict[str, Any]] = Field(default_factory=list)
    orders: List[Dict[str, Any]] = Field(default_factory=list)
    target_user_id: Optional[str] = None
    target_product_id: Optional[str] = None
    top_k: int = 8
    clusters: int = 4