from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from app.core.schemas import MLRequest
from app.services.clustering_service import kmeans_clustering
from app.services.sentiment_service import run_sentiment_analysis
from app.utils.normalizers import to_id

def admin_analytics(request: MLRequest) -> Dict[str, Any]:
    # Phân tích cảm xúc
    sentiment_result = run_sentiment_analysis(request.reviews)

    # Phân cụm sản phẩm
    product_clusters = kmeans_clustering(request, cluster_type="products")

    # Phân cụm người dùng
    user_clusters = kmeans_clustering(request, cluster_type="users")

    reviews_df = pd.DataFrame(sentiment_result.get("reviews", []))
    top_products = []
    if not reviews_df.empty:
        grouped = reviews_df.groupby("product_id").agg(
            total=("label", "count"),
            good=("label", lambda x: int((x == "Good").sum())),
            bad=("label", lambda x: int((x == "Bad").sum())),
        )
        grouped["good_ratio"] = grouped["good"] / grouped["total"].replace(0, 1)
        grouped = grouped.sort_values(by=["good_ratio", "total"], ascending=[False, False]).head(10)

        names = {str(to_id(p.get("_id"))): p.get("name", "") for p in request.products}
        for product_id, row in grouped.iterrows():
            top_products.append(
                {
                    "product_id": product_id,
                    "name": names.get(product_id, ""),
                    "total_reviews": int(row["total"]),
                    "good": int(row["good"]),
                    "bad": int(row["bad"]),
                    "good_ratio": float(round(row["good_ratio"], 4)),
                }
            )

    return {
        "sentiment": sentiment_result,
        "clusters": {
            "products": product_clusters,
            "users": user_clusters,
        },
        "top_products_by_sentiment": top_products,
    }
