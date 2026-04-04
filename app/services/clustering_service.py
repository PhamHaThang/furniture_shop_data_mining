from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.core.schemas import MLRequest
from app.services.data_prep_service import products_to_df
from app.utils.normalizers import to_id


""""
K-Means Clustering
+ Phân cụm sản phẩm: Dựa trên các đặc điểm như giá cả, đánh giá trung bình, số lượng đánh giá, số lượng đã bán, tồn kho, số lượng hình ảnh và có hỗ trợ 3D hay không.
+ Phân cụm người dùng: Dựa trên hành vi mua sắm và đánh giá của người dùng, bao gồm tổng số đơn hàng, tổng số lượng đã mua, tổng chi tiêu, số lượng đánh giá và đánh giá trung bình.

Output: Mỗi cụm sẽ bao gồm một danh sách các sản phẩm hoặc người dùng, cùng với các đặc điểm chính của họ và nhãn cụm tương ứng.
"""
def kmeans_clustering(request:MLRequest, cluster_type: str = "products") ->Dict[str, Any]:
    if cluster_type == "products":
        # cluster_type == "products"
        # Chuẩn bị dữ liệu sản phẩm
        df = products_to_df(request.products)
        if df.empty:
           return {"cluster_type": "products", "clusters": []}
        
        # Các đặc trưng để phân cụm sản phẩm
        feature_cols = [
            "price",
            "average_rating",
            "total_reviews",
            "sold_count",
            "stock",
            "image_count",
            "has_3d"
        ]
        x = df[feature_cols].values
        labels_key = "product_id"
        names_key = "name"
    else:
        # cluster_type == "users"
        user_stats = defaultdict(
            lambda: {
                "total_orders": 0.0,
                "total_qty": 0.0,
                "total_spent": 0.0,
                "total_reviews": 0.0,
                "avg_rating_sum": 0.0,
            }
        )

        # Tính toán các đặc trưng cho mỗi user dựa trên đơn hàng: số lượng đơn hàng, tổng chi tiêu, tổng số lượng đã mua
        for order in request.orders:
            uid = to_id(order.get("user"))
            user_stats[uid]["total_orders"] += 1
            user_stats[uid]["total_spent"] += float(order.get("totalAmount") or 0)
            qty = sum(float(item.get("quantity") or 0) for item in order.get("items") or [])
            user_stats[uid]["total_qty"] += qty

        # Tính toán các đặc trưng cho mỗi user dựa trên đánh giá: số lượng đánh giá và đánh giá trung bình
        for review in request.reviews:
            uid = to_id(review.get("user"))
            user_stats[uid]["total_reviews"] += 1
            user_stats[uid]["avg_rating_sum"] += float(review.get("rating") or 0)
        
        rows = []

        for uid, stats in user_stats.items():
            # Tính toán giá trị trung bình cho đánh giá và đơn hàng
            avg_rating = stats["avg_rating_sum"] / stats["total_reviews"] if stats["total_reviews"] else 0
            avg_order = stats["total_spent"] / stats["total_orders"] if stats["total_orders"] else 0
            rows.append(
                {
                    "user_id": uid,
                    "name": uid,
                    "total_orders": stats["total_orders"],
                    "total_qty": stats["total_qty"],
                    "total_spent": stats["total_spent"],
                    "avg_order_value": avg_order,
                    "total_reviews": stats["total_reviews"],
                    "avg_rating": avg_rating,
                }
            )
        
        df = pd.DataFrame(rows)
        if df.empty:
            return {"cluster_type": "users", "clusters": []}

        feature_cols = ["total_orders", "total_qty", "total_spent", "avg_order_value", "total_reviews", "avg_rating"]
        x = df[feature_cols].values
        labels_key = "user_id"
        names_key = "name"

    n_samples = x.shape[0]
    # Logic để xác định số lượng cụm phù hợp: ít nhất 2 cụm và không vượt quá số lượng mẫu
    n_clusters = max(2, min(request.clusters, n_samples)) if n_samples > 1 else 1

    # Chuẩn hóa dữ liệu trước khi phân cụm
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Áp dụng K-Means để phân cụm
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = model.fit_predict(x_scaled)

    result = []
    for idx, row in df.iterrows():
        result.append(
            {
                "id": row[labels_key],
                "name": row[names_key],
                "cluster": int(cluster_labels[idx]),
                "features": {col: float(row[col]) for col in feature_cols},
            }
        )

    return {
        "cluster_type": cluster_type,
        "cluster_count": int(n_clusters),
        "clusters": result,
    }