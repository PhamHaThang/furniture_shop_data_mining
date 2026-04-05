from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from app.core.schemas import MLRequest
from app.services.clustering_service import kmeans_clustering
from app.services.sentiment_service import run_sentiment_analysis
from app.utils.normalizers import to_id

def _build_orders_analytics(request: MLRequest) -> Dict[str, Any]:
    product_names = {str(to_id(p.get("_id"))): p.get("name", "") for p in request.products}
    rows = []
    item_rows = []

    for order in request.orders:
        items = order.get("items") or []
        total_items = sum(float(item.get("quantity") or 0) for item in items)
        total_amount = float(order.get("totalAmount") or 0)
        rows.append(
            {
                "order_id": to_id(order.get("_id")),
                "user_id": to_id(order.get("user")),
                "status": str(order.get("status") or "pending"),
                "total_amount": total_amount,
                "total_items": total_items,
                "created_at": order.get("createdAt"),
            }
        )

        for item in items:
            product_id = to_id(item.get("product"))
            qty = float(item.get("quantity") or 0)
            item_rows.append(
                {
                    "product_id": product_id,
                    "name": product_names.get(product_id, ""),
                    "quantity": qty,
                    "revenue": float(item.get("price") or 0) * qty,
                }
            )

    orders_df = pd.DataFrame(rows)
    if orders_df.empty:
        return {
            "summary": {
                "total_orders": 0,
                "total_revenue": 0.0,
                "avg_order_value": 0.0,
                "total_items_sold": 0.0,
            },
            "status_distribution": [],
            "trend": [],
            "top_products": [],
        }

    orders_df["created_at"] = pd.to_datetime(orders_df["created_at"], errors="coerce")
    orders_df["date"] = orders_df["created_at"].dt.strftime("%Y-%m-%d")

    total_orders = int(len(orders_df))
    total_revenue = float(orders_df["total_amount"].sum())
    avg_order_value = float(total_revenue / total_orders) if total_orders else 0.0
    total_items_sold = float(orders_df["total_items"].sum())

    status_group = (
        orders_df.groupby("status")["order_id"]
        .count()
        .sort_values(ascending=False)
        .to_dict()
    )
    status_distribution = [
        {
            "status": status,
            "count": int(count),
            "ratio": float(round(count / total_orders, 4)) if total_orders else 0.0,
        }
        for status, count in status_group.items()
    ]

    trend_df = (
        orders_df.dropna(subset=["date"])
        .groupby("date")
        .agg(
            orders=("order_id", "count"),
            revenue=("total_amount", "sum"),
            items=("total_items", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )
    trend = [
        {
            "date": row["date"],
            "orders": int(row["orders"]),
            "revenue": float(round(row["revenue"], 2)),
            "items": float(round(row["items"], 2)),
        }
        for _, row in trend_df.iterrows()
    ]

    items_df = pd.DataFrame(item_rows)
    top_products = []
    if not items_df.empty:
        top_products_df = (
            items_df.groupby(["product_id", "name"], dropna=False)
            .agg(
                quantity=("quantity", "sum"),
                revenue=("revenue", "sum"),
            )
            .reset_index()
            .sort_values(by=["quantity", "revenue"], ascending=[False, False])
            .head(10)
        )

        for _, row in top_products_df.iterrows():
            top_products.append(
                {
                    "product_id": str(row["product_id"]),
                    "name": str(row["name"] or ""),
                    "quantity": float(round(row["quantity"], 2)),
                    "revenue": float(round(row["revenue"], 2)),
                }
            )

    return {
        "summary": {
            "total_orders": total_orders,
            "total_revenue": float(round(total_revenue, 2)),
            "avg_order_value": float(round(avg_order_value, 2)),
            "total_items_sold": float(round(total_items_sold, 2)),
        },
        "status_distribution": status_distribution,
        "trend": trend,
        "top_products": top_products,
    }


def _build_reviews_analytics(request: MLRequest, sentiment: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    rating_counter = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    sentiment_rows = sentiment.get("reviews", [])
    sentiment_map = {str(item.get("review_id")): item for item in sentiment_rows}

    for review in request.reviews:
        review_id = to_id(review.get("_id"))
        rating = int(float(review.get("rating") or 0))
        rating = max(1, min(5, rating))
        rating_counter[rating] += 1
        sentiment_row = sentiment_map.get(review_id, {})

        rows.append(
            {
                "review_id": review_id,
                "product_id": to_id(review.get("product")),
                "user_id": to_id(review.get("user")),
                "rating": rating,
                "label": sentiment_row.get("label", "Bad"),
                "created_at": review.get("createdAt"),
            }
        )

    reviews_df = pd.DataFrame(rows)
    if reviews_df.empty:
        return {
            "summary": {
                "total_reviews": 0,
                "avg_rating": 0.0,
            },
            "rating_distribution": [],
            "trend": [],
            "top_products": [],
        }

    total_reviews = int(len(reviews_df))
    avg_rating = float(reviews_df["rating"].mean()) if total_reviews else 0.0

    rating_distribution = [
        {
            "rating": rating,
            "count": int(count),
            "ratio": float(round(count / total_reviews, 4)) if total_reviews else 0.0,
        }
        for rating, count in sorted(rating_counter.items())
    ]

    reviews_df["created_at"] = pd.to_datetime(reviews_df["created_at"], errors="coerce")
    reviews_df["date"] = reviews_df["created_at"].dt.strftime("%Y-%m-%d")

    trend_df = (
        reviews_df.dropna(subset=["date"])
        .groupby("date")
        .agg(
            reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            good=("label", lambda x: int((x == "Good").sum())),
            bad=("label", lambda x: int((x == "Bad").sum())),
        )
        .reset_index()
        .sort_values("date")
    )
    trend = [
        {
            "date": row["date"],
            "reviews": int(row["reviews"]),
            "avg_rating": float(round(row["avg_rating"], 3)),
            "good": int(row["good"]),
            "bad": int(row["bad"]),
        }
        for _, row in trend_df.iterrows()
    ]

    product_names = {str(to_id(p.get("_id"))): p.get("name", "") for p in request.products}
    top_products_df = (
        reviews_df.groupby("product_id")
        .agg(
            total_reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            good=("label", lambda x: int((x == "Good").sum())),
            bad=("label", lambda x: int((x == "Bad").sum())),
        )
        .reset_index()
        .sort_values(by=["total_reviews", "avg_rating"], ascending=[False, False])
        .head(10)
    )
    top_products = [
        {
            "product_id": str(row["product_id"]),
            "name": product_names.get(str(row["product_id"]), ""),
            "total_reviews": int(row["total_reviews"]),
            "avg_rating": float(round(row["avg_rating"], 3)),
            "good": int(row["good"]),
            "bad": int(row["bad"]),
        }
        for _, row in top_products_df.iterrows()
    ]

    return {
        "summary": {
            "total_reviews": total_reviews,
            "avg_rating": float(round(avg_rating, 3)),
        },
        "rating_distribution": rating_distribution,
        "trend": trend,
        "top_products": top_products,
    }
def admin_analytics(request: MLRequest) -> Dict[str, Any]:
    # Phân tích cảm xúc
    sentiment_result = run_sentiment_analysis(request.reviews)

    # Phân cụm sản phẩm
    product_clusters = kmeans_clustering(request, cluster_type="products")

    # Phân cụm người dùng
    user_clusters = kmeans_clustering(request, cluster_type="users")

    # Xây dựng phân tích đơn hàng và đánh giá
    orders = _build_orders_analytics(request)
    reviews = _build_reviews_analytics(request, sentiment_result)

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
        "orders": orders,
        "reviews": reviews,
        "top_products_by_sentiment": top_products,
    }
