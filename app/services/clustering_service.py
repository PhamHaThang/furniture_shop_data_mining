from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.core.schemas import MLRequest
from app.services.data_prep_service import products_to_df
from app.utils.normalizers import to_id
from app.core.cluster_label_config import (
    PRODUCT_DEFAULT_LABEL,
    PRODUCT_DEFAULT_SUFFIX,
    PRODUCT_LABEL_RULES,
    PRODUCT_SUFFIX_RULES,
    USER_DEFAULT_LABEL,
    USER_DEFAULT_SUFFIX,
    USER_LABEL_RULES,
    USER_SUFFIX_RULES,
)
def _safe_quantile(df: pd.DataFrame, col: str, q: float, default: float = 0.0) -> float:
    if col not in df.columns or df.empty:
        return default
    value = df[col].quantile(q)
    if pd.isna(value):
        return default
    return float(value)


def _compare(left: float, op: str, right: float) -> bool:
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == "==":
        return left == right
    return False


def _conditions_match(
    cluster_stats: Dict[str, float],
    thresholds: Dict[str, float],
    conditions: List[tuple],
) -> bool:
    for feature_name, operator, threshold_key in conditions:
        left = float(cluster_stats.get(feature_name, 0.0))
        right = float(thresholds.get(threshold_key, 0.0))
        if not _compare(left, operator, right):
            return False
    return True


def _pick_label_from_rules(
    cluster_stats: Dict[str, float],
    thresholds: Dict[str, float],
    rules: List[Dict[str, Any]],
    default_label: str,
) -> str:
    for rule in rules:
        conditions = rule.get("all", [])
        if _conditions_match(cluster_stats, thresholds, conditions):
            return str(rule.get("label") or default_label)
    return default_label


def _pick_suffix_from_rules(
    cluster_stats: Dict[str, float],
    thresholds: Dict[str, float],
    rules: List[Dict[str, Any]],
    default_suffix: str,
) -> str:
    for rule in rules:
        conditions = rule.get("all", [])
        if _conditions_match(cluster_stats, thresholds, conditions):
            return str(rule.get("suffix") or default_suffix)
    return default_suffix


def _label_product_cluster(cluster_stats: Dict[str, float], thresholds: Dict[str, float]) -> str:
    return _pick_label_from_rules(
        cluster_stats,
        thresholds,
        PRODUCT_LABEL_RULES,
        PRODUCT_DEFAULT_LABEL,
    )


def _label_user_cluster(cluster_stats: Dict[str, float], thresholds: Dict[str, float]) -> str:
    return _pick_label_from_rules(
        cluster_stats,
        thresholds,
        USER_LABEL_RULES,
        USER_DEFAULT_LABEL,
    )


def _product_label_suffix(cluster_stats: Dict[str, float], thresholds: Dict[str, float]) -> str:
    return _pick_suffix_from_rules(
        cluster_stats,
        thresholds,
        PRODUCT_SUFFIX_RULES,
        PRODUCT_DEFAULT_SUFFIX,
    )


def _user_label_suffix(cluster_stats: Dict[str, float], thresholds: Dict[str, float]) -> str:
    return _pick_suffix_from_rules(
        cluster_stats,
        thresholds,
        USER_SUFFIX_RULES,
        USER_DEFAULT_SUFFIX,
    )


def _make_cluster_labels_unique(
    summaries: Dict[int, Dict[str, Any]],
    cluster_type: str,
    thresholds: Dict[str, float],
) -> None:
    label_to_clusters: Dict[str, List[int]] = defaultdict(list)
    for cluster_id, summary in summaries.items():
        label_to_clusters[summary.get("label", "")].append(cluster_id)

    for label, cluster_ids in label_to_clusters.items():
        if len(cluster_ids) <= 1:
            continue

        for cluster_id in cluster_ids:
            stats = summaries[cluster_id].get("avg_features", {})
            suffix = (
                _product_label_suffix(stats, thresholds)
                if cluster_type == "products"
                else _user_label_suffix(stats, thresholds)
            )
            summaries[cluster_id]["label"] = f"{label} - {suffix}"

    # Nếu vẫn trùng sau khi thêm suffix thì thêm số thứ tự để đảm bảo unique tuyệt đối.
    seen_counts: Dict[str, int] = defaultdict(int)
    for cluster_id in sorted(summaries.keys()):
        label = summaries[cluster_id].get("label", "")
        seen_counts[label] += 1
        if seen_counts[label] > 1:
            summaries[cluster_id]["label"] = f"{label} #{seen_counts[label]}"


def _build_cluster_summaries(
    df: pd.DataFrame,
    cluster_labels: List[int],
    feature_cols: List[str],
    cluster_type: str,
) -> Dict[int, Dict[str, Any]]:
    cluster_df = df.copy()
    cluster_df["cluster"] = cluster_labels

    thresholds = {}
    if cluster_type == "products":
        thresholds = {
            "price_p25": _safe_quantile(df, "price", 0.25),
            "price_p75": _safe_quantile(df, "price", 0.75),
            "rating_p75": _safe_quantile(df, "average_rating", 0.75),
            "sold_p25": _safe_quantile(df, "sold_count", 0.25),
            "sold_p50": _safe_quantile(df, "sold_count", 0.5),
            "sold_p75": _safe_quantile(df, "sold_count", 0.75),
            "reviews_p75": _safe_quantile(df, "total_reviews", 0.75),
            "stock_p75": _safe_quantile(df, "stock", 0.75),
        }
    else:
        thresholds = {
            "spent_p75": _safe_quantile(df, "total_spent", 0.75),
            "orders_p50": _safe_quantile(df, "total_orders", 0.5),
            "orders_p75": _safe_quantile(df, "total_orders", 0.75),
            "avg_order_p50": _safe_quantile(df, "avg_order_value", 0.5),
            "avg_order_p75": _safe_quantile(df, "avg_order_value", 0.75),
            "reviews_p75": _safe_quantile(df, "total_reviews", 0.75),
        }

    summaries: Dict[int, Dict[str, Any]] = {}
    grouped = cluster_df.groupby("cluster")

    for cluster_id, cluster_rows in grouped:
        means = {
            col: float(cluster_rows[col].mean())
            for col in feature_cols
            if col in cluster_rows.columns
        }

        label = (
            _label_product_cluster(means, thresholds)
            if cluster_type == "products"
            else _label_user_cluster(means, thresholds)
        )

        summaries[int(cluster_id)] = {
            "cluster": int(cluster_id),
            "label": label,
            "size": int(len(cluster_rows)),
            "avg_features": means,
        }

    _make_cluster_labels_unique(summaries, cluster_type, thresholds)

    return summaries

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
    cluster_summaries = _build_cluster_summaries(
        df=df,
        cluster_labels=cluster_labels,
        feature_cols=feature_cols,
        cluster_type=cluster_type,
    )
    result = []
    for idx, row in df.iterrows():
        cluster_id = int(cluster_labels[idx])
        cluster_info = cluster_summaries.get(cluster_id, {})
        result.append(
            {
                "id": row[labels_key],
                "name": row[names_key],
                "cluster": cluster_id,
                "cluster_label": cluster_info.get("label", ""),
                "features": {col: float(row[col]) for col in feature_cols},
            }
        )

    return {
        "cluster_type": cluster_type,
        "cluster_count": int(n_clusters),
        "cluster_summaries": sorted(
            cluster_summaries.values(),
            key=lambda item: item["cluster"],
        ),
        "clusters": result,
    }