"""
Chuẩn bị dữ liệu cho mô hình học máy bằng cách chuyển đổi các sản phẩm và tương tác người dùng thành DataFrame của pandas. 
Các hàm này giúp chuẩn hóa và tổ chức dữ liệu để dễ dàng sử dụng trong các bước tiếp theo của quy trình học máy, như huấn luyện mô hình hoặc tạo đề xuất.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

from app.utils.normalizers import join_tokens, to_id

def products_to_df(products:List[Dict[str, Any]]) ->pd.DataFrame:
    """
    Chuyển đổi danh sách sản phẩm thành một DataFrame của pandas, bao gồm các trường như product_id, name, price, average_rating, total_reviews, sold_count, stock, image_count, has_3d và corpus.
    Trường corpus là một chuỗi kết hợp các thông tin văn bản liên quan đến sản phẩm, được sử dụng để tạo đặc trưng cho mô hình học máy. 
    Các trường khác cung cấp thông tin định lượng về sản phẩm.
    """
    rows = []
    for product in products:
        product_id = to_id(product.get("_id"))
        category =  product.get("category") or {}
        brand = product.get("brand") or {}
        category_name = category.get("name") if isinstance(category, dict) else ""
        brand_name = brand.get("name") if isinstance(brand, dict) else ""
        images = product.get("images") or []

        # Tạo trường corpus bằng cách kết hợp các thông tin văn bản liên quan đến sản phẩm, bao gồm tên, mô tả, danh mục, thương hiệu, thẻ tag, màu sắc, chất liệu, số lượng hình ảnh và thông tin về mô hình 3D.
        # Đây là Feature text được sử dụng để content-based filtering.
        corpus = " ".join(
            [
                str(product.get("name", "")),
                str(product.get("description", "")),
                str(category_name or ""),
                str(brand_name or ""),
                join_tokens(product.get("tags")),
                join_tokens(product.get("colors")),
                join_tokens(product.get("materials")),
                f"image_count_{len(images)}",
                "has_3d_model" if product.get("model3DUrl") else "no_3d_model",
            ]
        )

        rows.append(
            {
                "product_id": product_id,
                "name": product.get("name", ""),
                "price": float(product.get("price") or 0),
                "average_rating": float(product.get("averageRating") or 0),
                "total_reviews": float(product.get("totalReviews") or 0),
                "sold_count": float(product.get("soldCount") or 0),
                "stock": float(product.get("stock") or 0),
                "image_count": float(len(images)),
                "has_3d": 1.0 if product.get("model3DUrl") else 0.0,
                "corpus": corpus,
            }
        )
    return pd.DataFrame(rows)

def interactions_to_df(
    reviews:List[Dict[str, Any]],
    orders:List[Dict[str, Any]],
) ->pd.DataFrame:
    """
    Chuyển đổi các tương tác người dùng (đánh giá và đơn hàng) thành một DataFrame của pandas, bao gồm các trường như user_id, product_id và score.
    Trường score được tính toán dựa trên các đánh giá và đơn hàng, phản ánh mức độ tương tác của người dùng với sản phẩm.
    Đánh giá được tính dựa trên điểm số đánh giá, trong khi đơn hàng được tính dựa trên số lượng sản phẩm đã mua, với trọng số cao hơn để phản ánh giá trị của đơn hàng so với đánh giá.
    """
    signals: Dict[tuple[str, str], float] = defaultdict(float)

    for review in reviews:
        user_id = to_id(review.get("user"))
        product_id = to_id(review.get("product"))
        rating = float(review.get("rating") or 0)
        if user_id and product_id:
            signals[(user_id, product_id)] += rating

    for order in orders:
        user_id = to_id(order.get("user"))
        for item in order.get("items") or []:
            product_id = to_id(item.get("product"))
            quantity = float(item.get("quantity") or 1)
            if user_id and product_id:
                signals[(user_id, product_id)] += quantity * 1.5
    
    rows = [
        {"user_id": user_id, "product_id": product_id, "score": score}
        for (user_id, product_id), score in signals.items()
    ]

    return pd.DataFrame(rows)
