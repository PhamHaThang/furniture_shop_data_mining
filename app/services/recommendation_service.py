

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.schemas import MLRequest
from app.services.data_prep_service import interactions_to_df, products_to_df
from app.utils.normalizers import to_id


def _build_popularity_fallback(p_df: pd.DataFrame, top_k: int, model: str) -> Dict[str, Any]:
    """Fallback recommendation bằng sold_count và average_rating."""
    popularity = p_df.sort_values(by=["sold_count", "average_rating"], ascending=False).head(top_k)
    recommendations = [
        {
            "product_id": row["product_id"],
            "name": row["name"],
            "score": float(row["sold_count"] + row["average_rating"]),
        }
        for _, row in popularity.iterrows()
    ]
    return {"recommendations": recommendations, "model": model}


def _build_collaborative_fallback(interactions: pd.DataFrame, top_k: int) -> Dict[str, Any]:
    """
    Fallback recommendation dựa trên điểm số tương tác đã tổng hợp.
        Sản phẩm nào được mua nhiều hoặc đánh giá cao sẽ có điểm số cao hơn và được đề xuất nhiều hơn.
    """
    if interactions.empty:
        return {"recommendations": [], "model": "collaborative"}

    product_scores = interactions.groupby("product_id")["score"].sum().sort_values(ascending=False)
    recommendations = [
        {"product_id": pid, "score": float(score)}
        for pid, score in product_scores.head(top_k).items()
    ]
    return {"recommendations": recommendations, "model": "collaborative"}

"""
Content-based filtering
    + Dung TfidfVectorizer + cosine_similarity
    + Neu co target_product_id:
        - lay item gan nhat voi product dang xem
    + Neu co target_user_id:
        - lay lich su item cua user
        - tao profile similarity trung binh
    + Fallback khi data mong:
        - xep theo sold_count + average_rating
"""
def content_based_recommendation(request:MLRequest) ->Dict[str, Any]:
    # Chuyển đổi danh sách sản phẩm thành một DataFrame của pandas, bao gồm các trường như product_id, name, price, average_rating, total_reviews, sold_count, stock, image_count, has_3d và corpus.
    p_df = products_to_df(request.products)
    if p_df.empty:
        return {"recommendations": [], "model": "content-based"}
    
    # Tạo ma trận TF-IDF từ trường corpus của sản phẩm
    # stop_words="english" - loại bỏ các từ dừng phổ biến trong tiếng Anh
    # ngram_range=(1, 2) -  bao gồm cả unigram và bigram
    # max_features=5000 - giới hạn số lượng đặc trưng được tạo ra.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),max_features=5000)

    # Mỗi 1 product -> 1 vector TF-IDF dựa trên trường corpus của nó. 
    tfidf_matrix = vectorizer.fit_transform(p_df["corpus"])
    sim_matrix = cosine_similarity(tfidf_matrix)

    products_ids = p_df["product_id"].tolist()
    idx_by_product = {pid: idx for idx, pid in enumerate(products_ids)}


    # Recommendation theo Product đang xem (target_product_id)
    if request.target_product_id:
        target_id = str(request.target_product_id)
        if target_id not in idx_by_product:
            raise HTTPException(status_code=404, detail="Target product not found")
        
        target_idx = idx_by_product[target_id]

        # Tìm kiếm các sản phẩm tương tự dựa trên cosine similarity giữa vector TF-IDF của sản phẩm mục tiêu và tất cả các sản phẩm khác.
        scores = sim_matrix[target_idx]

        # Sắp xếp các sản phẩm theo điểm số similarity từ cao đến thấp, loại bỏ sản phẩm mục tiêu và chọn ra top K sản phẩm có điểm số cao nhất để đề xuất.
        rank = np.argsort(scores)[::-1]

        recommendations = []
        for idx in rank:
            if products_ids[idx] == target_id:
                continue
            recommendations.append(
               {
                    "product_id": products_ids[idx],
                    "score": float(scores[idx]),
                    "name": p_df.iloc[idx]["name"],
               }
            )
            if len(recommendations) >= request.top_k:
                break
        return {"recommendations": recommendations, "model": "content-based"}
    
    # Recommendation theo User (target_user_id)
    # Dựa theo Review + Order -> interaction -> user profile -> similarity -> recommendation
    interactions = interactions_to_df(request.reviews, request.orders)
    if(interactions.empty or not request.target_user_id):
        # Fallback: xep theo sold_count + average_rating nếu không có dữ liệu tương tác hoặc không có target_user_id
        # Đề xuất dựa trên số lựa đã bán và đánh giá trung bình của sản phẩm
        return _build_popularity_fallback(p_df, request.top_k, "content-based")
    
    # Lấy sản phẩm user đã tương tác (đánh giá hoặc mua hàng)
    user_items = interactions[interactions["user_id"] == request.target_user_id]["product_id"].tolist()
    if not user_items:
        # Fallback: xep theo sold_count + average_rating
        return _build_popularity_fallback(p_df, request.top_k, "content-based")
    
    # Lấy chỉ số của các sản phẩm mà user đã tương tác trong ma trận similarity
    user_indices = [idx_by_product[p] for p in user_items if p in idx_by_product]
    if not user_indices:
        # Fallback: xep theo sold_count + average_rating
        return _build_popularity_fallback(p_df, request.top_k, "content-based")
    
    # Tính điểm similarity trung bình giữa các sản phẩm mà user đã tương tác và tất cả các sản phẩm khác 
    # Tạo ra một Profile đại diện cho sở thích của User 
    profile = sim_matrix[user_indices].mean(axis=0)

    ranked = np.argsort(profile)[::-1]
    # Loại bỏ các sản phẩm mà user đã tương tác
    seen = set(user_items)

    recommendations = []
    for idx in ranked:
        pid = products_ids[idx]
        if pid in seen:
            continue
        recommendations.append(
            {
                "product_id": pid,
                "score": float(profile[idx]),
                "name": p_df.iloc[idx]["name"],
            }
        )
        if len(recommendations) >= request.top_k:
            break
    return {"recommendations": recommendations, "model": "content-based"}

"""
Collaborative filtering
    + Dung cosine similarity tren user-item matrix
    + Neu co target_user_id:
        - lay lich su item cua user
        - tinh similarity voi user khac
        - weighted average rating cua san pham chua xem
    + Fallback khi data mong:
        - xep theo sold_count + average_rating
"""
def collaborative_recommendation(request:MLRequest) ->Dict[str, Any]:
    interactions = interactions_to_df(request.reviews, request.orders)
    if interactions.empty:
        return {"recommendations": [], "model": "collaborative"}
    
    # Tạo ma trận user-item từ DataFrame tương tác, với user_id là chỉ số hàng, product_id là chỉ số cột và score là giá trị của ma trận. Các giá trị thiếu được điền bằng 0.
    matrix = interactions.pivot_table(index="user_id", columns="product_id", values="score", fill_value=0)
    if request.target_user_id not in matrix.index:
        # Fallback: xep theo sold_count + average_rating
        return _build_collaborative_fallback(interactions, request.top_k)

    # Tính user similarity bằng cosine similarity giữa các hàng của ma trận user-item. 
    # Kết quả là một ma trận vuông với kích thước bằng số lượng người dùng, trong đó mỗi phần tử (i, j) đại diện cho độ tương đồng giữa người dùng i và người dùng j.
    user_sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

    target = request.target_user_id

    #  Lấy trọng số similarity giữa người dùng mục tiêu và tất cả người dùng khác từ ma trận similarity.
    weights = sim_df.loc[target].drop(target)
    if weights.abs().sum() == 0:
        # Fallback: xep theo sold_count + average_rating
        return _build_collaborative_fallback(interactions, request.top_k)

    # Dự đoán điểm số cho các sản phẩm mà người dùng mục tiêu chưa tương tác bằng cách tính trung bình có trọng số của điểm số của tất cả người dùng khác.
    weighted = np.dot(weights.values, matrix.loc[weights.index].values)
    pred = weighted / (np.abs(weights.values).sum() + 1e-8)

    target_seen = set(matrix.columns[matrix.loc[target] > 0])
    ranked_idx = np.argsort(pred)[::-1]
    recommendations = []
    columns = matrix.columns.tolist()
    for idx in ranked_idx:
        pid = columns[idx]
        if pid in target_seen:
            continue
        recommendations.append(
            {
                "product_id": pid,
                "score": float(pred[idx]),
            }
        )
        if len(recommendations) >= request.top_k:
            break
    return {
        "recommendations": recommendations,
        "model": "collaborative"
    }


"""
Hybrid recommendation
    + Kết hợp điểm số từ cả 2 model content-based và collaborative filtering bằng cách tính trung bình có trọng số.
    + Score_hybrid = weight_content * Score_content + weight_collab * Score_collab
        - weight_content = 0.55
        - weight_collab = 0.45
"""
def hybrid_recommendation(request:MLRequest) ->Dict[str, Any]:
    # Lấy kết quả từ 2 model: Content-based và Collaborative filtering
    content = content_based_recommendation(request).get("recommendations", [])
    collaborative = collaborative_recommendation(request).get("recommendations", [])

    score_map: Dict[str, float] = defaultdict(float)

    weight_content = 0.55
    weight_collab = 0.45
    
    for item in content:
        score_map[item["product_id"]] += weight_content * float(item.get("score", 0))

    for item in collaborative:
        score_map[item["product_id"]] += weight_collab * float(item.get("score", 0))
    
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    top = ranked[:request.top_k]

    names = {str(to_id(p.get("_id"))): p.get("name", "") for p in request.products}
    recommendations = [
        {
            "product_id": pid,
            "score": score,
            "name": names.get(pid, "")
        }
        for pid, score in top
    ]
    return {
        "recommendations": recommendations,
        "model": "hybrid",
        "weights": {
            "content": weight_content,
            "collaborative": weight_collab
        }
    }