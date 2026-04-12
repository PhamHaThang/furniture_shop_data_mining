from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict
import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.schemas import MLRequest
from app.services.data_prep_service import interactions_to_df, products_to_df
from app.utils.normalizers import to_id

VIETNAMESE_STOP_WORDS = [
    "và", "của", "cho", "với", "trong", "là", "được", "này",
    "có", "từ", "các", "một", "những", "để", "đã", "sẽ",
    "không", "rất", "đến", "theo", "về", "như", "khi", "tại",
    
]

# Simple memory cache for TF-IDF since products change infrequently
_tfidf_cache: Dict[str, Any] = {}

def _get_tfidf_matrices(p_df: pd.DataFrame) -> tuple[Any, Any]:
    # Hash the product IDs + corpus to invalid cache if products change
    corpus_str = "".join(p_df["corpus"].tolist())
    corpus_hash = hashlib.md5(corpus_str.encode('utf-8')).hexdigest()
    
    if corpus_hash in _tfidf_cache:
        return _tfidf_cache[corpus_hash]
    
    vectorizer = TfidfVectorizer(
        stop_words=VIETNAMESE_STOP_WORDS,
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(p_df["corpus"])
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Store in cache
    _tfidf_cache.clear() # clear old cache to save memory
    _tfidf_cache[corpus_hash] = (tfidf_matrix, sim_matrix)
    return tfidf_matrix, sim_matrix


def _build_popularity_fallback(p_df: pd.DataFrame, top_k: int, model: str) -> Dict[str, Any]:
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
    if interactions.empty:
        return {"recommendations": [], "model": "collaborative"}

    product_scores = interactions.groupby("product_id")["score"].sum().sort_values(ascending=False)
    recommendations = [
        {"product_id": pid, "score": float(score)}
        for pid, score in product_scores.head(top_k).items()
    ]
    return {"recommendations": recommendations, "model": "collaborative"}


def content_based_recommendation(request:MLRequest) ->Dict[str, Any]:
    p_df = products_to_df(request.products)
    if p_df.empty:
        return {"recommendations": [], "model": "content-based"}
    
    tfidf_matrix, sim_matrix = _get_tfidf_matrices(p_df)

    products_ids = p_df["product_id"].tolist()
    idx_by_product = {pid: idx for idx, pid in enumerate(products_ids)}

    if request.target_product_id:
        target_id = str(request.target_product_id)
        if target_id not in idx_by_product:
            raise HTTPException(status_code=404, detail="Target product not found")
        
        target_idx = idx_by_product[target_id]
        scores = sim_matrix[target_idx]
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
    
    interactions = interactions_to_df(request.reviews, request.orders)
    if interactions.empty or not request.target_user_id:
        return _build_popularity_fallback(p_df, request.top_k, "content-based")
    
    user_interactions = interactions[interactions["user_id"] == request.target_user_id]
    if user_interactions.empty:
        return _build_popularity_fallback(p_df, request.top_k, "content-based")
    
    user_indices = []
    user_weights = []
    user_items = set()
    
    for _, row in user_interactions.iterrows():
        pid = row["product_id"]
        if pid in idx_by_product:
            user_indices.append(idx_by_product[pid])
            user_weights.append(float(row["score"]))
            user_items.add(pid)
            
    if not user_indices:
        return _build_popularity_fallback(p_df, request.top_k, "content-based")
    
    weights = np.array(user_weights)
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones(len(weights)) / len(weights)
        
    # Weighted Profile
    profile = np.average(sim_matrix[user_indices], axis=0, weights=weights)

    ranked = np.argsort(profile)[::-1]

    recommendations = []
    for idx in ranked:
        pid = products_ids[idx]
        if pid in user_items:
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


def collaborative_recommendation(request:MLRequest) ->Dict[str, Any]:
    interactions = interactions_to_df(request.reviews, request.orders)
    if interactions.empty:
        return {"recommendations": [], "model": "collaborative"}
    
    matrix = interactions.pivot_table(index="user_id", columns="product_id", values="score", fill_value=0)
    target = request.target_user_id
    
    if target not in matrix.index:
        return _build_collaborative_fallback(interactions, request.top_k)

    # Mean Centering
    matrix_centered = matrix.sub(matrix.mean(axis=1), axis=0)
    for user in matrix.index:
        user_rated = matrix.loc[user] > 0
        if user_rated.sum() > 0:
            mean_val = matrix.loc[user, user_rated].mean()
            matrix_centered.loc[user, user_rated] -= mean_val

    user_sim = cosine_similarity(matrix_centered)
    sim_df = pd.DataFrame(user_sim, index=matrix.index, columns=matrix.index)

    sim_scores = sim_df.loc[target].drop(target)
    # Get top 10 neighbors that have similarity > 0.1
    top_neighbors = sim_scores[sim_scores > 0.1].nlargest(10)
    
    if top_neighbors.empty:
        return _build_collaborative_fallback(interactions, request.top_k)

    weighted = np.dot(top_neighbors.values, matrix.loc[top_neighbors.index].values)
    pred = weighted / (np.abs(top_neighbors.values).sum() + 1e-8)

    target_seen = set(matrix.columns[matrix.loc[target] > 0])
    ranked_idx = np.argsort(pred)[::-1]
    recommendations = []
    columns = matrix.columns.tolist()
    
    for idx in ranked_idx:
        pid = columns[idx]
        if pid in target_seen:
            continue
        if pred[idx] <= 0:
            continue
        recommendations.append(
            {
                "product_id": pid,
                "score": float(pred[idx]),
            }
        )
        if len(recommendations) >= request.top_k:
            break
            
    if not recommendations:
         return _build_collaborative_fallback(interactions, request.top_k)
         
    return {
        "recommendations": recommendations,
        "model": "collaborative"
    }


def _normalize_scores(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not recs:
        return []
    scores = [r.get("score", 0) for r in recs]
    min_s, max_s = min(scores), max(scores)
    rng = max_s - min_s
    for r in recs:
        if rng == 0:
            r["normalized_score"] = 1.0
        else:
            r["normalized_score"] = (float(r.get("score", 0)) - min_s) / rng
    return recs

def _compute_adaptive_weights(content_recs: List[Dict[str, Any]], collab_recs: List[Dict[str, Any]]) -> tuple[float, float]:
    base_content, base_collab = 0.55, 0.45
    if not collab_recs:
        return 1.0, 0.0
    if not content_recs:
        return 0.0, 1.0
    
    avg_content = np.mean([r["normalized_score"] for r in content_recs])
    avg_collab = np.mean([r["normalized_score"] for r in collab_recs])
    
    total = avg_content + avg_collab
    if total == 0:
        return base_content, base_collab
        
    quality_content = avg_content / total
    quality_collab = avg_collab / total
    
    w_content = 0.7 * base_content + 0.3 * quality_content
    w_collab = 0.7 * base_collab + 0.3 * quality_collab
    
    total_w = w_content + w_collab
    return w_content / total_w, w_collab / total_w
    
def hybrid_recommendation(request:MLRequest) ->Dict[str, Any]:
    content = content_based_recommendation(request).get("recommendations", [])
    collaborative = collaborative_recommendation(request).get("recommendations", [])

    content = _normalize_scores(content)
    collaborative = _normalize_scores(collaborative)
    
    weight_content, weight_collab = _compute_adaptive_weights(content, collaborative)
    
    score_map: Dict[str, float] = defaultdict(float)

    for item in content:
        score_map[item["product_id"]] += weight_content * item["normalized_score"]

    for item in collaborative:
        score_map[item["product_id"]] += weight_collab * item["normalized_score"]
    
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
            "content": round(weight_content, 3),
            "collaborative": round(weight_collab, 3)
        }
    }