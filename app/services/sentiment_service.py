from __future__ import annotations

from typing import Any, Dict, List

from app.utils.normalizers import to_id

def sentiment_label(comment: str, rating: float) -> Dict[str, Any]:
    positive_words = {
        "tot",
        "dep",
        "hai long",
        "chat luong",
        "xuat sac",
        "yeu thich",
        "recommend",
        "good",
        "great",
        "excellent",
        "worth",
    }
    negative_words = {
        "te",
        "kem",
        "that vong",
        "khong hai long",
        "vo",
        "xau",
        "bad",
        "poor",
        "terrible",
        "worst",
        "not good",
    }

    text = (comment or "").lower()
    score = 0

    for token in positive_words:
        if token in text:
            score += 1
    for token in negative_words:
        if token in text:
            score -= 1

    if rating >= 4 and score >= -1:
        label = "Good"
    elif rating <= 2 and score <= 1:
        label = "Bad"
    elif score > 0:
        label = "Good"
    else:
        label = "Bad"

    confidence = min(1.0, 0.55 + abs(score) * 0.15 + abs(rating - 3.0) * 0.05)
    return {"label": label, "confidence": round(confidence, 3), "score": score}

def sentiment_analysis(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    labelled_comments = []
    good_count = 0
    bad_count = 0

    for review in reviews:
        result = sentiment_label(review.get("comment", ""), float(review.get("rating") or 0))
        row = {
            "review_id": to_id(review.get("_id")),
            "product_id": to_id(review.get("product")),
            "user_id": to_id(review.get("user")),
            "rating": float(review.get("rating") or 0),
            "comment": review.get("comment", ""),
            **result,
        }
        labelled_comments.append(row)
        if result["label"] == "Good":
            good_count += 1
        else:
            bad_count += 1
    total = len(labelled_comments)
    return {
        "summary": {
            "total": total,
            "good": good_count,
            "bad": bad_count,
            "good_ratio": round(good_count / total, 4) if total else 0,
            "bad_ratio": round(bad_count / total, 4) if total else 0,
        },
        "reviews": labelled_comments,
    }