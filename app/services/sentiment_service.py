from __future__ import annotations
from typing import Any, Dict, List
from app.utils.normalizers import to_id

POSITIVE_WORDS = {
    # Vietnamese (unaccented + accented)
    "tot", "tốt", "dep", "đẹp", "hai long", "hài lòng", "chat luong", "chất lượng",
    "xuat sac", "xuất sắc", "yeu thich", "yêu thích", "tuyet voi", "tuyệt vời",
    "dang tien", "đáng tiền", "nhanh", "gia tot", "giá tốt", "uy tin", "uy tín",
    "ben", "bền", "sang trong", "sang trọng", "on dinh", "ổn định", "hoan hao",
    "hoàn hảo", "tien loi", "tiện lợi", "phu hop", "phù hợp", "rat thich",
    "rất thích", "mua tiep", "mua tiếp", "gioi thieu", "giới thiệu", "on", "ổn",
    "ok", "ưng ý", "ung y", "tuyet", "tuyệt", "xinh", "dễ thương",
    # English
    "recommend", "good", "great", "excellent", "worth", "amazing",
    "beautiful", "comfortable", "sturdy", "quality", "nice", "perfect", "love"
}

NEGATIVE_WORDS = {
    # Vietnamese
    "te", "tệ", "kem", "kém", "that vong", "thất vọng", "khong hai long", "không hài lòng",
    "xau", "xấu", "hu", "hư", "vo", "vỡ", "gay", "gãy", "loi", "lỗi", "cham", "chậm",
    "sai", "gia", "giả", "khong dung", "không đúng", "khong tot", "không tốt",
    "do", "dở", "chan", "chán", "hoi", "hôi", "khong phu hop", "không phù hợp",
    "tra hang", "trả hàng", "hoan tien", "hoàn tiền", "mac", "mắc", "dat", "đắt",
    # English
    "bad", "poor", "terrible", "worst", "not good", "broken", "damaged",
    "cheap", "waste", "disappointed", "awful"
}

NEGATION_TOKENS = {"khong", "không", "chua", "chưa", "chang", "chẳng", "ko", "k", "not"}

def _compute_sentiment_score(text: str) -> int:
    words = text.lower().split()
    score = 0
    negate = False
    
    for i, word in enumerate(words):
        if word in NEGATION_TOKENS:
            negate = True
            continue
        
        # Check bigrams if possible
        bigram = f"{words[i-1]} {word}" if i > 0 else ""
        
        if bigram in POSITIVE_WORDS or word in POSITIVE_WORDS:
            score += (-1 if negate else 1)
            negate = False # reset
        elif bigram in NEGATIVE_WORDS or word in NEGATIVE_WORDS:
            score += (1 if negate else -1) # "không tệ" -> +1
            negate = False # reset
        else:
            pass
            
    return score

def sentiment_label(comment: str, rating: float) -> Dict[str, Any]:
    text = (comment or "").strip()
    score = _compute_sentiment_score(text)

    # Classification Logic
    if rating >= 4 and score >= -1:
        label = "Good"
    elif rating <= 2 and score <= 1:
        label = "Bad"
    elif score > 0:
        label = "Good"
    elif score < 0:
        label = "Bad"
    else:
        if rating > 3:
            label = "Good"
        elif rating < 3:
            label = "Bad"
        else:
            label = "Neutral"

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
        elif result["label"] == "Bad":
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

def run_sentiment_analysis(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Backward-compatible alias for sentiment_analysis."""
    return sentiment_analysis(reviews)