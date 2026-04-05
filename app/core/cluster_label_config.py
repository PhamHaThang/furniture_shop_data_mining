from __future__ import annotations

# Rule labels cho cụm sản phẩm. Mỗi rule gồm danh sách điều kiện all.
# Condition format: (feature_name, operator, threshold_key)
PRODUCT_LABEL_RULES = [
    {
        "label": "Premium chất lượng cao",
        "all": [
            ("price", ">=", "price_p75"),
            ("average_rating", ">=", "rating_p75"),
        ],
    },
    {
        "label": "Bán chạy",
        "all": [
            ("sold_count", ">=", "sold_p75"),
            ("total_reviews", ">=", "reviews_p75"),
        ],
    },
    {
        "label": "Tồn kho cao bán chậm",
        "all": [
            ("stock", ">=", "stock_p75"),
            ("sold_count", "<=", "sold_p25"),
        ],
    },
    {
        "label": "Giá tốt bán ổn",
        "all": [
            ("price", "<=", "price_p25"),
            ("sold_count", ">=", "sold_p50"),
        ],
    },
]

PRODUCT_DEFAULT_LABEL = "Nhóm phổ thông"

PRODUCT_SUFFIX_RULES = [
    {"suffix": "giá cao", "all": [("price", ">=", "price_p75")]},
    {"suffix": "giá thấp", "all": [("price", "<=", "price_p25")]},
    {"suffix": "bán nhanh", "all": [("sold_count", ">=", "sold_p75")]},
    {"suffix": "tồn kho cao", "all": [("stock", ">=", "stock_p75")]},
    {
        "suffix": "đánh giá cao",
        "all": [("average_rating", ">=", "rating_p75")],
    },
]
PRODUCT_DEFAULT_SUFFIX = "nhóm phụ"

# Rule labels cho cụm người dùng.
USER_LABEL_RULES = [
    {
        "label": "Khách VIP trung thành",
        "all": [
            ("total_spent", ">=", "spent_p75"),
            ("total_orders", ">=", "orders_p75"),
        ],
    },
    {
        "label": "Mua thường xuyên",
        "all": [
            ("total_orders", ">=", "orders_p75"),
            ("avg_order_value", "<=", "avg_order_p50"),
        ],
    },
    {
        "label": "Giá trị đơn cao",
        "all": [
            ("avg_order_value", ">=", "avg_order_p75"),
            ("total_orders", "<=", "orders_p50"),
        ],
    },
    {
        "label": "Tương tác cao",
        "all": [("total_reviews", ">=", "reviews_p75")],
    },
]

USER_DEFAULT_LABEL = "Khách hàng phổ thông"

USER_SUFFIX_RULES = [
    {"suffix": "chi tiêu cao", "all": [("total_spent", ">=", "spent_p75")]},
    {"suffix": "mua nhiều", "all": [("total_orders", ">=", "orders_p75")]},
    {
        "suffix": "đơn cao",
        "all": [("avg_order_value", ">=", "avg_order_p75")],
    },
    {
        "suffix": "review nhiều",
        "all": [("total_reviews", ">=", "reviews_p75")],
    },
]
USER_DEFAULT_SUFFIX = "nhóm phụ"
