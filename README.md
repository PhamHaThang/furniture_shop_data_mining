# HOMI Shop — Data Mining Service

Dịch vụ FastAPI cung cấp gợi ý sản phẩm, phân cụm sản phẩm/người dùng và phân tích đánh giá cho HOMI Shop. Backend Node.js đọc dữ liệu từ MongoDB, gửi JSON sang dịch vụ này và đưa kết quả về frontend.

## Chức năng

| Chức năng | Phương pháp |
| --- | --- |
| Gợi ý theo nội dung | TF-IDF từ đặc trưng văn bản sản phẩm và cosine similarity |
| Gợi ý cộng tác | Tính tương đồng giữa người dùng dựa trên đánh giá và đơn hàng |
| Gợi ý kết hợp | Chuẩn hóa điểm và kết hợp hai phương pháp bằng trọng số thích ứng |
| Phân cụm | K-Means trên đặc trưng sản phẩm hoặc hành vi người dùng |
| Phân tích cảm xúc | Từ khóa tiếng Việt/Anh, xử lý phủ định và kết hợp số sao |
| Phân tích quản trị | Tổng hợp đơn hàng, đánh giá, phân cụm và sản phẩm theo cảm xúc |

Dịch vụ tính kết quả từ dữ liệu của từng request. Không cần file mô hình huấn luyện sẵn hoặc kết nối MongoDB riêng; cache TF-IDF được lưu trong bộ nhớ tiến trình.

## Công nghệ và yêu cầu

- Python 3.11 trở lên và pip.
- FastAPI `0.116.1`, Uvicorn `0.35.0`.
- NumPy `2.3.3`, pandas `2.3.2`, scikit-learn `1.7.2`.
- Pydantic `>=2.0.0`.

Các dependency được khai báo trong [requirements.txt](requirements.txt).

## Cài đặt và khởi động

Có thể clone riêng dịch vụ:

```powershell
git clone https://github.com/PhamHaThang/furniture_shop_data_mining.git
cd furniture_shop_data_mining
```

Nếu đã clone dự án tổng, mở terminal trong thư mục `data_mining`. Các lệnh sau chạy từ thư mục chứa `main.py` và `requirements.txt`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

### macOS/Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

Các lệnh dùng trực tiếp Python trong virtual environment, không cần kích hoạt môi trường trước. `--reload` phục vụ phát triển; bỏ tùy chọn này khi chạy tiến trình ổn định.

| Địa chỉ | Mục đích |
| --- | --- |
| http://127.0.0.1:8001/health | Kiểm tra dịch vụ |
| http://127.0.0.1:8001/docs | Swagger UI để xem schema và thử API |
| http://127.0.0.1:8001/redoc | Tài liệu ReDoc |
| http://127.0.0.1:8001/openapi.json | Đặc tả OpenAPI |

## Kết nối với backend

Trong `.env` của backend, đặt:

```dotenv
PYTHON_ML_SERVICE_URL=http://127.0.0.1:8001
```

Tên được `backend/services/mlService.js` sử dụng là **`PYTHON_ML_SERVICE_URL`**. File mẫu backend hiện ghi `ML_SERVICE_URL`, cần đổi tên khi cấu hình. Nếu thay cổng Uvicorn, cập nhật URL này và khởi động lại backend.

Backend giới hạn số bản ghi qua `ML_MAX_USERS`, `ML_MAX_PRODUCTS`, `ML_MAX_REVIEWS`, `ML_MAX_ORDERS`; các giới hạn mặc định lần lượt là `1500`, `3000`, `4000`, `4000`. Thời gian chờ HTTP của backend hiện là 10 giây.

## API

Các đường dẫn dưới đây gọi trực tiếp dịch vụ Python, **không có tiền tố `/api`**:

| Phương thức | Đường dẫn | Dữ liệu chính |
| --- | --- | --- |
| GET | `/health` | Không cần body; trả `{"status":"ok"}` |
| POST | `/recommend/content-based` | `products`, có thể thêm `target_product_id` hoặc `target_user_id`, `reviews`, `orders` |
| POST | `/recommend/collaborative` | `reviews`, `orders`, `target_user_id` |
| POST | `/recommend/hybrid` | `products`, `reviews`, `orders`, `target_user_id` |
| POST | `/cluster/kmeans?cluster_type=products` | `products`, `clusters` |
| POST | `/cluster/kmeans?cluster_type=users` | `orders`, `reviews`, `clusters` |
| POST | `/sentiment/reviews` | `reviews` |
| POST | `/analytics/admin` | `products`, `reviews`, `orders`, `clusters` |

`cluster_type` chỉ nhận `products` hoặc `users`, mặc định là `products`. Giá trị khác trả HTTP 400.

### Body dùng chung

Tất cả endpoint POST dùng [MLRequest](app/core/schemas.py):

```json
{
  "users": [],
  "products": [],
  "reviews": [],
  "orders": [],
  "target_user_id": null,
  "target_product_id": null,
  "top_k": 8,
  "clusters": 4
}
```

Các trường đều có giá trị mặc định như ví dụ. Chỉ cần gửi những trường endpoint sử dụng. Dùng `top_k` và `clusters` là số nguyên dương; số kết quả thực tế phụ thuộc dữ liệu đầu vào. Các phần tử trong danh sách hiện là dictionary linh hoạt, vì vậy phía gửi cần bảo đảm trường dữ liệu đúng kiểu và ID nhất quán.

| Loại bản ghi | Các trường được thuật toán sử dụng |
| --- | --- |
| Sản phẩm | `_id`, `name`, `description`, `price`, `averageRating`, `totalReviews`, `soldCount`, `stock`, `images`, `model3DUrl`, `category.name`, `brand.name`, `tags`, `colors`, `materials` |
| Đánh giá | `_id`, `user`, `product`, `rating`, `comment`, `createdAt` |
| Đơn hàng | `_id`, `user`, `items[].product`, `items[].quantity`, `totalAmount`, `status`, `payment.status`, `createdAt` |

Có thể dùng ID dạng chuỗi để liên kết `user` và `product` giữa các bản ghi. `createdAt` nên dùng chuỗi ISO 8601. Endpoint phân cụm người dùng tổng hợp hành vi từ `orders` và `reviews`; chỉ gửi danh sách `users` sẽ không tạo được đặc trưng hành vi.

### Ví dụ gợi ý sản phẩm tương tự

Thử trong Swagger hoặc gửi body sau tới `POST /recommend/content-based`:

```json
{
  "products": [
    {"_id": "p1", "name": "Sofa vai", "description": "Sofa phong khach", "price": 5000000},
    {"_id": "p2", "name": "Sofa vai nho", "description": "Sofa phong khach nho", "price": 4000000},
    {"_id": "p3", "name": "Ban go", "description": "Ban an bang go", "price": 3000000}
  ],
  "target_product_id": "p1",
  "top_k": 2
}
```

Kết quả có `model: "content-based"` và mảng `recommendations`, mỗi phần tử gồm `product_id`, `name`, `score`. Sản phẩm đích được loại khỏi danh sách; `score` là điểm xếp hạng, không phải xác suất mua hàng.

### Ví dụ phân tích đánh giá bằng PowerShell

```powershell
$payload = @{
    reviews = @(
        @{
            _id = "r1"
            user = "u1"
            product = "p1"
            rating = 5
            comment = "good quality"
        }
    )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8001/sentiment/reviews -Method Post -ContentType "application/json" -Body $payload
```

Phản hồi gồm `summary` và `reviews`. Với ví dụ trên, nhãn đánh giá là `Good`; `summary.total` và `summary.good` đều bằng `1`.

## Cách thuật toán xử lý dữ liệu

- **Gợi ý theo nội dung:** kết hợp tên, mô tả, danh mục, thương hiệu, tags, màu sắc, chất liệu và thông tin ảnh/3D thành văn bản đặc trưng. TF-IDF dùng n-gram từ 1 đến 3, tối đa 5.000 đặc trưng; cosine similarity dùng để xếp hạng.
- **Tương tác:** cộng tín hiệu đánh giá và số lượng mua, với tín hiệu mua có hệ số `1.5`. Trọng số giảm theo thời gian, chu kỳ bán rã mặc định khoảng 90 ngày.
- **Gợi ý cộng tác:** dùng ma trận người dùng–sản phẩm và cosine similarity; chọn tối đa 10 người dùng lân cận có độ tương đồng lớn hơn `0.1`.
- **Dữ liệu mới hoặc thưa:** content-based có thể chuyển sang xếp hạng theo lượt bán và điểm đánh giá; collaborative có thể dùng tổng điểm tương tác. Không có tương tác thì collaborative trả danh sách rỗng.
- **Phân cụm:** xử lý ngoại lệ bằng IQR, chuẩn hóa bằng `StandardScaler`, thử số cụm và chọn theo silhouette score. `clusters` định hướng phạm vi tìm kiếm, số cụm kết quả có thể khác giá trị truyền vào. Quy tắc đặt tên cụm nằm trong [cluster_label_config.py](app/core/cluster_label_config.py).
- **Cảm xúc:** phân loại `Good`, `Neutral`, `Bad` bằng luật từ khóa và số sao. Đây là cơ chế dựa trên luật, chưa sử dụng mô hình ngôn ngữ để phân tích ngữ cảnh.

Phân cụm sản phẩm dùng giá, điểm đánh giá, số đánh giá, số bán, tồn kho, số ảnh và trạng thái có mô hình 3D. Phân cụm người dùng dùng số đơn, tổng số lượng mua, tổng chi tiêu, giá trị đơn trung bình, số đánh giá và điểm đánh giá trung bình.

## Cấu trúc mã nguồn

```text
data_mining/
├── main.py                         # Export app cho Uvicorn
├── requirements.txt
└── app/
    ├── main.py                     # Khởi tạo FastAPI và CORS
    ├── api/routes/
    │   ├── health.py
    │   ├── recommendations.py      # Gợi ý và phân cụm
    │   └── analytics.py            # Cảm xúc và dashboard
    ├── core/
    │   ├── schemas.py              # MLRequest
    │   └── cluster_label_config.py # Quy tắc nhãn cụm
    ├── services/
    │   ├── data_prep_service.py
    │   ├── recommendation_service.py
    │   ├── clustering_service.py
    │   ├── sentiment_service.py
    │   └── analytics_service.py
    └── utils/normalizers.py        # Chuẩn hóa ID và văn bản
```

## Kiểm tra và xử lý lỗi

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

Health check chỉ xác nhận ứng dụng phản hồi; dùng các ví dụ POST ở trên để kiểm tra xử lý dữ liệu. Repository hiện chưa có bộ kiểm thử tự động.

| Hiện tượng | Cách kiểm tra |
| --- | --- |
| Không import được module `app` | Chạy Uvicorn tại thư mục có `main.py` và `requirements.txt` |
| Thiếu dependency | Dùng đúng Python trong `.venv` để cài `requirements.txt` và chạy Uvicorn |
| Backend không gọi được Python | Kiểm tra cổng, địa chỉ và `PYTHON_ML_SERVICE_URL` của backend |
| HTTP 422 | Kiểm tra body theo schema trong `/docs` |
| HTTP 404 khi gợi ý theo sản phẩm | Kiểm tra `target_product_id` tồn tại trong mảng `products` gửi lên |
| Gợi ý hoặc phân cụm rỗng | Kiểm tra dữ liệu sản phẩm và tương tác thực sự được gửi trong request |
| Phân cụm báo lỗi với dữ liệu ít đa dạng | Kiểm tra số bản ghi và số nhóm đặc trưng khác nhau; silhouette cần nhiều hơn một nhãn cụm |
| Request chậm hoặc backend hết thời gian chờ | Giảm lượng dữ liệu qua `ML_MAX_*` và kiểm tra tài nguyên tiến trình Python |

Dịch vụ hiện chưa có xác thực riêng và CORS cho phép mọi origin. Lệnh khởi động trong tài liệu chỉ lắng nghe trên `127.0.0.1`; khi triển khai cần giới hạn truy cập dịch vụ cho backend. Ma trận tương đồng sản phẩm/người dùng được tính trong bộ nhớ, nên lượng dữ liệu lớn sẽ làm tăng nhu cầu RAM và thời gian xử lý.
