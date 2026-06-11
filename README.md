# **Srt Video Matcher - Qwen3-VL-Embedding**

An AI-powered video and subtitle matching application. It matches subtitle segments (SRT) with corresponding video scenes using local vision-language embeddings, trimming and stitching them together into a final clip.

Ứng dụng đối chiếu và cắt ghép video theo phụ đề tự động bằng Trí tuệ Nhân tạo. Chương trình sử dụng mô hình nhúng ngôn ngữ - hình ảnh cục bộ để tìm kiếm phân cảnh phù hợp nhất với nội dung phụ đề SRT và tự động cắt ghép thành video hoàn chỉnh.

**Languages:** [English](#english) · [Tiếng Việt](#tiếng-việt)

---

<a name="english"></a>
## **English Version**

### **Features**
- **AI-Powered Matching**: Uses the offline `Qwen/Qwen3-VL-Embedding-2B` model to calculate cosine similarity between subtitle texts and video chunks.
- **Dual Processing Modes**:
  - **Mode 1 (AI + Fallback)**: If the AI doesn't find a strong match (below confidence threshold), it falls back to the original SRT timeline to ensure no content is missed.
  - **Mode 2 (AI Only)**: If the AI doesn't find a match, it skips the segment (yielding a shorter, highly-relevant video).
- **Multiple Interfaces**:
  - **Desktop GUI (`app.py`)**: A modern Windows-native GUI built with `PySide6` and `QFluentWidgets` (Fluent Design).
  - **Web UI (`webui.py`)**: A web interface built with `Gradio` featuring a sleek glassmorphism theme, perfect for remote execution.
  - **Google Colab (`SentrySearch_Colab.ipynb`)**: Zero-setup notebook to run the Web UI on free T4 cloud GPUs.
- **Hardware Acceleration**: Fully supports NVIDIA CUDA for fast local execution.

---

### **Getting Started**

#### **1. Prerequisites**
- Python 3.11 or 3.12 (highly recommended).
- FFmpeg installed on your system PATH.
- (Optional) NVIDIA GPU with CUDA drivers configured.

#### **2. Installation**
We recommend using [uv](https://docs.astral.sh/uv/) for fast package management:
```bash
# Clone the repository
git clone https://github.com/ltteamvn/Sentry-Search.git
cd Sentry-Search

# Install dependencies
uv pip install -e .[local] PySide6 PySide6-Fluent-Widgets pysrt moviepy gradio
```

#### **3. Run Desktop GUI**
```bash
uv run python app.py
```

#### **4. Run Web UI**
```bash
uv run python webui.py
```
After running, open the local URL (typically `http://localhost:7860`) or the public share URL.

#### **5. Google Colab**
Upload `SentrySearch_Colab.ipynb` to Google Colab, select a GPU runtime, and run all cells to get a public Gradio Web UI link.

---

<a name="tiếng-việt"></a>
## **Phiên Bản Tiếng Việt**

### **Tính năng nổi bật**
- **So khớp bằng AI cục bộ**: Sử dụng mô hình `Qwen/Qwen3-VL-Embedding-2B` hoàn toàn ngoại tuyến để tính toán độ tương đồng cosine giữa nội dung phụ đề và các phân cảnh video.
- **Hai chế độ xử lý linh hoạt**:
  - **Chế độ 1 (AI + Cắt bù theo timeline)**: Nếu AI không tìm thấy phân cảnh khớp (dưới ngưỡng tin cậy), phần mềm tự động cắt theo mốc thời gian gốc của file SRT để đảm bảo không bị mất nội dung.
  - **Chế độ 2 (Chỉ lấy đoạn khớp AI)**: Bỏ qua các đoạn phụ đề không khớp với bất kỳ cảnh nào trong video (kết quả video ngắn và cô đọng hơn).
- **Đa dạng giao diện người dùng**:
  - **Desktop GUI (`app.py`)**: Giao diện máy tính Windows hiện đại, mượt mà xây dựng trên `PySide6` và `QFluentWidgets` (phong cách Fluent Design của Windows 11).
  - **Web UI (`webui.py`)**: Giao diện Web được xây dựng bằng `Gradio` với phong cách Glassmorphism sang trọng, thích hợp chạy từ xa.
  - **Google Colab (`SentrySearch_Colab.ipynb`)**: File notebook tiện lợi giúp bạn chạy ứng dụng trên đám mây của Google bằng GPU T4 miễn phí mà không cần cài đặt phần cứng.
- **Tăng tốc phần cứng**: Hỗ trợ đầy đủ NVIDIA CUDA giúp chạy mô hình AI cực nhanh trên GPU.

---

### **Hướng dẫn cài đặt & sử dụng**

#### **1. Yêu cầu hệ thống**
- Python 3.11 hoặc 3.12 (khuyên dùng).
- Đã cài đặt FFmpeg và thêm vào biến môi trường PATH.
- (Tùy chọn) GPU NVIDIA đã cài đặt driver CUDA.

#### **2. Hướng dẫn cài đặt**
Khuyên dùng trình quản lý gói siêu tốc [uv](https://docs.astral.sh/uv/):
```bash
# Tải mã nguồn về máy
git clone https://github.com/ltteamvn/Sentry-Search.git
cd Sentry-Search

# Cài đặt toàn bộ môi trường và các gói thư viện
uv pip install -e .[local] PySide6 PySide6-Fluent-Widgets pysrt moviepy gradio
```

#### **3. Chạy giao diện Desktop GUI**
```bash
uv run python app.py
```

#### **4. Chạy giao diện Web UI**
```bash
uv run python webui.py
```
Sau khi khởi động, truy cập đường dẫn cục bộ (thường là `http://localhost:7860`) hoặc đường dẫn công khai (Public URL).

#### **5. Sử dụng Google Colab**
Tải tệp tin `SentrySearch_Colab.ipynb` lên Google Colab, đổi cấu hình Runtime sang GPU và chạy lần lượt các bước để nhận liên kết Web UI chạy trên đám mây.
