#  Báo Cáo Giai Đoạn 1: Image Captioning với Tập Dữ Liệu Flickr8K

Đồ án tập trung vào việc nghiên cứu, xây dựng và đánh giá các hệ thống **Image Captioning** (chú thích ảnh tự động). Mục tiêu cốt lõi của Giai đoạn 1 là thực nghiệm và so sánh hiệu suất trích xuất đặc trưng ảnh giữa hai trường phái kiến trúc: **Convolutional Neural Networks (CNN)** và **Vision Transformers (ViT)**.

**Sinh viên thực hiện:** Nguyễn Vương Hồng Vỹ  
**Mã số sinh viên:** 2001231070  
**Môi trường huấn luyện:** Google Colab / Kaggle (GPU Tesla T4)  
**Tập dữ liệu:** Flickr8K (6,472 Train | 809 Val | 810 Test)

---

## Các Kiến Trúc Đã Thực Nghiệm

Dự án áp dụng mô hình theo cơ chế **Encoder-Decoder**. Trong đó, Decoder sử dụng **Transformer Decoder** (3 layers, 8 heads, d_model=256) cố định, và Visual Encoder được thay đổi để so sánh:

1. **ResNet-50:** Kiến trúc CNN cổ điển với Residual Block.
2. **EfficientNet-B3:** Kiến trúc CNN tối ưu hóa sự cân bằng giữa độ sâu, độ rộng và độ phân giải.
3. **ViT-B/16:** Vision Transformer tiêu chuẩn được pre-train trên ImageNet.
4. **CLIP-ViT-B/16:** Vision Transformer được huấn luyện đối chiếu (Contrastive Learning) trên các cặp Ảnh-Văn bản, mang lại khả năng thấu hiểu ngữ nghĩa sâu sắc.

---

##  Kết Quả Đánh Giá Điểm Số

Kết quả được đánh giá tự động trên tập Test (810 ảnh) bằng các độ đo phổ biến trong xử lý ngôn ngữ tự nhiên:

| Mô Hình Visual Encoder | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | 0.637 | 0.177 | 0.417 | 0.440 | 0.472 |
| **EfficientNet-B3** | 0.605 | 0.179 | 0.409 | 0.429 | 0.444 |
| **ViT-B/16** | 0.617 | 0.197 | 0.411 | 0.439 | 0.485 |
| **CLIP-ViT-B/16** | **0.692** | **0.223** | **0.472** | **0.492** | **0.614** |

**Nhận xét:** 
Mô hình sử dụng **CLIP-ViT-B/16** cho kết quả vượt trội hoàn toàn so với phần còn lại, chứng minh sức mạnh của mô hình nền tảng đa phương thức (Multimodal Foundation Model) trong bài toán kết hợp Thị giác và Ngôn ngữ.

---

## Cấu Trúc Repository

Repository này chứa 4 tệp Jupyter Notebook tương ứng với tiến trình huấn luyện và cấu hình của 4 mô hình:
- `BaoCao_CLIP-ViT-B-16.ipynb`: Cấu hình, log huấn luyện và inference của CLIP.
- `BaoCao_EfficientNet.ipynb`: Cấu hình, log huấn luyện và inference của EfficientNet.
- `BaoCao_ViT.ipynb`: Cấu hình, log huấn luyện và inference của ViT.
- `BaoCao_...ipynb`: Cấu hình, log huấn luyện và inference của ResNet-50.

Tất cả các mô hình đều được huấn luyện tối đa **25 Epochs**, sử dụng thuật toán tối ưu **AdamW**, Learning Rate `3e-4`, Batch Size `32` và kết hợp cơ chế **Early Stopping** để chống overfitting. Phương pháp sinh câu sử dụng ở giai đoạn này là **Greedy Decoding**.
