# 🔐 Advanced Image Encryption System: Bit-Plane & Chaos Theory
**Source Code đính kèm Báo cáo Đồ án Tốt nghiệp – 2026**

Hệ thống nghiên cứu, phân tích và kiểm thử các thuật toán **mã hóa ảnh kỹ thuật số**, đặc biệt hướng đến **ảnh y tế (Medical Images)**.  
Dự án sử dụng các phương pháp dựa trên **Bit-Plane Slicing** và **Chaos Theory** để thực hiện hai quá trình cốt lõi của mã hóa ảnh:

- **Confusion (Xáo trộn)**
- **Diffusion (Khuếch tán)**

Hệ thống cung cấp **Dashboard trực quan** để thực thi mã hóa, giải mã và đánh giá độ an toàn của thuật toán.

---

# 💡 Ý tưởng & Đóng góp chính của đồ án

Trong quá trình nghiên cứu, đồ án phân tích **hai phương pháp mã hóa đã được công bố trước đó** và từ đó **đề xuất một phương pháp Hybrid mới** bằng cách kết hợp các thành phần mạnh nhất của chúng.

---

## 1️⃣ Normal Mode (Bit-Plane Rotation)

Đây là phương pháp mã hóa cơ sở được nghiên cứu trong đồ án.

Quy trình gồm hai bước:

**Confusion**
- Phân rã ảnh thành **8 mặt phẳng bit (Bit-Plane Slicing)**
- Thực hiện **xoay ma trận con (Rotation)** để xáo trộn vị trí bit

**Diffusion**
- Sử dụng **Hyper-Chaos system** để khuếch tán dữ liệu
- Khóa khởi tạo được sinh từ **MD5 hash**

Tóm tắt quy trình:

```
Rotation → Hyper-Chaos Diffusion
```

---

## 2️⃣ Medical Hyper-Chaos Mode

Phương pháp này được thiết kế chuyên biệt cho **ảnh y tế**, thường dùng với:

- X-Ray
- MRI
- CT Scan

Quy trình:

**Confusion**
- Phân rã ảnh thành các **bit-plane**
- Sử dụng **Arnold Transform** để xáo trộn bit trên từng mặt phẳng

**Diffusion**

- Sử dụng **Logistic Map**
- Kết hợp **XOR với khóa sinh từ SHA-512**

Tóm tắt quy trình:

```
Arnold Transform → Logistic Diffusion
```

---

## 3️⃣ Hybrid Encryption System (Đề xuất của đồ án)

Trong quá trình nghiên cứu hai phương pháp trên, nhận thấy:

- **Arnold Transform** có khả năng **xáo trộn bit mạnh hơn phép xoay**
- **Hyper-Chaos diffusion** có khả năng **khuếch tán tốt hơn Logistic map**

Vì vậy đồ án đề xuất **một thuật toán Hybrid**, kết hợp các phần mạnh nhất của hai phương pháp:

**Confusion**

- Sử dụng **Arnold Transform trên Bit-Planes**

**Diffusion**

- Sử dụng **Hyper-Lorenz Chaos System**

Tóm tắt quy trình Hybrid:

```
Arnold Transform → Hyper-Chaos Diffusion
```

Ý tưởng cốt lõi của phương pháp Hybrid dựa trên nguyên lý cơ bản của mã hóa ảnh:

> **Hoán vị (Permutation) + Khuếch tán (Diffusion)**

Bằng cách kết hợp hai thành phần mạnh nhất của hai phương pháp trước, hệ Hybrid giúp:

- Tăng **Plaintext Sensitivity**
- Tăng **khả năng chống tấn công vi sai**
- Cải thiện **Entropy và Correlation**

---

# 📂 Cấu trúc Repository

```
Medical-Image-Encryption-BitPlane
│
├── Bit_Plane_Rotation/      # Thuật toán cơ sở
├── Medical/                 # Thuật toán cho ảnh y tế
├── Hybrid/                  # Thuật toán Hybrid (đề xuất)
├── common_core/             # Các hàm dùng chung + AES đối chứng
├── Image_Test/              # Tập dữ liệu ảnh test
│
├── main_dashboard.py        # Dashboard mã hóa & phân tích
├── main_dashboard_decrypt.py# Dashboard giải mã
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Cài đặt và chạy chương trình

## 1. Yêu cầu hệ thống

- Python **3.8+**
- Khuyến nghị dùng **Virtual Environment**

---

## 2. Cài đặt thư viện

```
pip install -r requirements.txt
```

Các thư viện chính:

- numpy  
- opencv-python  
- pillow  
- matplotlib  
- pycryptodome  

---

## 3. Chạy hệ thống

### Dashboard mã hóa và phân tích

```
python main_dashboard.py
```

Chức năng:

- Mã hóa ảnh
- Phân tích chỉ số bảo mật
- So sánh với AES-128

---

### Dashboard giải mã

```
python main_dashboard_decrypt.py
```

Chức năng:

- Giải mã ảnh
- Kiểm tra độ chính xác khôi phục dữ liệu

---

# 📊 Các chỉ số đánh giá hệ thống

Dashboard tự động tính toán các chỉ số đánh giá chuẩn trong lĩnh vực **Image Encryption**.

### Randomness Analysis

- Entropy
- Histogram Analysis
- Correlation (Horizontal / Vertical / Diagonal)

### Differential Attack Resistance

- NPCR
- UACI

### Image Quality Metrics

- PSNR
- SSIM
- BER

---

# 👨‍💻 Tác giả

**Vũ Minh Quang** &
**Lưu Đình Tuấn**

Đồ án Tốt nghiệp – 2026  
Chủ đề: **Image Encryption using Bit-Plane and Chaos Theory**
