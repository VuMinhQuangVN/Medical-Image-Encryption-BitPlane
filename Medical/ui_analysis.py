# Anh_Y_Te/ui_analysis.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import math
import os
import cv2
import time

# --- IMPORT CÁC THUẬT TOÁN CORE Y TẾ ---
from core.medical_encrypt import encrypt_medical
from core.medical_decrypt import decrypt_medical

# --- IMPORT CÁC HÀM PHÂN TÍCH TỪ UTILS (DÙNG CHUNG) ---
from core.analysis_utils import (
    calculate_entropy, 
    calculate_correlation, 
    calculate_npcr_uaci,
    calculate_psnr, 
    calculate_ssim, 
    calculate_ber
)

class MedicalAutoAnalysisUI:
    def __init__(self, parent):
        self.parent = parent
        # Nếu dùng độc lập thì parent là root, nếu tích hợp thì parent là Frame
        if isinstance(parent, tk.Tk):
            self.parent.title("Hệ Thống Phân Tích Tự Động Ảnh Y Tế - Đồ Án 2026")
            self.parent.geometry("1200x600")

        self.img_orig = None
        self.filename = ""
        self.setup_ui()

    def setup_ui(self):
        bg_color = "#f4f7f6"
        
        # Frame chính để có thể nhúng vào Tab
        self.main_container = tk.Frame(self.parent, bg=bg_color)
        self.main_container.pack(fill="both", expand=True)

        # --- PANEL ĐIỀU KHIỂN (Bên trái) ---
        ctrl_frame = tk.LabelFrame(self.main_container, text=" THIẾT LẬP THỰC NGHIỆM ", bg=bg_color, font=("Arial", 10, "bold"), padx=15, pady=15)
        ctrl_frame.pack(side="left", fill="y", padx=20, pady=20)

        tk.Label(ctrl_frame, text="Mật khẩu SHA-512 (Key 1):", bg=bg_color, font=("Arial", 9)).pack(anchor="w")
        self.ent_pwd = tk.Entry(ctrl_frame, width=25, font=("Arial", 10), bd=2)
        self.ent_pwd.insert(0, "Mật mã y tế 2026")
        self.ent_pwd.pack(pady=5)

        tk.Button(ctrl_frame, text="📁 1. CHỌN ẢNH Y TẾ", command=self.load_image, bg="#34495e", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=10)
        self.lbl_info = tk.Label(ctrl_frame, text="Chưa nạp ảnh", bg=bg_color, font=("Arial", 8, "italic"), fg="#7f8c8d")
        self.lbl_info.pack(pady=(0, 20))

        tk.Button(ctrl_frame, text="🚀 2. CHẠY PHÂN TÍCH TỰ ĐỘNG", command=self.run_auto_analysis, bg="#27ae60", fg="white", font=("Arial", 10, "bold"), height=2).pack(fill="x", pady=5)
        
        tk.Button(ctrl_frame, text="🗑 Làm mới bảng số liệu", command=self.clear_table, bg="#e74c3c", fg="white").pack(fill="x", pady=20)

        # --- BẢNG KẾT QUẢ PHÂN TÍCH (Bên phải) ---
        res_frame = tk.LabelFrame(self.main_container, text=" BẢNG SỐ LIỆU ĐỊNH LƯỢNG (KẾT QUẢ THỰC NGHIỆM) ", bg="white", font=("Arial", 10, "bold"), padx=10, pady=10)
        res_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Định nghĩa các cột theo tiêu chuẩn bài báo khoa học
        columns = ("name", "entropy", "corr_h", "corr_v", "corr_d", "npcr", "uaci", "e_psnr", "e_ssim", "d_psnr", "d_ssim", "d_ber")
        self.tree = ttk.Treeview(res_frame, columns=columns, show="headings")
        
        headers = ["Tên ảnh", "Entropy", "Corr(H)", "Corr(V)", "Corr(D)", "NPCR", "UACI", "PSNR(E)", "SSIM(E)", "PSNR(D)", "SSIM(D)", "BER(D)"]
        for col, head in zip(columns, headers):
            self.tree.heading(col, text=head)
            self.tree.column(col, width=80, anchor="center") 

        # Thêm Scrollbar cho bảng
        scrollbar = ttk.Scrollbar(res_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            # Sử dụng OpenCV để nạp và chuẩn hóa về 256x256 (Đồng bộ với Encrypt/Decrypt)
            img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_cv is not None:
                self.img_orig = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                self.filename = os.path.basename(path)
                self.lbl_info.config(text=f"Đã nạp: {self.filename}", fg="#27ae60")

    def run_auto_analysis(self):
        if self.img_orig is None:
            messagebox.showwarning("Thông báo", "Vui lòng nạp ảnh gốc trước khi phân tích!")
            return
        
        pwd = self.ent_pwd.get()
        if not pwd:
            messagebox.showwarning("Thông báo", "Vui lòng nhập mật khẩu SHA-512!")
            return

        try:
            # --- BƯỚC 1: MÃ HÓA ẢNH GỐC (Sử dụng mode is_analysis=True) ---
            # Kết quả cho ra Cipher C1 và giá trị k2
            cipher_c1, k2 = encrypt_medical(self.img_orig, pwd, is_analysis=True)

            # --- BƯỚC 2: TẠO ẢNH SAI 1 PIXEL & MÃ HÓA (Để tính NPCR/UACI) ---
            img_mod = self.img_orig.copy()
            # Thay đổi pixel tại tọa độ [0,0]
            img_mod[0, 0] = (int(img_mod[0, 0]) + 1) % 256
            # Mã hóa bản copy này để lấy Cipher C2
            cipher_c2, _ = encrypt_medical(img_mod.astype(np.uint8), pwd, is_analysis=True)

            # --- BƯỚC 3: GIẢI MÃ (Để tính khả năng khôi phục) ---
            # Truyền original_ref=self.img_orig để giải mã mode Analysis
            decrypted_img = decrypt_medical(cipher_c1, pwd, k2, original_ref=self.img_orig)

            # --- BƯỚC 4: TÍNH TOÁN CÁC CHỈ SỐ (GỌI TỪ UTILS) ---
            
            # 4.1. Chỉ số bảo mật (So sánh giữa Ảnh gốc và Bản mã C1)
            entropy_v = calculate_entropy(cipher_c1)
            ch, cv, cd = calculate_correlation(cipher_c1)
            npcr_v, uaci_v = calculate_npcr_uaci(cipher_c1, cipher_c2)
            e_psnr = calculate_psnr(self.img_orig, cipher_c1)
            e_ssim = calculate_ssim(self.img_orig, cipher_c1)

            # 4.2. Chỉ số khôi phục (So sánh giữa Ảnh gốc và Ảnh sau giải mã)
            d_psnr = calculate_psnr(self.img_orig, decrypted_img)
            d_ssim = calculate_ssim(self.img_orig, decrypted_img)
            d_ber = calculate_ber(self.img_orig, decrypted_img)

            # --- BƯỚC 5: ĐẨY DỮ LIỆU VÀO BẢNG HIỂN THỊ ---
            # Xử lý hiển thị PSNR vô cùng cho chuyên nghiệp
            d_psnr_txt = "∞" if math.isinf(d_psnr) else f"{d_psnr:.2f}"
            # Xử lý BER (nếu bằng 0 thì hiện 0, ngược lại hiện số mũ)
            d_ber_txt = "0" if d_ber == 0 else f"{d_ber:.1e}"

            self.tree.insert("", 0, values=(
                self.filename, 
                f"{entropy_v:.4f}", 
                f"{ch:.4f}", f"{cv:.4f}", f"{cd:.4f}",
                f"{npcr_v:.2f}%", 
                f"{uaci_v:.2f}%", 
                f"{e_psnr:.2f}",
                f"{e_ssim:.4f}",
                d_psnr_txt, 
                f"{d_ssim:.4f}", 
                d_ber_txt
            ))
            
            messagebox.showinfo("Thành công", f"Đã hoàn thành phân tích định lượng cho tệp: {self.filename}")

        except Exception as e:
            messagebox.showerror("Lỗi thực thi", f"Quá trình phân tích tự động gặp lỗi:\n{str(e)}")

    def clear_table(self):
        """Xóa toàn bộ các dòng trong bảng Treeview"""
        for item in self.tree.get_children():
            self.tree.delete(item)

# Khối chạy thử nghiệm
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalAutoAnalysisUI(root)
    root.mainloop()