import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
import cv2
import time
import math

# --- IMPORT LOGIC HYBRID ĐÃ TÁCH ---
from core.hybrid_encrypt import run_hybrid_logic
from core.hybrid_decrypt import run_hybrid_decrypt_logic

# --- IMPORT CÁC HÀM PHÂN TÍCH ---
from core.analysis_utils import (
    calculate_entropy, calculate_correlation, calculate_npcr_uaci,
    calculate_psnr, calculate_ssim, calculate_ber
)

class HybridAnalysisUI:
    def __init__(self, parent):
        self.parent = parent
        self.img_orig = None
        self.filename = ""
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.parent, bg="#f4f7f6")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- PANEL ĐIỀU KHIỂN ---
        ctrl_frame = tk.LabelFrame(main_frame, text=" Cấu hình phân tích tự động ", bg="#f4f7f6", font=("Arial", 10, "bold"), padx=15, pady=15)
        ctrl_frame.pack(side="left", fill="y", padx=10, pady=10)

        tk.Label(ctrl_frame, text="Mật khẩu Hybrid:", bg="#f4f7f6").pack(anchor="w")
        self.ent_pwd = tk.Entry(ctrl_frame, width=25, font=("Arial", 10), bd=2)
        self.ent_pwd.insert(0, "Mật mã Hybrid 2026")
        self.ent_pwd.pack(pady=5)

        tk.Button(ctrl_frame, text="📁 1. CHỌN ẢNH Y TẾ", command=self.load_image, bg="#34495e", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=10)
        self.lbl_info = tk.Label(ctrl_frame, text="Chưa nạp ảnh", bg="#f4f7f6", font=("Arial", 8, "italic"), fg="#7f8c8d")
        self.lbl_info.pack(pady=(0, 20))

        tk.Button(ctrl_frame, text="🚀 2. CHẠY PHÂN TÍCH TỔNG THỂ", command=self.run_auto_analysis, bg="#27ae60", fg="white", font=("Arial", 10, "bold"), height=2).pack(fill="x", pady=5)
        
        tk.Button(ctrl_frame, text="🗑 Làm mới bảng", command=self.clear_table, bg="#e74c3c", fg="white").pack(fill="x", pady=20)

        # --- BẢNG SỐ LIỆU ---
        res_frame = tk.LabelFrame(main_frame, text=" Kết quả định lượng (Security & Recovery) ", bg="white", font=("Arial", 10, "bold"), padx=10, pady=10)
        res_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        columns = ("name", "entropy", "corr_h", "corr_v", "corr_d", "npcr", "uaci", "e_psnr", "e_ssim", "d_psnr", "d_ssim", "d_ber")
        self.tree = ttk.Treeview(res_frame, columns=columns, show="headings")
        
        headers = ["Tên ảnh", "Entropy", "Corr(H)", "Corr(V)", "Corr(D)", "NPCR", "UACI", "PSNR(E)", "SSIM(E)", "PSNR(D)", "SSIM(D)", "BER(D)"]
        for col, head in zip(columns, headers):
            self.tree.heading(col, text=head)
            self.tree.column(col, width=80, anchor="center") 

        scrollbar = ttk.Scrollbar(res_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # Ép về 256x256 để chuẩn hóa mọi bước lai ghép
            self.img_orig = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            self.filename = os.path.basename(path)
            self.lbl_info.config(text=f"Đã nạp: {self.filename}", fg="#27ae60")

    def run_auto_analysis(self):
        if self.img_orig is None:
            messagebox.showwarning("Lỗi", "Vui lòng nạp ảnh gốc trước!")
            return
        
        pwd = self.ent_pwd.get()
        if not pwd:
            messagebox.showwarning("Lỗi", "Vui lòng nhập mật khẩu!")
            return

        try:
            start_time = time.time()
            # --- BƯỚC 1: MÃ HÓA (C1) ---
            cipher_c1, q_val = run_hybrid_logic(self.img_orig, pwd)

            # --- BƯỚC 2: TÍNH VI SAI (NPCR/UACI) ---
            # Tạo ảnh lỗi 1 bit
            img_mod = self.img_orig.copy()
            img_mod[0, 0] = (int(img_mod[0, 0]) + 1) % 256
            # Mã hóa ảnh lỗi (C2)
            cipher_c2, _ = run_hybrid_logic(img_mod.astype(np.uint8), pwd)

            # --- BƯỚC 3: GIẢI MÃ (Khôi phục) ---
            decrypted_img = run_hybrid_decrypt_logic(cipher_c1, pwd)

            # --- BƯỚC 4: TÍNH TOÁN CÁC CHỈ SỐ ---
            # 4.1. Chỉ số bảo mật
            ent_v = calculate_entropy(cipher_c1)
            ch, cv, cd = calculate_correlation(cipher_c1)
            npcr_v, uaci_v = calculate_npcr_uaci(cipher_c1, cipher_c2)
            e_psnr = calculate_psnr(self.img_orig, cipher_c1)
            e_ssim = calculate_ssim(self.img_orig, cipher_c1)

            # 4.2. Chỉ số khôi phục
            d_psnr = calculate_psnr(self.img_orig, decrypted_img)
            d_ssim = calculate_ssim(self.img_orig, decrypted_img)
            d_ber = calculate_ber(self.img_orig, decrypted_img)

            # --- BƯỚC 5: HIỂN THỊ ---
            d_psnr_txt = "∞" if math.isinf(d_psnr) else f"{d_psnr:.2f}"
            d_ber_txt = "0" if d_ber == 0 else f"{d_ber:.1e}"

            self.tree.insert("", 0, values=(
                self.filename, 
                f"{ent_v:.4f}",
                f"{ch:.4f}", f"{cv:.4f}", f"{cd:.4f}",
                f"{npcr_v:.2f}%", f"{uaci_v:.2f}%", 
                f"{e_psnr:.2f}", f"{e_ssim:.4f}",
                d_psnr_txt, f"{d_ssim:.4f}", d_ber_txt
            ))
            
            
            duration = time.time() - start_time
            messagebox.showinfo("Thành công", f"Đã phân tích xong: {self.filename}\nThời gian xử lý: {duration:.2f}s")

        except Exception as e:
            messagebox.showerror("Lỗi", f"Quá trình phân tích gặp lỗi:\n{str(e)}")

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hệ Thống Phân Tích Tự Động Ảnh Hybrid - Đồ Án 2026")
    root.geometry("1150x550")
    HybridAnalysisUI(root)
    root.mainloop()