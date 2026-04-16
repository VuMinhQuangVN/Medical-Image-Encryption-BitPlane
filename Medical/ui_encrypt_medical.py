# Anh_Y_Te/ui_encrypt_medical.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import time
import os
import math
import cv2

# --- IMPORT CÁC THUẬT TOÁN CORE Y TẾ ---
from core.medical_encrypt import encrypt_medical
# --- IMPORT CÁC HÀM PHÂN TÍCH ---
from core.analysis_utils import (
    calculate_entropy, calculate_correlation, calculate_npcr_uaci,
    calculate_psnr, calculate_ssim, get_histogram_image 
)

class MedicalEncryptUI:
    def __init__(self, parent):
        self.parent = parent
        self.file_path = None
        self.original_np = None
        self.encrypted_np = None 
        
        # Biến trạng thái cho dấu tích (Checkbox)
        self.var_analysis = tk.BooleanVar(value=True) # Mặc định là True
        
        self.tk_orig = None
        self.tk_enc = None
        self.tk_h_orig = None
        self.tk_h_enc = None
        
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.parent, bg="#f4f7f6")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- CỘT TRÁI: ĐIỀU KHIỂN & CHỈ SỐ ---
        left_panel = tk.Frame(main_frame, bg="#f4f7f6", width=320)
        left_panel.pack(side="left", fill="y")
        left_panel.pack_propagate(False)

        # 1. Nhập Mật khẩu
        tk.Label(left_panel, text="1. Cấu hình bảo mật:", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
        
        tk.Label(left_panel, text="Mật khẩu SHA-512:", bg="#f4f7f6", font=("Arial", 9)).pack(anchor="w")
        self.ent_pwd = tk.Entry(left_panel, font=("Arial", 11), bd=2)
        self.ent_pwd.insert(0, "Mật mã y tế 2026")
        self.ent_pwd.pack(fill="x", pady=5)

        # THÊM DẤU TÍCH LINH ĐỘNG TẠI ĐÂY
        self.chk_analysis = tk.Checkbutton(
            left_panel, 
            text="Gắn kết nội dung ảnh (Plaintext Sensitivity)", 
            variable=self.var_analysis,
            bg="#f4f7f6",
            activebackground="#f4f7f6",
            font=("Arial", 8, "italic"),
            fg="#2c3e50"
        )
        self.chk_analysis.pack(anchor="w", pady=2)

        # 2. Nạp ảnh
        tk.Label(left_panel, text="2. Chọn ảnh y tế đầu vào:", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15,0))
        tk.Button(left_panel, text="📁 TẢI ẢNH LÊN", command=self.load_image, bg="#34495e", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=5)
        self.lbl_filename = tk.Label(left_panel, text="Chưa chọn ảnh", bg="#f4f7f6", font=("Arial", 8, "italic"), fg="#7f8c8d")
        self.lbl_filename.pack()

        # 3. BẢNG KẾT QUẢ
        res_analysis_frame = tk.LabelFrame(left_panel, text=" Chỉ số an ninh (Encryption Metrics) ", bg="white", font=("Arial", 9, "bold"))
        res_analysis_frame.pack(fill="x", pady=15, padx=2)

        metrics = [
            ("Entropy:", "res_entropy"), 
            ("Corr (Ngang):", "res_corr_h"), 
            ("Corr (Dọc):", "res_corr_v"), 
            ("Corr (Chéo):", "res_corr_d"), 
            ("NPCR:", "res_npcr"), 
            ("UACI:", "res_uaci"),
            ("PSNR(E):", "res_psnr"), 
            ("SSIM(E):", "res_ssim")
        ]
        
        self.label_vars = {}
        for i, (name, var_name) in enumerate(metrics):
            tk.Label(res_analysis_frame, text=name, bg="white", font=("Arial", 9)).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.label_vars[var_name] = tk.Label(res_analysis_frame, text="-", bg="white", font=("Arial", 9, "bold"), fg="#2980b9")
            self.label_vars[var_name].grid(row=i, column=1, sticky="e", padx=15, pady=2)
        res_analysis_frame.columnconfigure(1, weight=1)

        # 4. Thao tác thực thi
        self.btn_encrypt = tk.Button(left_panel, text="🔒 MÃ HÓA & PHÂN TÍCH", command=self.process_encrypt, bg="#27ae60", fg="white", font=("Arial", 10, "bold"), height=2, state="disabled")
        self.btn_encrypt.pack(fill="x", pady=5)

        self.btn_save = tk.Button(left_panel, text="📥 LƯU ẢNH MÃ HÓA", command=self.save_image, bg="#d35400", fg="white", font=("Arial", 10, "bold"), state="disabled")
        self.btn_save.pack(fill="x", pady=5)

        # Log Console
        self.txt_log = tk.Text(left_panel, height=8, font=("Consolas", 9), bg="#1e1e1e", fg="#00ff00", padx=10, pady=5)
        self.txt_log.pack(pady=5, fill="both", expand=True)

        # --- CỘT PHẢI ---
        right_panel = tk.Frame(main_frame, bg="#f4f7f6")
        right_panel.pack(side="right", fill="both", expand=True, padx=(20, 0))

        # Ảnh Gốc
        self.frame_orig = tk.LabelFrame(right_panel, text=" ẢNH Y TẾ GỐC ", bg="white", font=("Arial", 10, "bold"))
        self.frame_orig.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.canvas_orig = tk.Label(self.frame_orig, bg="white")
        self.canvas_orig.pack(fill="both", expand=True)
        self.canvas_hist_orig = tk.Label(self.frame_orig, bg="white")
        self.canvas_hist_orig.pack(fill="both", expand=True, pady=5)

        # Ảnh Mã Hóa
        self.frame_res = tk.LabelFrame(right_panel, text=" BẢN MÃ (CIPHER) ", bg="white", font=("Arial", 10, "bold"))
        self.frame_res.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.canvas_res = tk.Label(self.frame_res, bg="white")
        self.canvas_res.pack(fill="both", expand=True)
        self.canvas_hist_res = tk.Label(self.frame_res, bg="white")
        self.canvas_hist_res.pack(fill="both", expand=True, pady=5)

    def log(self, message):
        self.txt_log.insert(tk.END, f"> {message}\n")
        self.txt_log.see(tk.END)
        self.parent.update()

    def load_image(self):
        # 1. Mở hộp thoại chọn file
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        
        if self.file_path:
            img_cv = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
            
            if img_cv is None:
                messagebox.showerror("Lỗi", "OpenCV không thể đọc được tệp tin này!")
                return

            self.original_np = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            self.lbl_filename.config(text=os.path.basename(self.file_path))

            img_pil = Image.fromarray(self.original_np)
            img_display = img_pil.resize((300, 300), Image.Resampling.LANCZOS)
            self.tk_orig = ImageTk.PhotoImage(img_display)
            self.canvas_orig.config(image=self.tk_orig)
            self.tk_h_orig = get_histogram_image(self.original_np, "Histogram Ảnh Gốc")
            self.canvas_hist_orig.config(image=self.tk_h_orig)
            self.btn_encrypt.config(state="normal")
            self.log(f"Đã nạp thành công (OpenCV): {os.path.basename(self.file_path)}")

    def process_encrypt(self):
        pwd = self.ent_pwd.get()
        if not pwd or self.original_np is None:
            messagebox.showwarning("Cảnh báo", "Thiếu dữ liệu!")
            return

        try:
            # Lấy giá trị từ dấu tích Checkbox
            use_analysis = self.var_analysis.get()
            
            self.log(f"Đang mã hóa (Analysis Mode: {use_analysis})...")
            start_t = time.time()
            
            # 1. THỰC HIỆN MÃ HÓA
            self.encrypted_np, k2 = encrypt_medical(self.original_np, pwd, is_analysis=use_analysis)
            
            # 2. TÍNH NPCR/UACI (Độ nhạy)
            img_mod = self.original_np.copy()
            img_mod[0,0] = (int(img_mod[0,0]) + 1) % 256
            enc_mod, _ = encrypt_medical(img_mod.astype(np.uint8), pwd, is_analysis=use_analysis)
            
            # 3. TÍNH CHỈ SỐ
            entropy_v = calculate_entropy(self.encrypted_np)
            ch, cv, cd = calculate_correlation(self.encrypted_np) 
            npcr_v, uaci_v = calculate_npcr_uaci(self.encrypted_np, enc_mod)
            psnr_v = calculate_psnr(self.original_np, self.encrypted_np)
            ssim_v = calculate_ssim(self.original_np, self.encrypted_np)
            
            # 4. CẬP NHẬT UI
            self.label_vars['res_entropy'].config(text=f"{entropy_v:.4f}")
            self.label_vars['res_corr_h'].config(text=f"{ch:.4f}")
            self.label_vars['res_corr_v'].config(text=f"{cv:.4f}")
            self.label_vars['res_corr_d'].config(text=f"{cd:.4f}")
            self.label_vars['res_npcr'].config(text=f"{npcr_v:.4f}%")
            self.label_vars['res_uaci'].config(text=f"{uaci_v:.2f}%")
            self.label_vars['res_psnr'].config(text=f"{psnr_v:.2f}")
            self.label_vars['res_ssim'].config(text=f"{ssim_v:.4f}")

            enc_pil = Image.fromarray(self.encrypted_np)
            self.tk_enc = ImageTk.PhotoImage(enc_pil.resize((300, 300)))
            self.canvas_res.config(image=self.tk_enc)
            self.tk_h_enc = get_histogram_image(self.encrypted_np, "Histogram Bản Mã")
            self.canvas_hist_res.config(image=self.tk_h_enc)
            
            self.log(f"Mã hóa thành công! k2 = {k2}")
            print("k2:", k2)  
            self.log(f"Hoàn thành trong {time.time() - start_t:.2f}s")
            self.btn_save.config(state="normal")
            
        except Exception as e:
            self.log(f"Lỗi: {e}")
            messagebox.showerror("Lỗi", str(e))

    def save_image(self):
        if self.encrypted_np is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if save_path:
                Image.fromarray(self.encrypted_np).save(save_path)
                messagebox.showinfo("Thành công", "Đã lưu ảnh!")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hệ Thống Mã Hóa Ảnh Y Tế")
    root.geometry("1150x750")
    app = MedicalEncryptUI(root)
    root.mainloop()