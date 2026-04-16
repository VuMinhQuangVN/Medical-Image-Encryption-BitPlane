# Anh_Y_Te/ui_decrypt_medical.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import time
import os
import math
import cv2

# --- IMPORT CÁC THUẬT TOÁN CORE Y TẾ ---
from core.medical_decrypt import decrypt_medical
# --- IMPORT CÁC HÀM PHÂN TÍCH ---
from core.analysis_utils import (
    calculate_psnr, calculate_ssim, calculate_ber, get_histogram_image 
)

class MedicalDecryptUI:
    def __init__(self, parent):
        self.parent = parent
        self.cipher_np = None     
        self.original_np = None   
        self.decrypted_np = None  
        
        self.tk_cipher = None
        self.tk_plain = None
        self.tk_h_cipher = None
        self.tk_h_plain = None
        
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.parent, bg="#f4f7f6")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- CỘT TRÁI: ĐIỀU KHIỂN ---
        left_panel = tk.Frame(main_frame, bg="#f4f7f6", width=320)
        left_panel.pack(side="left", fill="y")
        left_panel.pack_propagate(False)

        # 1. Nhập Mật khẩu & k2
        tk.Label(left_panel, text="1. Thông tin giải mã:", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
        
        tk.Label(left_panel, text="Mật khẩu (Key 1):", bg="#f4f7f6", font=("Arial", 9)).pack(anchor="w")
        self.ent_pwd = tk.Entry(left_panel, font=("Arial", 11), bd=2)
        self.ent_pwd.insert(0, "Mật mã y tế 2026")
        self.ent_pwd.pack(fill="x", pady=2)

        tk.Label(left_panel, text="Tham số k2 (từ bản mã):", bg="#f4f7f6", font=("Arial", 9)).pack(anchor="w", pady=(5,0))
        self.ent_k2 = tk.Entry(left_panel, font=("Consolas", 11), bd=2, fg="red", justify="center")
        self.ent_k2.pack(fill="x", pady=2)

        # 2. Nạp file
        tk.Label(left_panel, text="2. Tải tệp tin dữ liệu:", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15,0))
        tk.Button(left_panel, text="📁 TẢI ẢNH MÃ HÓA (CIPHER)", command=self.load_cipher, bg="#34495e", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=2)
        
        tk.Button(left_panel, text="🖼️ NẠP ẢNH GỐC (TÙY CHỌN)", command=self.load_original, bg="#7f8c8d", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=2)
        self.lbl_orig_info = tk.Label(left_panel, text="Chưa nạp ảnh đối chứng (Sẽ hiện N/A)", bg="#f4f7f6", font=("Arial", 8, "italic"), fg="#7f8c8d")
        self.lbl_orig_info.pack()

        # 3. BẢNG CHỈ SỐ KHÔI PHỤC
        res_analysis_frame = tk.LabelFrame(left_panel, text=" Độ chính xác khôi phục ", bg="white", font=("Arial", 9, "bold"))
        res_analysis_frame.pack(fill="x", pady=15, padx=2)

        metrics = [("PSNR(D):", "res_psnr"), ("SSIM(D):", "res_ssim"), ("BER(D):", "res_ber")]
        self.label_vars = {}
        for i, (name, var_name) in enumerate(metrics):
            tk.Label(res_analysis_frame, text=name, bg="white", font=("Arial", 9)).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            self.label_vars[var_name] = tk.Label(res_analysis_frame, text="-", bg="white", font=("Arial", 10, "bold"), fg="#27ae60")
            self.label_vars[var_name].grid(row=i, column=1, sticky="e", padx=15, pady=5)
        res_analysis_frame.columnconfigure(1, weight=1)

        # 4. Thao tác thực thi
        self.btn_decrypt = tk.Button(left_panel, text="🔓 GIẢI MÃ HỆ THỐNG", command=self.process_decrypt, bg="#8e44ad", fg="white", font=("Arial", 10, "bold"), height=2, state="disabled")
        self.btn_decrypt.pack(fill="x", pady=5)

        self.btn_save = tk.Button(left_panel, text="📥 LƯU ẢNH GIẢI MÃ", command=self.save_image, bg="#d35400", fg="white", font=("Arial", 10, "bold"), state="disabled")
        self.btn_save.pack(fill="x", pady=5)

        # Log Console
        self.txt_log = tk.Text(left_panel, height=6, font=("Consolas", 9), bg="#1e1e1e", fg="#00ff00", padx=10, pady=5)
        self.txt_log.pack(pady=5, fill="both", expand=True)

        # --- CỘT PHẢI ---
        right_panel = tk.Frame(main_frame, bg="#f4f7f6")
        right_panel.pack(side="right", fill="both", expand=True, padx=(20, 0))

        # Trái: Cipher
        self.frame_cipher = tk.LabelFrame(right_panel, text=" ẢNH MÃ HÓA (INPUT) ", bg="white", font=("Arial", 10, "bold"))
        self.frame_cipher.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.canvas_cipher = tk.Label(self.frame_cipher, bg="white")
        self.canvas_cipher.pack(fill="both", expand=True)
        self.canvas_hist_cipher = tk.Label(self.frame_cipher, bg="white")
        self.canvas_hist_cipher.pack(fill="both", expand=True, pady=5)

        # Phải: Plain
        self.frame_plain = tk.LabelFrame(right_panel, text=" ẢNH GIẢI MÃ (OUTPUT) ", bg="white", font=("Arial", 10, "bold"))
        self.frame_plain.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.canvas_plain = tk.Label(self.frame_plain, bg="white")
        self.canvas_plain.pack(fill="both", expand=True)
        self.canvas_hist_plain = tk.Label(self.frame_plain, bg="white")
        self.canvas_hist_plain.pack(fill="both", expand=True, pady=5)

    def log(self, message):
        self.txt_log.insert(tk.END, f"> {message}\n")
        self.txt_log.see(tk.END)
        self.parent.update()

    def load_cipher(self):
        path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        
        if path:
            img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if img_cv is None:
                messagebox.showerror("Lỗi", "OpenCV không thể đọc được tệp tin bản mã này!")
                return

            self.cipher_np = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            img_pil = Image.fromarray(self.cipher_np)
            img_display = img_pil.resize((300, 300), Image.Resampling.LANCZOS)
            
            self.tk_cipher = ImageTk.PhotoImage(img_display)
            self.canvas_cipher.config(image=self.tk_cipher)

            self.tk_h_cipher = get_histogram_image(self.cipher_np, "Histogram Bản Mã")
            self.canvas_hist_cipher.config(image=self.tk_h_cipher)

            self.btn_decrypt.config(state="normal")
            filename = os.path.basename(path)
            self.log(f"Đã nạp bản mã (OpenCV): {filename}")

    def load_original(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        
        if path:
            img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if img_cv is None:
                messagebox.showerror("Lỗi", "OpenCV không thể đọc được file ảnh gốc này!")
                return

            self.original_np = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            filename = os.path.basename(path)
            self.lbl_orig_info.config(text=f"Đối chứng: {filename}", fg="#27ae60")
            self.log(f"Đã nạp ảnh gốc đối chứng (OpenCV): {filename}")

    def process_decrypt(self):
        pwd = self.ent_pwd.get()
        k2_str = self.ent_k2.get()
        if not pwd or not k2_str or self.cipher_np is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đủ thông tin!")
            return

        try:
            start_t = time.time()
            k2_val = float(k2_str)

            # --- LOGIC LINH ĐỘNG ---
            if self.original_np is not None:
                self.log("Giải mã với tham chiếu ảnh gốc (Mode Analysis)...")
                # Truyền original_ref để hàm băm SHA-512 khớp với lúc mã hóa
                self.decrypted_np = decrypt_medical(self.cipher_np, pwd, k2_val, original_ref=self.original_np)
                
                # Tính toán chỉ số
                psnr_v = calculate_psnr(self.original_np, self.decrypted_np)
                ssim_v = calculate_ssim(self.original_np, self.decrypted_np)
                ber_v = calculate_ber(self.original_np, self.decrypted_np)
                
                psnr_txt = "∞" if math.isinf(psnr_v) else f"{psnr_v:.2f}"
                ber_txt = "0" if ber_v == 0 else f"{ber_v:.2e}"
                
                self.label_vars['res_psnr'].config(text=psnr_txt)
                self.label_vars['res_ssim'].config(text=f"{ssim_v:.4f}")
                self.label_vars['res_ber'].config(text=ber_txt)
            else:
                # Giải mã không có ảnh tham chiếu (chỉ dùng mật khẩu)
                self.decrypted_np = decrypt_medical(self.cipher_np, pwd, k2_val, original_ref=None)
                
                # Gán giá trị N/A cho bảng chỉ số
                self.label_vars['res_psnr'].config(text="N/A", fg="#7f8c8d")
                self.label_vars['res_ssim'].config(text="N/A", fg="#7f8c8d")
                self.label_vars['res_ber'].config(text="N/A", fg="#7f8c8d")

            # Hiển thị ảnh & Histogram
            plain_pil = Image.fromarray(self.decrypted_np)
            self.tk_plain = ImageTk.PhotoImage(plain_pil.resize((300, 300)))
            self.canvas_plain.config(image=self.tk_plain)
            self.tk_h_plain = get_histogram_image(self.decrypted_np, "Histogram Giải Mã")
            self.canvas_hist_plain.config(image=self.tk_h_plain)
            
            self.log(f"Xong! Thời gian: {time.time() - start_t:.3f}s")
            self.btn_save.config(state="normal")
            
        except Exception as e:
            self.log(f"Lỗi: {e}")
            messagebox.showerror("Lỗi", str(e))

    def save_image(self):
        if self.decrypted_np is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if save_path:
                Image.fromarray(self.decrypted_np).save(save_path)
                messagebox.showinfo("Thành công", "Đã lưu ảnh!")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hệ Thống Giải Mã Ảnh Y Tế")
    root.geometry("1150x750")
    app = MedicalDecryptUI(root)
    root.mainloop()