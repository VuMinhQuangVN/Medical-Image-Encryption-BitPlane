import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Anh_Y_Te_Hybrid/ui_hybrid_decrypt.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
import math

# --- IMPORT CÁC THÀNH PHẦN HYBRID TỪ CORE ---
from core.hybrid_utils import get_sha512_hex, get_sha512_ints, bit_plane_slice, bit_plane_rejoin, inverse_arnold, compute_N0_Q_hybrid
from core.key_generator import get_all_keys_hybrid
from core.hybrid_diffusion import hybrid_diffusion_backward
from core.analysis_utils import calculate_psnr, calculate_ssim, calculate_ber, get_histogram_image
from core.hybrid_decrypt import run_hybrid_decrypt_logic

class HybridDecryptUI:
    def __init__(self, parent):
        self.parent = parent
        self.cipher_np = None
        self.original_np = None
        self.decrypted_np = None
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.parent, bg="#f4f7f6")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- CỘT TRÁI ---
        left_panel = tk.Frame(main_frame, bg="#f4f7f6", width=330)
        left_panel.pack(side="left", fill="y")
        left_panel.pack_propagate(False)

        tk.Label(left_panel, text="1. Nhập Mật khẩu giải mã:", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
        self.ent_pwd = tk.Entry(left_panel, font=("Arial", 11), bd=2)
        self.ent_pwd.insert(0, "Mật mã Hybrid 2026")
        self.ent_pwd.pack(fill="x", pady=5)

        tk.Label(left_panel, text="2. Tải dữ liệu:", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15,0))
        tk.Button(left_panel, text="📁 TẢI BẢN MÃ (CIPHER)", command=self.load_cipher, bg="#34495e", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=2)
        tk.Button(left_panel, text="🖼️ NẠP ẢNH GỐC (ĐỐI CHỨNG)", command=self.load_original, bg="#7f8c8d", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=2)
        self.lbl_orig_info = tk.Label(left_panel, text="Chưa nạp ảnh đối chứng", bg="#f4f7f6", font=("Arial", 8, "italic"), fg="#7f8c8d")
        self.lbl_orig_info.pack()

        # Bảng chỉ số
        metrics_frame = tk.LabelFrame(left_panel, text=" Độ chính xác khôi phục ", bg="white", font=("Arial", 9, "bold"))
        metrics_frame.pack(fill="x", pady=15, padx=2)
        self.label_vars = {}
        for i, (name, var) in enumerate([("PSNR(D):", "res_psnr"), ("SSIM(D):", "res_ssim"), ("BER(D):", "res_ber")]):
            tk.Label(metrics_frame, text=name, bg="white").grid(row=i, column=0, sticky="w", padx=5, pady=5)
            self.label_vars[var] = tk.Label(metrics_frame, text="-", bg="white", font=("Arial", 10, "bold"), fg="#27ae60")
            self.label_vars[var].grid(row=i, column=1, sticky="e", padx=15, pady=5)
        metrics_frame.columnconfigure(1, weight=1)

        self.btn_decrypt = tk.Button(left_panel, text="🔓 GIẢI MÃ HYBRID TỰ ĐỘNG", command=self.process_decrypt, 
                                     bg="#8e44ad", fg="white", font=("Arial", 10, "bold"), height=2, state="disabled")
        self.btn_decrypt.pack(fill="x", pady=5)

        self.btn_save = tk.Button(left_panel, text="📥 LƯU ẢNH GIẢI MÃ", command=self.save_image, 
                                  bg="#d35400", fg="white", font=("Arial", 10, "bold"), state="disabled")
        self.btn_save.pack(fill="x", pady=5)

        self.txt_log = tk.Text(left_panel, height=10, font=("Consolas", 9), bg="#1e1e1e", fg="#00ff00", padx=10, pady=5)
        self.txt_log.pack(pady=5, fill="both", expand=True)

        # --- CỘT PHẢI ---
        right_panel = tk.Frame(main_frame, bg="#f4f7f6")
        right_panel.pack(side="right", fill="both", expand=True, padx=(20, 0))
        
        # Khung Cipher
        f_cip = tk.LabelFrame(right_panel, text=" BẢN MÃ ĐẦU VÀO ", bg="white", font=("Arial", 10, "bold"))
        f_cip.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.canvas_cipher = tk.Label(f_cip, bg="white"); self.canvas_cipher.pack(fill="both", expand=True)
        self.canvas_hist_cipher = tk.Label(f_cip, bg="white"); self.canvas_hist_cipher.pack(fill="both", expand=True, pady=5)

        # Khung Plain
        f_pla = tk.LabelFrame(right_panel, text=" ẢNH KHÔI PHỤC ", bg="white", font=("Arial", 10, "bold"))
        f_pla.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.canvas_plain = tk.Label(f_pla, bg="white"); self.canvas_plain.pack(fill="both", expand=True)
        self.canvas_hist_plain = tk.Label(f_pla, bg="white"); self.canvas_hist_plain.pack(fill="both", expand=True, pady=5)

    def log(self, message):
        self.txt_log.insert(tk.END, f"> {message}\n"); self.txt_log.see(tk.END); self.parent.update()

    def load_cipher(self):
        path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if path:
            img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.cipher_np = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            self.tk_cipher = ImageTk.PhotoImage(Image.fromarray(self.cipher_np).resize((300, 300)))
            self.canvas_cipher.config(image=self.tk_cipher)
            self.tk_h_cipher = get_histogram_image(self.cipher_np, "Histogram Bản Mã")
            self.canvas_hist_cipher.config(image=self.tk_h_cipher)
            
            # Bây giờ biến này đã tồn tại, không còn lỗi AttributeError
            self.btn_decrypt.config(state="normal") 
            self.log(f"Đã nạp bản mã: {os.path.basename(path)}")

    def load_original(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.bmp")])
        if path:
            img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.original_np = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            self.lbl_orig_info.config(text=f"Đã khớp: {os.path.basename(path)}", fg="#27ae60")
            self.log("Đã nạp ảnh đối chứng.")

    def process_decrypt(self):
        pwd = self.ent_pwd.get()
        if not pwd or self.cipher_np is None: return
        try:
            self.log("Đang giải mã Hybrid (Tự động khôi phục khóa)...")
            start_t = time.time()
            self.decrypted_np = run_hybrid_decrypt_logic(self.cipher_np, pwd)
            
            # 3. Tính toán đối chứng
            if self.original_np is not None:
                psnr_v = calculate_psnr(self.original_np, self.decrypted_np)
                self.label_vars['res_psnr'].config(text="∞" if math.isinf(psnr_v) else f"{psnr_v:.2f}")
                self.label_vars['res_ssim'].config(text=f"{calculate_ssim(self.original_np, self.decrypted_np):.4f}")
                ber_v = calculate_ber(self.original_np, self.decrypted_np)
                self.label_vars['res_ber'].config(text="0" if ber_v == 0 else f"{ber_v:.1e}")

            # 4. Hiển thị kết quả
            self.tk_plain = ImageTk.PhotoImage(Image.fromarray(self.decrypted_np).resize((300, 300)))
            self.canvas_plain.config(image=self.tk_plain)
            self.tk_h_plain = get_histogram_image(self.decrypted_np, "Histogram Giải Mã")
            self.canvas_hist_plain.config(image=self.tk_h_plain)
            
            self.log(f"Giải mã hoàn tất trong {time.time() - start_t:.3f}s")
            self.btn_save.config(state="normal")
        except Exception as e:
            self.log(f"Lỗi: {e}")
            messagebox.showerror("Lỗi giải mã", str(e))

    def save_image(self):
        if self.decrypted_np is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if path:
                cv2.imwrite(path, self.decrypted_np)
                messagebox.showinfo("Thành công", "Đã lưu ảnh giải mã!")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hệ Thống Giải Mã Ảnh Hybrid")
    root.geometry("1150x750")
    HybridDecryptUI(root)
    root.mainloop()