import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
import os
import math

# --- IMPORT CÁC THÀNH PHẦN HYBRID TỪ CORE ---
from core.hybrid_utils import get_sha512_hex, get_sha512_ints, bit_plane_slice, bit_plane_rejoin, arnold_transform, compute_N0_Q_hybrid
from core.key_generator import get_all_keys_hybrid
from core.hybrid_diffusion import hybrid_diffusion_forward
from core.analysis_utils import calculate_entropy, calculate_correlation, calculate_npcr_uaci, calculate_psnr, calculate_ssim, get_histogram_image
from core.hybrid_encrypt import run_hybrid_logic

class HybridEncryptUI:
    def __init__(self, parent):
        self.parent = parent
        self.original_np = None
        self.encrypted_np = None
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.parent, bg="#f4f7f6")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- CỘT TRÁI: ĐIỀU KHIỂN & CHỈ SỐ ---
        left_panel = tk.Frame(main_frame, bg="#f4f7f6", width=330)
        left_panel.pack(side="left", fill="y")
        left_panel.pack_propagate(False)

        tk.Label(left_panel, text="1. Hệ thống Khóa (Hybrid):", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
        self.ent_pwd = tk.Entry(left_panel, font=("Arial", 11), bd=2)
        self.ent_pwd.insert(0, "Mật mã Hybrid 2026")
        self.ent_pwd.pack(fill="x", pady=5)

        tk.Button(left_panel, text="📁 TẢI ẢNH GỐC (CV2)", command=self.load_image, bg="#34495e", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=5)
        self.lbl_filename = tk.Label(left_panel, text="Chưa nạp ảnh", bg="#f4f7f6", font=("Arial", 8, "italic"), fg="#7f8c8d")
        self.lbl_filename.pack()

        # Bảng chỉ số định lượng
        metrics_frame = tk.LabelFrame(left_panel, text=" Phân tích định lượng (Hybrid) ", bg="white", font=("Arial", 9, "bold"))
        metrics_frame.pack(fill="x", pady=15, padx=2)

        metrics = [
            ("Entropy:", "res_ent"), 
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
            tk.Label(metrics_frame, text=name, bg="white").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.label_vars[var_name] = tk.Label(metrics_frame, text="-", bg="white", font=("Arial", 9, "bold"), fg="#2980b9")
            self.label_vars[var_name].grid(row=i, column=1, sticky="e", padx=15, pady=2)
        metrics_frame.columnconfigure(1, weight=1)

        self.btn_encrypt = tk.Button(left_panel, text="🔒 MÃ HÓA HYBRID", command=self.process_encrypt, bg="#27ae60", fg="white", font=("Arial", 10, "bold"), height=2, state="disabled")
        self.btn_encrypt.pack(fill="x", pady=5)

        self.btn_save = tk.Button(left_panel, text="📥 LƯU BẢN MÃ", command=self.save_image, bg="#d35400", fg="white", font=("Arial", 10, "bold"), state="disabled")
        self.btn_save.pack(fill="x", pady=5)

        self.txt_log = tk.Text(left_panel, height=8, font=("Consolas", 9), bg="#1e1e1e", fg="#00ff00", padx=10, pady=5)
        self.txt_log.pack(pady=5, fill="both", expand=True)

        # --- CỘT PHẢI: HIỂN THỊ ---
        right_panel = tk.Frame(main_frame, bg="#f4f7f6")
        right_panel.pack(side="right", fill="both", expand=True, padx=(20, 0))

        # View Ảnh Gốc
        self.f_orig = tk.LabelFrame(right_panel, text=" ẢNH GỐC (256x256) ", bg="white", font=("Arial", 10, "bold"))
        self.f_orig.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.canvas_orig = tk.Label(self.f_orig, bg="white")
        self.canvas_orig.pack(fill="both", expand=True)
        self.canvas_hist_orig = tk.Label(self.f_orig, bg="white")
        self.canvas_hist_orig.pack(fill="both", expand=True, pady=5)

        # View Bản Mã
        self.f_res = tk.LabelFrame(right_panel, text=" BẢN MÃ (HYBRID CIPHER) ", bg="white", font=("Arial", 10, "bold"))
        self.f_res.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.canvas_res = tk.Label(self.f_res, bg="white")
        self.canvas_res.pack(fill="both", expand=True)
        self.canvas_hist_res = tk.Label(self.f_res, bg="white")
        self.canvas_hist_res.pack(fill="both", expand=True, pady=5)

    def log(self, message):
        self.txt_log.insert(tk.END, f"> {message}\n")
        self.txt_log.see(tk.END)
        self.parent.update()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            # OpenCV Load & Resize
            img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.original_np = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            self.lbl_filename.config(text=os.path.basename(path))
            
            # Hiển thị
            img_pil = Image.fromarray(self.original_np)
            self.tk_orig = ImageTk.PhotoImage(img_pil.resize((300, 300)))
            self.canvas_orig.config(image=self.tk_orig)
            
            self.tk_h_orig = get_histogram_image(self.original_np, "Histogram Gốc")
            self.canvas_hist_orig.config(image=self.tk_h_orig)
            
            self.btn_encrypt.config(state="normal")
            self.log(f"Đã nạp: {os.path.basename(path)}")

    def process_encrypt(self):
        pwd = self.ent_pwd.get()
        if not pwd or self.original_np is None: return

        try:
            self.log("Bắt đầu Hybrid Encryption...")
            start_t = time.time()
            
            # 1. Mã hóa chính
            self.encrypted_np, q_val = run_hybrid_logic(self.original_np, pwd)
            
            # 2. Tính NPCR/UACI (Tự động tạo ảnh lỗi 1 pixel)
            img_mod = self.original_np.copy()
            img_mod[0,0] = (int(img_mod[0,0]) + 1) % 256
            enc_mod, _ = run_hybrid_logic(img_mod.astype(np.uint8), pwd)
            
            # 3. Tính các chỉ số định lượng
            ent_v = calculate_entropy(self.encrypted_np)
            ch, cv, cd = calculate_correlation(self.encrypted_np) 
            npcr_v, uaci_v = calculate_npcr_uaci(self.encrypted_np, enc_mod)
            psnr_v = calculate_psnr(self.original_np, self.encrypted_np)
            ssim_v = calculate_ssim(self.original_np, self.encrypted_np)
            
            # 4. Cập nhật UI
            self.label_vars['res_ent'].config(text=f"{ent_v:.4f}")
            self.label_vars['res_corr_h'].config(text=f"{ch:.4f}")
            self.label_vars['res_corr_v'].config(text=f"{cv:.4f}")
            self.label_vars['res_corr_d'].config(text=f"{cd:.4f}")
            self.label_vars['res_npcr'].config(text=f"{npcr_v:.4f}%")
            self.label_vars['res_uaci'].config(text=f"{uaci_v:.2f}%")
            self.label_vars['res_psnr'].config(text=f"{psnr_v:.2f}")
            self.label_vars['res_ssim'].config(text=f"{ssim_v:.4f}")

            # Hiển thị kết quả
            enc_pil = Image.fromarray(self.encrypted_np)
            self.tk_enc = ImageTk.PhotoImage(enc_pil.resize((300, 300)))
            self.canvas_res.config(image=self.tk_enc)
            
            self.tk_h_enc = get_histogram_image(self.encrypted_np, "Histogram Bản Mã")
            self.canvas_hist_res.config(image=self.tk_h_enc)
            
            self.log(f"Thành công! Pixel mồi Q: {q_val}")
            self.log(f"Thời gian: {time.time() - start_t:.3f}s")
            self.btn_save.config(state="normal")
            
        except Exception as e:
            self.log(f"Lỗi: {e}")
            messagebox.showerror("Lỗi", str(e))

    def save_image(self):
        if self.encrypted_np is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if path:
                cv2.imwrite(path, self.encrypted_np)
                messagebox.showinfo("Xong", "Đã lưu bản mã Hybrid!")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hệ Thống Mã Hóa Ảnh Hybrid")
    root.geometry("1150x750")
    HybridEncryptUI(root)
    root.mainloop()