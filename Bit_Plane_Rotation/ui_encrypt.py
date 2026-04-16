# File: ui_encrypt.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import os
import math

# Import các thuật toán từ thư mục core
from core.md5_custom import derive_initial_values, my_md5_string_tool
from core.key_generator import generate_all_keys
from core.encrypt_image import encrypt_image
# Import các hàm phân tích
from core.analysis_utils import (
    calculate_entropy, calculate_correlation, calculate_npcr_uaci,
    calculate_psnr, calculate_ssim, get_histogram_image 
)

class EncryptUI:
    def __init__(self, parent):
        self.parent = parent
        self.res_md5 = None
        self.file_path = None
        self.original_np = None
        self.encrypted_np = None 
        self.generated_keys = None
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.parent, bg="#f4f7f6")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- CỘT TRÁI: ĐIỀU KHIỂN ---
        left_panel = tk.Frame(main_frame, bg="#f4f7f6", width=320)
        left_panel.pack(side="left", fill="y")
        left_panel.pack_propagate(False)

        # 1. Nhập Seed
        tk.Label(left_panel, text="1. Chuỗi bảo mật (Seed):", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
        self.ent_seed = tk.Entry(left_panel, font=("Arial", 11), bd=2)
        self.ent_seed.insert(0, "Mật mã đồ án 2026")
        self.ent_seed.pack(fill="x", pady=5)

        # 2. Chọn ảnh
        tk.Label(left_panel, text="2. Chọn ảnh gốc (Original):", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15,0))
        tk.Button(left_panel, text="📁 TẢI ẢNH LÊN", command=self.load_image, bg="#34495e", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=5)
        self.lbl_filename = tk.Label(left_panel, text="Chưa chọn ảnh", bg="#f4f7f6", font=("Arial", 8, "italic"), fg="#7f8c8d")
        self.lbl_filename.pack()

        # 3. KẾT QUẢ PHÂN TÍCH (Ô hiển thị giống trong ảnh bạn yêu cầu)
        res_analysis_frame = tk.LabelFrame(left_panel, text=" Bảng chỉ số mã hóa (Encryption) ", bg="white", font=("Arial", 9, "bold"))
        res_analysis_frame.pack(fill="x", pady=15, padx=2)

        # Tạo bảng 2 cột đơn giản bằng Label
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

        # 4. Thao tác Mã hóa
        self.btn_encrypt = tk.Button(left_panel, text="🔒 MÃ HÓA & PHÂN TÍCH", command=self.process_encrypt, bg="#27ae60", fg="white", font=("Arial", 10, "bold"), height=2, state="disabled")
        self.btn_encrypt.pack(fill="x", pady=5)

        self.btn_save = tk.Button(left_panel, text="📥 LƯU ẢNH MÃ HÓA", command=self.save_image, bg="#d35400", fg="white", font=("Arial", 10, "bold"), state="disabled")
        self.btn_save.pack(fill="x", pady=5)

        # Log Console (nhỏ lại một chút để nhường chỗ cho bảng)
        self.txt_log = tk.Text(left_panel, height=8, font=("Consolas", 9), bg="#1e1e1e", fg="#00ff00", padx=10, pady=5)
        self.txt_log.pack(pady=5, fill="both", expand=True)

        # --- CỘT PHẢI: HIỂN THỊ ẢNH ---
        right_panel = tk.Frame(main_frame, bg="#f4f7f6")
        right_panel.pack(side="right", fill="both", expand=True, padx=(20, 0))

        self.frame_orig = tk.LabelFrame(right_panel, text=" ẢNH GỐC ", bg="white", font=("Arial", 10, "bold"))
        self.frame_orig.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.canvas_orig = tk.Label(self.frame_orig, bg="white")
        self.canvas_orig.pack(fill="both", expand=True)
        self.canvas_hist_orig = tk.Label(self.frame_orig, bg="white")
        self.canvas_hist_orig.pack(fill="both", expand=True, pady=5)

        self.frame_res = tk.LabelFrame(right_panel, text=" ẢNH MÃ HÓA ", bg="white", font=("Arial", 10, "bold"))
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
        # 1. Mở hộp thoại chọn file ảnh gốc
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        
        if self.file_path:
            # 2. Đọc ảnh bằng OpenCV ở chế độ Ảnh Xám (Grayscale)
            img_cv = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
            
            # Kiểm tra an toàn xem file có lỗi không
            if img_cv is None:
                messagebox.showerror("Lỗi", "OpenCV không thể đọc được tệp tin này!")
                return

            # 3. CHUẨN HÓA KÍCH THƯỚC (256x256)
            # Dùng LANCZOS4 để giữ chi tiết ảnh tốt nhất cho quá trình mã hóa
            self.original_np = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # Cập nhật nhãn tên file trên giao diện
            self.lbl_filename.config(text=os.path.basename(self.file_path))

            # 4. CHUẨN BỊ HIỂN THỊ TRÊN CANVAS (350x350)
            # Chuyển NumPy (OpenCV) -> PIL Image -> ImageTk để Tkinter hiểu được
            img_pil = Image.fromarray(self.original_np)
            
            # Resize nhẹ lên 350x350 để hiển thị cho đẹp trong khung ảnh gốc
            img_preview = img_pil.resize((350, 350), Image.Resampling.LANCZOS)
            self.original_img_tk = ImageTk.PhotoImage(img_preview)
            self.canvas_orig.config(image=self.original_img_tk)

            # 5. VẼ HISTOGRAM ẢNH GỐC
            # Sử dụng mảng NumPy 256x256 đã chuẩn hóa để vẽ biểu đồ
            self.hist_orig_tk = get_histogram_image(self.original_np, "Histogram Gốc")
            self.canvas_hist_orig.config(image=self.hist_orig_tk)

            # 6. CẬP NHẬT TRẠNG THÁI HỆ THỐNG
            self.btn_encrypt.config(state="normal")
            self.log(f"Đã nạp & chuẩn hóa (CV2): {os.path.basename(self.file_path)}")
            
    def process_encrypt(self):
        # 1. Chuẩn bị hệ thống khóa
        seed = self.ent_seed.get()
        if not seed: return
        self.res_md5 = my_md5_string_tool(seed)
        p = derive_initial_values(self.res_md5['md5_hex'])
        m, n = self.original_np.shape
        self.generated_keys = generate_all_keys(p['x0'], p['y0'], p['z0'], p['w0'], p['x10'], p['x20'], p['x30'], p['x40'], p['x50'], p['x60'], self.res_md5['N0'], m, n)

        try:
            self.log("Bắt đầu mã hóa & tính toán an toàn...")
            start_t = time.time()
            
            # 2. Thực hiện mã hóa ảnh thực tế
            self.encrypted_np = encrypt_image(self.original_np, self.generated_keys, self.res_md5['Q'])
            
            # 3. TỰ ĐỘNG: Tạo ảnh biến đổi 1 bit để tính NPCR/UACI
            # (Thay đổi pixel đầu tiên [0,0] tăng 1 đơn vị)
            img_mod = self.original_np.copy()
            img_mod[0,0] = (int(img_mod[0,0]) + 1) % 256
            enc_mod = encrypt_image(img_mod.astype(np.uint8), self.generated_keys, self.res_md5['Q'])
            
            # 4. TÍNH TOÁN CÁC CHỈ SỐ
            entropy_v = calculate_entropy(self.encrypted_np)
            ch, cv, cd = calculate_correlation(self.encrypted_np) 
            npcr_v, uaci_v = calculate_npcr_uaci(self.encrypted_np, enc_mod)
            psnr_v = calculate_psnr(self.original_np, self.encrypted_np)
            ssim_v = calculate_ssim(self.original_np, self.encrypted_np)
            
            # 5. CẬP NHẬT LÊN BẢNG CHỈ SỐ
            self.label_vars['res_entropy'].config(text=f"{entropy_v:.4f}")
            self.label_vars['res_corr_h'].config(text=f"{ch:.4f}")
            self.label_vars['res_corr_v'].config(text=f"{cv:.4f}")
            self.label_vars['res_corr_d'].config(text=f"{cd:.4f}")
            self.label_vars['res_npcr'].config(text=f"{npcr_v:.4f}%")
            self.label_vars['res_uaci'].config(text=f"{uaci_v:.2f}%")
            self.label_vars['res_psnr'].config(text=f"{psnr_v:.2f}")
            self.label_vars['res_ssim'].config(text=f"{ssim_v:.4f}")

            # 6. HIỂN THỊ ẢNH KẾT QUẢ & HISTOGRAM
            res_img = Image.fromarray(self.encrypted_np)
            preview_res = res_img.copy()
            preview_res.thumbnail((350, 350))
            self.res_img_tk = ImageTk.PhotoImage(preview_res)
            self.canvas_res.config(image=self.res_img_tk)
            
            self.hist_res_tk = get_histogram_image(self.encrypted_np, "Histogram Mã Hóa")
            self.canvas_hist_res.config(image=self.hist_res_tk)
            
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
    root.title("Hệ Thống Mã Hóa Ảnh Thường")
    root.geometry("1100x700")
    EncryptUI(root)
    root.mainloop()