import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import os
import math

# Import các thuật toán từ thư mục core
from core.analysis_utils import get_histogram_image, calculate_psnr, calculate_ssim, calculate_ber
from core.md5_custom import derive_initial_values, my_md5_string_tool
from core.key_generator import generate_all_keys
from core.decrypt_image import decrypt_full

class DecryptUI:
    def __init__(self, parent):
        self.parent = parent
        self.res_md5 = None
        self.file_path = None
        self.cipher_np = None     
        self.original_np = None   # Ảnh gốc để đối chứng
        self.decrypted_np = None  
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
        tk.Label(left_panel, text="1. Chuỗi khóa bí mật (Seed):", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
        self.ent_seed = tk.Entry(left_panel, font=("Arial", 11), bd=2)
        self.ent_seed.insert(0, "Mật mã đồ án 2026")
        self.ent_seed.pack(fill="x", pady=5)

        # 2. Chọn ảnh mã hóa
        tk.Label(left_panel, text="2. File ảnh đầu vào:", bg="#f4f7f6", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15,0))
        tk.Button(left_panel, text="📁 TẢI ẢNH MÃ HÓA (CIPHER)", command=self.load_cipher, bg="#34495e", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=2)
        
        # 3. Chọn ảnh gốc (Đối chứng)
        tk.Button(left_panel, text="🖼️ NẠP ẢNH GỐC (ĐỐI CHỨNG)", command=self.load_original, bg="#7f8c8d", fg="white", font=("Arial", 9, "bold")).pack(fill="x", pady=2)
        self.lbl_orig_info = tk.Label(left_panel, text="Chưa nạp ảnh gốc", bg="#f4f7f6", font=("Arial", 8, "italic"), fg="#e67e22")
        self.lbl_orig_info.pack()

        # 4. BẢNG CHỈ SỐ KHÔI PHỤC (Recovery Metrics)
        recovery_frame = tk.LabelFrame(left_panel, text=" Độ chính xác khôi phục ", bg="white", font=("Arial", 9, "bold"))
        recovery_frame.pack(fill="x", pady=15, padx=2)

        self.metrics = {
            "PSNR(D):": tk.Label(recovery_frame, text="-", bg="white", font=("Arial", 10, "bold"), fg="#27ae60"),
            "SSIM(D):": tk.Label(recovery_frame, text="-", bg="white", font=("Arial", 10, "bold"), fg="#27ae60"),
            "BER(D):":  tk.Label(recovery_frame, text="-", bg="white", font=("Arial", 10, "bold"), fg="#27ae60")
        }

        for i, (name, label) in enumerate(self.metrics.items()):
            tk.Label(recovery_frame, text=name, bg="white").grid(row=i, column=0, sticky="w", padx=5, pady=3)
            label.grid(row=i, column=1, sticky="e", padx=15, pady=3)
        recovery_frame.columnconfigure(1, weight=1)

        # 5. Thao tác
        self.btn_decrypt = tk.Button(left_panel, text="🔓 GIẢI MÃ & ĐỐI CHIẾU", command=self.process_decrypt, bg="#8e44ad", fg="white", font=("Arial", 10, "bold"), height=2, state="disabled")
        self.btn_decrypt.pack(fill="x", pady=5)

        self.btn_save = tk.Button(left_panel, text="📥 LƯU ẢNH GIẢI MÃ", command=self.save_image, bg="#d35400", fg="white", state="disabled")
        self.btn_save.pack(fill="x", pady=5)

        # Log
        self.txt_log = tk.Text(left_panel, height=8, font=("Consolas", 9), bg="#1e1e1e", fg="#00ff00", padx=10, pady=5)
        self.txt_log.pack(pady=5, fill="both", expand=True)

        # --- CỘT PHẢI: HIỂN THỊ ---
        right_panel = tk.Frame(main_frame, bg="#f4f7f6")
        right_panel.pack(side="right", fill="both", expand=True, padx=(20, 0))

        self.frame_cipher = tk.LabelFrame(right_panel, text=" ẢNH MÃ HÓA (INPUT) ", bg="white", font=("Arial", 10, "bold"))
        self.frame_cipher.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.canvas_cipher = tk.Label(self.frame_cipher, bg="white")
        self.canvas_cipher.pack(fill="both", expand=True)
        self.canvas_hist_cipher = tk.Label(self.frame_cipher, bg="white")
        self.canvas_hist_cipher.pack(fill="both", expand=True, pady=5)

        self.frame_plain = tk.LabelFrame(right_panel, text=" ẢNH KHÔI PHỤC (OUTPUT) ", bg="white", font=("Arial", 10, "bold"))
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
        # 1. Mở hộp thoại chọn file ảnh bản mã
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.bmp *.jpg")])
        if path:
            # 2. Đọc ảnh bằng OpenCV ở chế độ Ảnh Xám
            img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                messagebox.showerror("Lỗi", "Không thể đọc được file bản mã này!")
                return

            # 3. Chuẩn hóa về 256x256 bằng thuật toán nội suy LANCZOS4
            self.cipher_np = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # 4. CHUẨN BỊ HIỂN THỊ (NumPy -> PIL -> ImageTk)
            # Tạo ảnh hiển thị lớn hơn (300x300) cho Canvas
            img_pil = Image.fromarray(self.cipher_np)
            self.tk_cipher = ImageTk.PhotoImage(img_pil.resize((300, 300), Image.Resampling.LANCZOS))
            self.canvas_cipher.config(image=self.tk_cipher)
            
            # 5. Vẽ Histogram từ mảng NumPy của bản mã
            self.tk_h_cipher = get_histogram_image(self.cipher_np, "Histogram Bản Mã")
            self.canvas_hist_cipher.config(image=self.tk_h_cipher)
            
            # 6. Cập nhật trạng thái
            self.btn_decrypt.config(state="normal")
            self.log(f"Đã nạp bản mã (CV2): {os.path.basename(path)}")
            
    def load_original(self):
        # 1. Mở hộp thoại chọn file ảnh gốc đối chứng
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.bmp *.jpg")])
        if path:
            # 2. Đọc ảnh bằng OpenCV ở chế độ Ảnh Xám
            img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                messagebox.showerror("Lỗi", "Không thể đọc được file ảnh gốc này!")
                return

            # 3. Chuẩn hóa về 256x256 bằng thuật toán nội suy LANCZOS4
            # Giúp mảng NumPy của ảnh gốc khớp 100% với cách xử lý của bản mã
            self.original_np = cv2.resize(img_cv, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # 4. Cập nhật giao diện
            self.lbl_orig_info.config(text=f"Đã khớp: {os.path.basename(path)}", fg="#27ae60")
            self.log(f"Đã nạp ảnh gốc đối chứng (CV2): {os.path.basename(path)}")

    def process_decrypt(self):
        seed = self.ent_seed.get()
        if not seed: return
        
        try:
            self.log("Đang giải mã...")
            start_t = time.time()
            res_md5 = my_md5_string_tool(seed)
            p = derive_initial_values(res_md5['md5_hex'])
            m, n = self.cipher_np.shape
            keys = generate_all_keys(p['x0'], p['y0'], p['z0'], p['w0'], p['x10'], p['x20'], p['x30'], p['x40'], p['x50'], p['x60'], res_md5['N0'], m, n)

            # Giải mã
            self.decrypted_np = decrypt_full(self.cipher_np, keys, res_md5['Q'])
            
            # 1. Tính toán chỉ số nếu có ảnh gốc
            if self.original_np is not None:
                psnr_v = calculate_psnr(self.original_np, self.decrypted_np)
                ssim_v = calculate_ssim(self.original_np, self.decrypted_np)
                ber_v = calculate_ber(self.original_np, self.decrypted_np)
                
                psnr_txt = "∞" if math.isinf(psnr_v) else f"{psnr_v:.2f}"
                ber_txt = "0" if ber_v == 0 else f"{ber_v:.2e}"
                
                self.metrics["PSNR(D):"].config(text=psnr_txt)
                self.metrics["SSIM(D):"].config(text=f"{ssim_v:.4f}")
                self.metrics["BER(D):"].config(text=ber_txt)
                self.log("Đã hoàn tất đối chiếu với ảnh gốc.")
            else:
                self.log("Cảnh báo: Không có ảnh gốc để tính chỉ số khôi phục.")

            # 2. Hiển thị ảnh và Histogram
            res_img = Image.fromarray(self.decrypted_np)
            self.res_img_tk = ImageTk.PhotoImage(res_img.resize((300, 300)))
            self.canvas_plain.config(image=self.res_img_tk)
            
            self.hist_res_tk = get_histogram_image(self.decrypted_np, "Histogram Giải Mã")
            self.canvas_hist_plain.config(image=self.hist_res_tk)
            self.log(f"Hoàn thành trong {time.time() - start_t:.2f}s")
            self.btn_save.config(state="normal")
            self.log("Giải mã thành công!")

        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def save_image(self):
        if self.decrypted_np is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if save_path:
                Image.fromarray(self.decrypted_np).save(save_path)
                messagebox.showinfo("Thành công", "Đã lưu ảnh giải mã!")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hệ Thống Giải Mã Ảnh Thường")
    root.geometry("1100x700")
    DecryptUI(root)
    root.mainloop()