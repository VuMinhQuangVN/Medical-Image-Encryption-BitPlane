# ui_analysis.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import numpy as np
import math
import os

# --- IMPORT CÁC HÀM CORE MÃ HÓA ---
from core.md5_custom import derive_initial_values, my_md5_string_tool
from core.key_generator import generate_all_keys
from core.encrypt_image import encrypt_image
from core.decrypt_image import decrypt_full

# --- IMPORT CÁC HÀM PHÂN TÍCH TỪ UTILS (ĐÃ TÁCH) ---
from core.analysis_utils import (
    calculate_entropy, 
    calculate_correlation, 
    calculate_npcr_uaci,
    calculate_psnr, 
    calculate_ssim, 
    calculate_ber
)

class AutoAnalysisUI:
    def __init__(self, parent):
        self.parent = parent
        self.img_orig = None
        self.filename = ""
        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.parent, bg="#f0f2f5")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # KHỐI ĐIỀU KHIỂN (Bên trái)
        ctrl_frame = tk.LabelFrame(main_frame, text=" Hệ thống thực thi ", bg="#f0f2f5", font=("Arial", 10, "bold"))
        ctrl_frame.pack(side="left", fill="y", padx=5, pady=5)

        tk.Label(ctrl_frame, text="Khóa (Seed):", bg="#f0f2f5").pack(pady=(10,0))
        self.ent_seed = tk.Entry(ctrl_frame, width=22)
        self.ent_seed.insert(0, "Mật mã đồ án 2026")
        self.ent_seed.pack(pady=5, padx=10)

        tk.Button(ctrl_frame, text="📁 NẠP ẢNH GỐC", command=self.load_image, bg="#34495e", fg="white", cursor="hand2").pack(fill="x", padx=10, pady=5)
        self.lbl_info = tk.Label(ctrl_frame, text="Chưa nạp ảnh", bg="#f0f2f5", font=("Arial", 8), fg="#7f8c8d")
        self.lbl_info.pack()

        tk.Button(ctrl_frame, text="📊 CHẠY PHÂN TÍCH", command=self.run_process, bg="#27ae60", fg="white", font=("Arial", 10, "bold"), height=2, cursor="hand2").pack(fill="x", padx=10, pady=20)
        
        tk.Button(ctrl_frame, text="🗑 XÓA BẢNG", command=lambda: self.tree.delete(*self.tree.get_children()), bg="#e74c3c", fg="white").pack(fill="x", padx=10, pady=5)

        # KHỐI KẾT QUẢ (Bên phải)
        res_frame = tk.LabelFrame(main_frame, text=" Kết quả phân tích định lượng (Quantitative Analysis) ", bg="white", font=("Arial", 10, "bold"))
        res_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        columns = ("name", "entropy", "corr_h", "corr_v", "corr_d", "npcr", "uaci", "e_psnr", "e_ssim", "d_psnr", "d_ssim", "d_ber")
        self.tree = ttk.Treeview(res_frame, columns=columns, show="headings")
        
        headers = ["Tên ảnh", "Entropy", "Corr(H)", "Corr(V)", "Corr(D)", "NPCR", "UACI", "PSNR(E)", "SSIM(E)", "PSNR(D)", "SSIM(D)", "BER(D)"]
        for col, head in zip(columns, headers):
            self.tree.heading(col, text=head)
            self.tree.column(col, width=80, anchor="center") 

        # Thêm thanh cuộn
        scrollbar = ttk.Scrollbar(res_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            # Thuật toán yêu cầu (256x256)
            self.img_orig = np.array(Image.open(path).convert("L").resize((256, 256)))
            self.filename = os.path.basename(path)
            self.lbl_info.config(text=f"Đã nạp: {self.filename}", fg="#27ae60")

    def run_process(self):
        if self.img_orig is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        
        try:
            # 1. Khởi tạo khóa từ Seed
            seed = self.ent_seed.get()
            res_md5 = my_md5_string_tool(seed)
            p = derive_initial_values(res_md5['md5_hex'])
            keys = generate_all_keys(p['x0'], p['y0'], p['z0'], p['w0'], p['x10'], p['x20'], p['x30'], p['x40'], p['x50'], p['x60'], res_md5['N0'], 256, 256)

            # 2. Thực hiện mã hóa (Bản mã c1)
            c1 = encrypt_image(self.img_orig, keys, res_md5['Q'])
            
            # 3. Thực hiện mã hóa ảnh biến đổi 1 bit (Bản mã c2) để tính NPCR/UACI
            img_mod = self.img_orig.copy()
            img_mod[0,0] = (int(img_mod[0,0]) + 1) % 256
            c2 = encrypt_image(img_mod.astype(np.uint8), keys, res_md5['Q'])
            
            # 4. Thực hiện giải mã (Bản khôi phục dec)
            dec = decrypt_full(c1, keys, res_md5['Q'])

            # 5. TÍNH TOÁN CÁC CHỈ SỐ (Gọi từ utils)
            entropy_val = calculate_entropy(c1)
            ch, cv, cd = calculate_correlation(c1)
            npcr_v, uaci_v = calculate_npcr_uaci(c1, c2)
            
            e_psnr = calculate_psnr(self.img_orig, c1)
            e_ssim = calculate_ssim(self.img_orig, c1)
            
            d_psnr = calculate_psnr(self.img_orig, dec)
            d_ssim = calculate_ssim(self.img_orig, dec)
            d_ber = calculate_ber(self.img_orig, dec)

            # 6. ĐẨY DỮ LIỆU VÀO BẢNG
            d_psnr_txt = "∞" if math.isinf(d_psnr) else f"{d_psnr:.2f}"
            
            # Thêm vào dòng đầu tiên của bảng
            self.tree.insert("", 0, values=(
                self.filename, 
                f"{entropy_val:.4f}", 
                f"{ch:.4f}", f"{cv:.4f}", f"{cd:.4f}",
                f"{npcr_v:.2f}%", 
                f"{uaci_v:.2f}%", 
                f"{e_psnr:.2f}",
                f"{e_ssim:.4f}", 
                d_psnr_txt, 
                f"{d_ssim:.4f}", 
                "0" if d_ber == 0 else f"{d_ber:.2e}"
            ))

        except Exception as e:
            messagebox.showerror("Lỗi", f"Quá trình phân tích thất bại: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hệ Thống Phân Tích Tự Động Ảnh Thường - Đồ Án 2026")
    root.geometry("1100x500")
    AutoAnalysisUI(root)
    root.mainloop()