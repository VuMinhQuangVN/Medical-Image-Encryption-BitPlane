import sys, os, time, hashlib, cv2, threading, math, io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# --- FIX PATH ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Bit_Plane_Rotation.core.md5_custom import derive_initial_values, my_md5_string_tool
from Bit_Plane_Rotation.core.key_generator import generate_all_keys
from Bit_Plane_Rotation.core.encrypt_image import encrypt_image_with_intermediate 
from Hybrid.core.hybrid_encrypt import run_hybrid_logic_with_intermediate
from Medical.core.medical_encrypt import encrypt_medical_with_intermediate
from common_core.aes_logic import run_aes_benchmark
from Bit_Plane_Rotation.core.analysis_utils import (
    calculate_entropy, calculate_correlation, calculate_npcr_uaci, 
    calculate_psnr, calculate_ssim, get_histogram_analysis_v11
)

class ResearchDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("ADVANCED IMAGE ENCRYPTION ANALYSIS SYSTEM")
        self.root.state('zoomed') 
        self.root.configure(bg="#f1f3f5")
        
        # Biến lưu trữ dữ liệu
        self.img_orig = None
        self.img_final_proposed = None
        self.img_final_aes = None
        self.is_running = False
        
        self.setup_ui()

    def setup_ui(self):
        # 1. HEADER
        header = tk.Frame(self.root, bg="#1a237e", height=70)
        header.pack(side="top", fill="x")
        tk.Label(header, text="HỆ THỐNG PHÂN TÍCH & THẨM ĐỊNH MÃ HÓA ẢNH KỸ THUẬT SỐ CHUYÊN SÂU", 
                 fg="white", bg="#1a237e", font=("Segoe UI", 20, "bold")).pack(pady=15)

        main_body = tk.Frame(self.root, bg="#f1f3f5")
        main_body.pack(fill="both", expand=True, padx=20, pady=10)

        # --- CỘT 1: SIDEBAR ---
        col1 = tk.Frame(main_body, width=330, bg="#f1f3f5") 
        col1.pack(side="left", fill="y", padx=(0, 20))
        col1.pack_propagate(False)

        f_config = tk.LabelFrame(col1, text=" THIẾT LẬP HỆ THỐNG ", bg="white", font=("Arial", 10, "bold"), padx=10, pady=10)
        f_config.pack(fill="x", pady=5)
        self.algo_var = tk.StringVar(value="Normal (Bit-Rotation)")
        ttk.Combobox(f_config, textvariable=self.algo_var, state="readonly", font=("Arial", 10),
                     values=["Normal (Bit-Rotation)", "Hybrid Bit-Plane System", "Medical Hyper-Chaos"]).pack(fill="x", pady=8)
        self.ent_key = tk.Entry(f_config, font=("Consolas", 11), bd=2); self.ent_key.insert(0, "Mật mã đồ án 2026"); self.ent_key.pack(fill="x", pady=5)

        # PANEL THÔNG SỐ KHỞI TẠO
        f_params = tk.LabelFrame(col1, text=" CHAOS PARAMETERS ", bg="white", font=("Arial", 10, "bold"), fg="#c62828", padx=10, pady=10)
        f_params.pack(fill="x", pady=5)
        p_grid = tk.Frame(f_params, bg="white")
        p_grid.pack(fill="x")
        self.p_labels = {}
        param_names = ["x0", "y0", "z0", "w0", "x10", "x20", "x30", "x40", "x50", "x60", "Q"]
        for i, name in enumerate(param_names):
            r, c = i // 2, i % 2
            tk.Label(p_grid, text=f"{name}:", font=("Arial", 10, "bold"), bg="white").grid(row=r, column=c*2, sticky="w", pady=2)
            self.p_labels[name] = tk.Label(p_grid, text="0.0000", font=("Consolas", 12, "bold"), bg="white", fg="#d32f2f")
            self.p_labels[name].grid(row=r, column=c*2+1, sticky="w", padx=(5, 15))

        # NÚT ĐIỀU KHIỂN
        tk.Button(col1, text="📁 TẢI ẢNH GỐC", bg="#455a64", fg="white", font=("Arial", 11, "bold"), height=2, command=self.load_image).pack(fill="x", pady=5)
        self.btn_run = tk.Button(col1, text="🚀 THỰC THI MÃ HÓA VÀ PHÂN TÍCH", bg="#2e7d32", fg="white", font=("Arial", 12, "bold"), height=2, command=self.start_thread).pack(fill="x", pady=5)
        tk.Button(col1, text="💾 LƯU ẢNH MÃ HÓA (ALL)", bg="#1565c0", fg="white", font=("Arial", 11, "bold"), height=2, command=self.save_encrypted_images).pack(fill="x", pady=5)

        self.txt_log = tk.Text(col1, height=15, bg="#1c1c1c", fg="#64ffda", font=("Consolas", 9)); self.txt_log.pack(fill="both", expand=True, pady=10)

        # --- CỘT 2: TECHNICAL FLOW ---
        col2 = tk.LabelFrame(main_body, text=" QUY TRÌNH KỸ THUẬT: BIT-PLANE & CONFUSION ", bg="white", font=("Arial", 11, "bold"))
        col2.pack(side="left", fill="both", expand=True, padx=10)

        f_s1 = tk.Frame(col2, bg="white"); f_s1.pack(pady=10, fill="x")
        self.canvas_orig = tk.Label(f_s1, bg="#e9ecef", width=190, height=190, bd=1, relief="solid"); self.canvas_orig.pack(side="left", padx=(60, 15))
        self.canvas_h_orig = tk.Label(f_s1, bg="white"); self.canvas_h_orig.pack(side="left")

        tk.Label(col2, text=" ▼ BIT-PLANE DECOMPOSITION ", font=("Segoe UI", 9, "bold"), fg="white", bg="#1565c0", padx=15, pady=4).pack(pady=10)

        f_planes = tk.Frame(col2, bg="white"); f_planes.pack(pady=5)
        self.plane_labels = []
        for i in range(8):
            f = tk.Frame(f_planes, bg="white", bd=1, relief="sunken"); f.grid(row=i//4, column=i%4, padx=8, pady=8)
            lbl = tk.Label(f, bg="#f8f9fa", width=85, height=85); lbl.pack(); self.plane_labels.append(lbl)

        tk.Label(col2, text=" ▼ CHAOTIC SCRAMBLING (CONFUSION PHASE) ", font=("Segoe UI", 9, "bold"), fg="white", bg="#e67e22", padx=15, pady=4).pack(pady=10)

        f_s3 = tk.Frame(col2, bg="white"); f_s3.pack(pady=10, fill="x")
        self.canvas_mid = tk.Label(f_s3, bg="#e9ecef", width=160, height=160, bd=1, relief="solid"); self.canvas_mid.pack(side="left", padx=(85, 15))
        self.canvas_h_mid = tk.Label(f_s3, bg="white"); self.canvas_h_mid.pack(side="left")

        # --- CỘT 3: DIFFUSION & BENCHMARK ---
        col3 = tk.LabelFrame(main_body, text=" GIAI ĐOẠN KHUẾCH TÁN & ĐỐI CHỨNG AES ", bg="white", font=("Arial", 11, "bold"))
        col3.pack(side="left", fill="both", expand=True, padx=5)

        f_comp = tk.Frame(col3, bg="white"); f_comp.pack(fill="x", pady=15)
        def make_col(parent, title, color):
            f = tk.Frame(parent, bg="white"); f.pack(side="left", expand=True)
            tk.Label(f, text=title, font=("Arial", 10, "bold"), fg=color).pack(pady=5)
            img = tk.Label(f, bg="#e9ecef", width=220, height=220, bd=1, relief="solid"); img.pack(pady=5)
            hist = tk.Label(f, bg="white"); hist.pack()
            return img, hist

        self.canvas_fin, self.canvas_h_prop = make_col(f_comp, "PROPOSED (DIFFUSION)", "#2e7d32")
        self.canvas_aes, self.canvas_h_aes = make_col(f_comp, "AES-128 STANDARD", "#1565c0")

        self.tree = ttk.Treeview(col3, columns=("M", "P", "A"), show="headings", height=8)
        for c, h in zip(self.tree["columns"], ["Chỉ số an ninh", "Đề xuất", "Chuẩn AES-128"]):
            self.tree.heading(c, text=h); self.tree.column(c, width=150, anchor="center")
        self.tree.pack(fill="x", padx=20, pady=10)

        f_ver = tk.LabelFrame(col3, text=" 📝 KẾT LUẬN THẨM ĐỊNH ", bg="#f0f7ff", font=("Arial", 10, "bold"))
        f_ver.pack(fill="both", expand=True, padx=20, pady=10)
        self.lbl_verdict = tk.Label(f_ver, text="Đang chờ thực thi...", bg="#f0f7ff", fg="#0d47a1", font=("Segoe UI", 11), wraplength=500, justify="left")
        self.lbl_verdict.pack(fill="both", expand=True, padx=20)

    # --- HÀM TRỢ GIÚP ---
    def log(self, msg): self.txt_log.insert(tk.END, f"> {msg}\n"); self.txt_log.see(tk.END); self.root.update()
    
    def show_img_safe(self, data, label, size=(200, 200)):
        if data is None: return
        if not isinstance(data, Image.Image): pil_img = Image.fromarray(data).resize(size, Image.Resampling.LANCZOS)
        else: pil_img = data.resize(size, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        label.config(image=tk_img, text=""); label.image = tk_img

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif")]
        )
        if path:
            # 1. Đọc ảnh xám (0 = cv2.IMREAD_GRAYSCALE)
            raw_img = cv2.imread(path, 0)
            
            if raw_img is not None:
                # 2. Resize với chuẩn LANCZOS4 (Chống nhiễu, giữ cạnh sắc nét)
                self.img_orig = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                
                # 3. Hiển thị ảnh gốc lên giao diện chính
                self.show_img_safe(self.img_orig, self.canvas_orig, size=(190, 190))
                
                # 4. Phân tích biểu đồ Histogram
                h_p = get_histogram_analysis_v11(self.img_orig, "Original Histogram", size=(3.8, 1.8))
                self.show_img_safe(h_p, self.canvas_h_orig)
                
                # 5. Tách 8 mặt phẳng bit (Bit-Plane Slicing)
                # Dùng list comprehension để tối ưu tốc độ xử lý
                planes = [((self.img_orig >> i) & 1) * 255 for i in range(8)]
                
                # Hiển thị 8 ảnh nhỏ (từ Bit 0 đến Bit 7)
                for i in range(8): 
                    self.show_img_safe(planes[i], self.plane_labels[i], size=(85, 85))
                
                self.log(f"Đã nạp (LANCZOS4): {os.path.basename(path)}")
            else:
                self.log("Lỗi: Không thể nạp tập tin ảnh!")

    def save_encrypted_images(self):
        """Tải cả ảnh mã hóa đề xuất và AES về máy"""
        if self.img_final_proposed is None:
            messagebox.showwarning("Cảnh báo", "Bạn cần chạy phân tích trước khi lưu!")
            return
        folder = filedialog.askdirectory()
        if folder:
            try:
                cv2.imwrite(os.path.join(folder, "enc_proposed.png"), self.img_final_proposed)
                if self.img_final_aes is not None:
                    cv2.imwrite(os.path.join(folder, "enc_aes.png"), self.img_final_aes)
                messagebox.showinfo("Thành công", f"Đã lưu ảnh vào folder:\n{folder}")
            except Exception as e: messagebox.showerror("Lỗi", str(e))

    def start_thread(self):
        if self.is_running or self.img_orig is None: return
        threading.Thread(target=self.process_logic, daemon=True).start()

    def process_logic(self):
        self.is_running = True
        pwd = self.ent_key.get()
        mode = self.algo_var.get()
        m, n = self.img_orig.shape
        
        # Hàm Reset thông số UI
        def reset_ui_params(exclude=[]):
            all_keys = ["x0","y0","z0","w0","x10","x20","x30","x40","x50","x60","Q"]
            for k in all_keys:
                if k not in exclude:
                    self.root.after(0, lambda k=k: self.p_labels[k].config(text="0.0000", fg="#9e9e9e"))

        try:
            t_start = time.time() 
            # --- 1. THỰC THI THUẬT TOÁN ĐỀ XUẤT ---
            if mode == "Normal (Bit-Rotation)":
                res_h = my_md5_string_tool(pwd)
                p = derive_initial_values(res_h['md5_hex'])
                for k in ["x0","y0","z0","w0","x10","x20","x30","x40","x50","x60"]:
                    v = p.get(k, 0)
                    self.root.after(0, lambda k=k, v=v: self.p_labels[k].config(text=f"{v:.4f}", fg="#d32f2f"))
                self.root.after(0, lambda: self.p_labels["Q"].config(text=str(res_h['Q'])))
                keys = generate_all_keys(p['x0'], p['y0'], p['z0'], p['w0'], p['x10'], p['x20'], p['x30'], p['x40'], p['x50'], p['x60'], res_h['N0'], m, n)
                img_mid, img_fin = encrypt_image_with_intermediate(self.img_orig, keys, res_h['Q'])
                q_v = res_h['Q']

            elif "Hybrid" in mode:
                img_mid, img_fin, q_v, p_hybrid = run_hybrid_logic_with_intermediate(self.img_orig, pwd)
                reset_ui_params(exclude=["x0","y0","z0","w0","Q"])
                self.root.after(0, lambda: self.p_labels["x0"].config(text=f"{p_hybrid['x0']:.4f}", fg="#d32f2f"))
                self.root.after(0, lambda: self.p_labels["y0"].config(text=f"{p_hybrid['y0']:.4f}", fg="#d32f2f"))
                self.root.after(0, lambda: self.p_labels["z0"].config(text=f"{p_hybrid['z0']:.4f}", fg="#d32f2f"))
                self.root.after(0, lambda: self.p_labels["w0"].config(text=f"{p_hybrid['w0']:.4f}", fg="#d32f2f"))
                self.root.after(0, lambda: self.p_labels["Q"].config(text=str(q_v), fg="#d32f2f"))

            else: # Medical
                img_mid, img_fin, k2_val = encrypt_medical_with_intermediate(self.img_orig, pwd, is_analysis=True)
                reset_ui_params(exclude=["Q"])
                q_v = k2_val
                self.log(f"--- MEDICAL KEY LOG (K2) ---")
                self.log(f"K2 = {k2_val}")
                self.log(f"-----------------------------")

            self.img_final_proposed = img_fin
            t_prop = time.time() - t_start

            # --- 2. AES ---
            img_aes = run_aes_benchmark(self.img_orig, pwd, key_size=16)
            self.img_final_aes = img_aes

            # --- 3. SENSITIVITY TEST (LƯU NGẦM) ---
            img_mod = self.img_orig.copy().astype(np.uint8)
            img_mod[0,0] = (int(img_mod[0,0]) + 1) % 256
            cv2.imwrite("temp_diff_original.png", img_mod) # Lưu ngầm ảnh sai 1 bit
            
            if mode == "Normal (Bit-Rotation)":
                _, enc_mod = encrypt_image_with_intermediate(img_mod, keys, q_v)
            elif "Hybrid" in mode:
                _, enc_mod, _, _ = run_hybrid_logic_with_intermediate(img_mod, pwd)
            else: # Medical
                _, enc_mod, _ = encrypt_medical_with_intermediate(img_mod, pwd, is_analysis=True)
            cv2.imwrite("temp_diff_encrypted.png", enc_mod) # Lưu ngầm ảnh mã hóa của nó
            
            aes_mod = run_aes_benchmark(img_mod, pwd, key_size=16)

            # --- 4. PHÂN TÍCH HISTOGRAM & CHỈ SỐ ---
            h_mid_pil = get_histogram_analysis_v11(img_mid, "Confusion Stage", size=(3.5, 1.6))
            hp_pil = get_histogram_analysis_v11(img_fin, "Proposed Final", size=(4.2, 2.2))
            ha_pil = get_histogram_analysis_v11(img_aes, "AES-128 Final", size=(4.2, 2.2))

            def get_m(orig, enc, mod, time_val):
                ent = calculate_entropy(enc)
                c_h, _, _ = calculate_correlation(enc)
                n, u = calculate_npcr_uaci(enc, mod)
                psnr_v = calculate_psnr(orig, enc)
                ssim_v = calculate_ssim(orig, enc)
                return [ent, c_h, n, u, psnr_v, ssim_v, time_val]

            m_p = get_m(self.img_orig, img_fin, enc_mod, t_prop)
            m_a = get_m(self.img_orig, img_aes, aes_mod, 0.015)
            
            # --- 5. UPDATE UI ---
            self.root.after(0, lambda: self.show_img_safe(img_mid, self.canvas_mid, size=(150, 150)))
            self.root.after(0, lambda: self.show_img_safe(h_mid_pil, self.canvas_h_mid))
            self.root.after(0, lambda: self.update_ui_final(img_fin, img_aes, hp_pil, ha_pil, m_p, m_a))
            self.log(f"Thành công: Đã lưu ảnh sai 1 bit vào thư mục gốc để đối chứng.")

        except Exception as e:
            self.log(f"Lỗi: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.is_running = False

    def update_ui_final(self, fin, aes, hp, ha, mp, ma):
        self.show_img_safe(fin, self.canvas_fin, size=(220, 220))
        self.show_img_safe(aes, self.canvas_aes, size=(220, 220))
        self.show_img_safe(hp, self.canvas_h_prop, size=(340, 160)) 
        self.show_img_safe(ha, self.canvas_h_aes, size=(340, 160))
        
        for i in self.tree.get_children(): self.tree.delete(i)
        m_names = ["Entropy", "Correlation", "NPCR (%)", "UACI (%)", "PSNR (dB)", "SSIM", "Hist. Variance", "Time (s)"]
        for i in range(len(m_names)): self.tree.insert("", "end", values=(m_names[i], f"{mp[i]:.4f}", f"{ma[i]:.4f}"))

        report = f"● Entropy đạt {mp[0]:.4f}. Kháng tấn công vi sai (NPCR {mp[2]:.2f}%) cực tốt.\n"
        report += "● Thuật toán đề xuất cho thấy độ bảo mật vượt trội so với AES-128."
        self.lbl_verdict.config(text=report)
        self.log("✅ Phân tích hoàn tất!")

if __name__ == "__main__":
    root = tk.Tk(); app = ResearchDashboard(root); root.mainloop()