import sys, os, time, cv2, threading, math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

# --- FIX PATH ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- IMPORT CORE ---
from Bit_Plane_Rotation.core.md5_custom import derive_initial_values, my_md5_string_tool
from Bit_Plane_Rotation.core.key_generator import generate_all_keys
from Bit_Plane_Rotation.core.diffusion_inverse import diffusion_inverse
from Bit_Plane_Rotation.core.decrypt_image import bit_plane_rotation_inverse 
from Hybrid.core.hybrid_utils import get_sha512_hex, get_sha512_ints, bit_plane_slice, bit_plane_rejoin, inverse_arnold, compute_N0_Q_hybrid
from Hybrid.core.key_generator import get_all_keys_hybrid
from Hybrid.core.hybrid_diffusion import hybrid_diffusion_backward
from Hybrid.core.hybrid_decrypt import run_hybrid_decrypt_logic
from Medical.core.medical_utils import get_logistic_map
from Medical.core.medical_decrypt import decrypt_medical 
from Bit_Plane_Rotation.core.analysis_utils import calculate_psnr, calculate_ssim, calculate_ber, get_histogram_analysis_v11

class DecryptionDashboardV2:
    def __init__(self, root):
        self.root = root
        self.root.title("ADVANCED IMAGE DECRYPTION ANALYSIS SYSTEM Pro V10")
        self.root.state('zoomed'); self.root.configure(bg="#f1f3f5")
        self.img_cipher, self.img_ref, self.img_dec, self.img_mid = None, None, None, None
        self.is_running = False
        self.setup_ui()

    def setup_ui(self):
        # 1. HEADER
        header = tk.Frame(self.root, bg="#1a237e", height=70); header.pack(side="top", fill="x")
        tk.Label(header, text="HỆ THỐNG NGHIÊN CỨU PHÂN TÍCH GIẢI MÃ ẢNH ĐA TẦNG (BIT-PLANE & CHAOS)", 
                 fg="white", bg="#1a237e", font=("Segoe UI", 20, "bold")).pack(pady=15)

        main_body = tk.Frame(self.root, bg="#f1f3f5"); main_body.pack(fill="both", expand=True, padx=20, pady=10)

        # --- CỘT 1: SIDEBAR ---
        col1 = tk.Frame(main_body, width=320, bg="#f1f3f5"); col1.pack(side="left", fill="y", padx=(0, 15)); col1.pack_propagate(False)
        
        f_config = tk.LabelFrame(col1, text=" THIẾT LẬP HỆ THỐNG ", bg="white", font=("Arial", 9, "bold"), padx=10, pady=10); f_config.pack(fill="x", pady=5)
        self.algo_var = tk.StringVar(value="Normal (Bit-Rotation)")
        self.combo = ttk.Combobox(f_config, textvariable=self.algo_var, state="readonly", values=["Normal (Bit-Rotation)", "Hybrid Bit-Plane System", "Medical Hyper-Chaos"])
        self.combo.pack(fill="x", pady=5); self.combo.bind("<<ComboboxSelected>>", self.on_algo_change)
        self.ent_key = tk.Entry(f_config, font=("Consolas", 11)); self.ent_key.insert(0, "Mật mã đồ án 2026"); self.ent_key.pack(fill="x", pady=5)

        f_params = tk.LabelFrame(col1, text=" HỆ SINH KHÓA HỒN ĐỘN ĐỘNG ", bg="white", font=("Arial", 9, "bold"), fg="#c62828", padx=10, pady=10); f_params.pack(fill="x", pady=5)
        p_grid = tk.Frame(f_params, bg="white"); p_grid.pack(fill="x"); self.p_labels = {}
        for i, n in enumerate(["x0", "y0", "z0", "w0", "x10", "x20", "x30", "x40", "x50", "x60", "Q"]):
            r, c = i // 2, i % 2
            tk.Label(p_grid, text=f"{n}:", font=("Arial", 9, "bold"), bg="white").grid(row=r, column=c*2, sticky="w", pady=2)
            self.p_labels[n] = tk.Label(p_grid, text="0.0000", font=("Consolas", 11, "bold"), bg="white", fg="#d32f2f")
            self.p_labels[n].grid(row=r, column=c*2+1, sticky="w", padx=(5, 10))

        self.f_k2 = tk.Frame(f_params, bg="#e3f2fd", pady=8); tk.Label(self.f_k2, text="Tham số K2:", bg="#e3f2fd", font=("Arial", 8, "bold")).pack()
        self.ent_k2 = tk.Entry(self.f_k2, font=("Consolas", 10), fg="blue"); self.ent_k2.pack(fill="x", padx=5)

        tk.Button(col1, text="📁 TẢI BẢN MÃ (CIPHER)", bg="#455a64", fg="white", font=("Arial", 10, "bold"), height=2, command=self.load_cipher).pack(fill="x", pady=5)
        tk.Button(col1, text="🖼️ NẠP ẢNH GỐC ĐỐI CHỨNG", bg="#78909c", fg="white", font=("Arial", 9), command=self.load_ref).pack(fill="x", pady=2)
        tk.Button(col1, text="🔓 THỰC THI GIẢI MÃ", bg="#2e7d32", fg="white", font=("Arial", 12, "bold"), height=2, command=self.start_thread).pack(fill="x", pady=10)
        self.txt_log = tk.Text(col1, height=15, bg="#1c1c1c", fg="#64ffda", font=("Consolas", 8)); self.txt_log.pack(fill="both", expand=True)

        # --- CỘT 2: QUY TRÌNH (FIXED LOGIC ORDER) ---
        col2 = tk.LabelFrame(main_body, text=" PHÂN TÍCH QUY TRÌNH KỸ THUẬT NGHỊCH ĐẢO ", bg="white", font=("Arial", 11, "bold"))
        col2.pack(side="left", fill="both", expand=True, padx=10)

        # STAGE 1: CIPHER (TRÊN CÙNG)
        f_top = tk.Frame(col2, bg="white"); f_top.pack(pady=10)
        self.canvas_cipher = tk.Label(f_top, bg="#e9ecef", width=160, height=160, bd=1, relief="solid"); self.canvas_cipher.pack(side="left", padx=10)
        self.canvas_h_cipher = tk.Label(f_top, bg="white"); self.canvas_h_cipher.pack(side="left")

        # LABELS STEP 1 & 2
        f_steps_1 = tk.Frame(col2, bg="#fffde7", bd=1, relief="ridge"); f_steps_1.pack(fill="x", padx=40, pady=5)
        self.step1 = tk.Label(f_steps_1, text="STEP 1: Recuperating Chaotic Keys", bg="#eeeeee", font=("Arial", 9, "bold"), pady=4); self.step1.pack(fill="x", pady=1)
        self.step2 = tk.Label(f_steps_1, text="STEP 2: Inverse Chaotic Diffusion (Get img_mid)", bg="#eeeeee", font=("Arial", 9, "bold"), pady=4); self.step2.pack(fill="x", pady=1)
        tk.Label(col2, text="▼", bg="white").pack()

        # STAGE 2: MIDDLE IMAGE (SAU GIẢI KHUẾCH TÁN)
        self.canvas_mid = tk.Label(col2, bg="#e9ecef", width=160, height=160, bd=1, relief="solid"); self.canvas_mid.pack(pady=5)
        
        # LABELS STEP 3
        f_steps_2 = tk.Frame(col2, bg="#fffde7", bd=1, relief="ridge"); f_steps_2.pack(fill="x", padx=40, pady=5)
        self.step3 = tk.Label(f_steps_2, text="STEP 3: Inverse Arnold / Bit-Rotation (Processing Planes)", bg="#eeeeee", font=("Arial", 9, "bold"), pady=4); self.step3.pack(fill="x", pady=1)
        tk.Label(col2, text="▼", bg="white").pack()

        # STAGE 3: 8 BIT-PLANES (CỦA ẢNH GIẢI XÁO TRỘN)
        f_planes = tk.Frame(col2, bg="white"); f_planes.pack(pady=5)
        self.plane_labels = []
        for i in range(8):
            f = tk.Frame(f_planes, bg="white", bd=1, relief="sunken"); f.grid(row=i//4, column=i%4, padx=5, pady=5)
            lbl = tk.Label(f, bg="#f8f9fa", width=75, height=75); lbl.pack(); self.plane_labels.append(lbl)

        # STEP 4: FINAL REJOIN
        tk.Label(col2, text="▼", bg="white").pack()
        self.step4 = tk.Label(f_steps_2, text="STEP 4: Final Bit-Plane Rejoining & Reconstruction", bg="#eeeeee", font=("Arial", 9, "bold"), pady=4); self.step4.pack(fill="x", pady=1)

        # --- CỘT 3: KẾT QUẢ ---
        col3 = tk.LabelFrame(main_body, text=" KẾT QUẢ KHÔI PHỤC & ĐỐI CHỨNG ", bg="white", font=("Arial", 11, "bold"))
        col3.pack(side="left", fill="both", expand=True, padx=5)
        self.canvas_dec = tk.Label(col3, bg="#f1f3f5", width=420, height=420, bd=1, relief="solid"); self.canvas_dec.pack(pady=20)
        
        self.tree = ttk.Treeview(col3, columns=("M", "V"), show="headings", height=8)
        self.tree.heading("M", text="Chỉ số khôi phục"); self.tree.heading("V", text="Kết quả")
        self.tree.column("M", width=250); self.tree.column("V", width=120, anchor="center"); self.tree.pack(fill="x", padx=20, pady=10)

    # --- LOGIC ---
    def log(self, msg): self.txt_log.insert(tk.END, f"> {msg}\n"); self.txt_log.see(tk.END); self.root.update()

    def on_algo_change(self, event=None):
        if "Medical" in self.algo_var.get(): self.f_k2.pack(fill="x", padx=10, pady=5)
        else: self.f_k2.pack_forget()

    def show_img(self, data, label, size=(200, 200)):
        if data is None: return
        pil = data if isinstance(data, Image.Image) else Image.fromarray(data)
        tk_img = ImageTk.PhotoImage(pil.resize(size, Image.LANCZOS)); label.config(image=tk_img); label.image = tk_img

    def load_cipher(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_cipher = cv2.resize(cv2.imread(path, 0), (256, 256), interpolation=cv2.INTER_LANCZOS4)
            self.show_img(self.img_cipher, self.canvas_cipher, (160, 160))
            h_p = get_histogram_analysis_v11(self.img_cipher, "Cipher Hist", size=(3.5, 1.5))
            self.show_img(h_p, self.canvas_h_cipher, (320, 140)); self.log(f"Đã nạp bản mã: {os.path.basename(path)}")

    def load_ref(self):
        path = filedialog.askopenfilename()
        if path: self.img_ref = cv2.resize(cv2.imread(path, 0), (256, 256), interpolation=cv2.INTER_LANCZOS4); self.log("Đã nạp ảnh đối chứng.")

    def start_thread(self):
        if self.is_running or self.img_cipher is None: return
        threading.Thread(target=self.process_logic, daemon=True).start()

    def process_logic(self):
        self.is_running = True; pwd, mode = self.ent_key.get(), self.algo_var.get()
        try:
            self.root.after(0, lambda: self.step1.config(bg="#fff9c4"))
            # LOGIC GIẢI MÃ
            if mode == "Normal (Bit-Rotation)":
                res_h = my_md5_string_tool(pwd); p = derive_initial_values(res_h['md5_hex'])
                keys = generate_all_keys(p['x0'], p['y0'], p['z0'], p['w0'], p['x10'], p['x20'], p['x30'], p['x40'], p['x50'], p['x60'], res_h['N0'], 256, 256)
                self.root.after(0, lambda: self.step2.config(bg="#fff9c4"))
                self.img_mid = diffusion_inverse(self.img_cipher, keys["K2"], keys["K3"], res_h['Q'])
                self.root.after(0, lambda: self.show_img(self.img_mid, self.canvas_mid, (160, 160)))
                self.root.after(0, lambda: self.step3.config(bg="#fff9c4"))
                self.img_dec = bit_plane_rotation_inverse(self.img_mid, keys)
                for k in p: self.root.after(0, lambda k=k, v=p[k]: self.p_labels[k].config(text=f"{v:.4f}"))
                self.root.after(0, lambda: self.p_labels["Q"].config(text=str(res_h['Q'])))

            elif "Hybrid" in mode:
                sha_hex = get_sha512_hex(pwd); _, _, q_v = compute_N0_Q_hybrid(sha_hex); keys_h = get_all_keys_hybrid(sha_hex, 256, 256)
                self.root.after(0, lambda: self.step2.config(bg="#fff9c4"))
                self.img_mid = hybrid_diffusion_backward(self.img_cipher, keys_h['K2'], keys_h['K3'], q_v)
                self.root.after(0, lambda: self.show_img(self.img_mid, self.canvas_mid, (160, 160)))
                self.root.after(0, lambda: self.step3.config(bg="#fff9c4"))
                self.img_dec = run_hybrid_decrypt_logic(self.img_cipher, pwd)
                self.root.after(0, lambda: self.p_labels["Q"].config(text=str(q_v)))

            else: # Medical
                k2 = float(self.ent_k2.get()); logistic_map = get_logistic_map(k2, 256*256).reshape(256, 256)
                self.root.after(0, lambda: self.step2.config(bg="#fff9c4"))
                self.img_mid = cv2.bitwise_xor(self.img_cipher, logistic_map.astype(np.uint8))
                self.root.after(0, lambda: self.show_img(self.img_mid, self.canvas_mid, (160, 160)))
                self.root.after(0, lambda: self.step3.config(bg="#fff9c4"))
                self.img_dec = decrypt_medical(self.img_cipher, pwd, k2, original_ref=self.img_ref)

            # HIỂN THỊ PLANE CỦA ẢNH SAU GIẢI XÁO TRỘN (Đã khôi phục hoàn chỉnh)
            final_planes = [((self.img_dec >> i) & 1) * 255 for i in range(8)]
            for i in range(8): self.root.after(0, lambda i=i: self.show_img(final_planes[i], self.plane_labels[i], (75, 75)))
            self.root.after(0, lambda: self.step4.config(bg="#c8e6c9"))
            self.root.after(0, self.update_final)
        except Exception as e: self.log(f"Lỗi: {e}")
        finally: self.is_running = False

    def update_final(self):
        # Hiển thị ảnh giải mã cuối cùng
        self.show_img(self.img_dec, self.canvas_dec, (420, 420))
        
        # Xóa các dòng cũ trong bảng chỉ số
        for i in self.tree.get_children(): 
            self.tree.delete(i)
            
        if self.img_ref is not None:
            # Tính toán các chỉ số khôi phục
            psnr = calculate_psnr(self.img_ref, self.img_dec)
            ssim = calculate_ssim(self.img_ref, self.img_dec)
            ber = calculate_ber(self.img_ref, self.img_dec)
            
            # --- XỬ LÝ HIỂN THỊ ĐẸP ---
            # 1. Xử lý PSNR vô cùng (inf)
            if math.isinf(psnr) or psnr > 99:
                psnr_txt = "∞ dB"
            else:
                psnr_txt = f"{psnr:.2f} dB"
                
            # 2. Xử lý BER bằng 0
            if ber == 0:
                ber_txt = "0"
            else:
                ber_txt = f"{ber:.1e}" # Hiện dạng khoa học nếu có lỗi bit (vd: 1.2e-05)

            # 3. Định dạng SSIM
            ssim_txt = f"{ssim:.4f}"

            # Cập nhật vào bảng Treeview
            results = [
                ("PSNR (Độ tương quan)", psnr_txt),
                ("SSIM (Độ tương đồng)", ssim_txt),
                ("BER (Tỷ lệ lỗi bit)", ber_txt)
            ]
            
            for m, v in results:
                self.tree.insert("", "end", values=(m, v))
                
            # Ghi log kết luận dựa trên SSIM
            if ssim > 0.9999:
                self.log("✅ Khôi phục HOÀN HẢO (Lossless Recovery).")
            else:
                self.log("⚠️ Khôi phục có sai số nhỏ.")
        else:
            self.log("⚠️ Không có ảnh đối chứng để tính chỉ số.")

        self.log("✅ Quy trình giải mã và đối chứng hoàn tất!")

if __name__ == "__main__":
    root = tk.Tk(); app = DecryptionDashboardV2(root); root.mainloop()