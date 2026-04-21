# core/analysis_utils.py
import matplotlib.pyplot as plt
import numpy as np
import io
import math
from PIL import Image, ImageTk

# --- NHÓM 1: CÁC HÀM TÍNH TOÁN ĐỊNH LƯỢNG ---

def calculate_entropy(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,255))
    p = hist / np.sum(hist)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def calculate_correlation(img):
    """Tính bộ 3 hệ số tương quan: (Ngang, Dọc, Chéo)"""
    img = img.astype(np.float64)
    
    # 1. Phương Ngang (H)
    h_x, h_y = img[:, :-1].flatten(), img[:, 1:].flatten()
    corr_h = np.corrcoef(h_x, h_y)[0, 1]
    
    # 2. Phương Dọc (V)
    v_x, v_y = img[:-1, :].flatten(), img[1:, :].flatten()
    corr_v = np.corrcoef(v_x, v_y)[0, 1]
    
    # 3. Phương Chéo (D)
    d_x, d_y = img[:-1, :-1].flatten(), img[1:, 1:].flatten()
    corr_d = np.corrcoef(d_x, d_y)[0, 1]
    
    res = [c if not np.isnan(c) else 0.0 for c in [corr_h, corr_v, corr_d]]
    return tuple(res)

def calculate_npcr_uaci(c1, c2):
    diff = (c1 != c2).astype(np.float64)
    npcr = np.sum(diff) / diff.size * 100
    abs_diff = np.abs(c1.astype(np.float64) - c2.astype(np.float64))
    uaci = np.mean(abs_diff / 255.0) * 100
    return npcr, uaci

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0: return float('inf')
    return 10 * math.log10((255.0 ** 2) / mse)

def calculate_ssim(img1, img2):
    P, E = img1.astype(np.float64), img2.astype(np.float64)
    mu_P, mu_E = np.mean(P), np.mean(E)
    var_P, var_E = np.var(P), np.var(E)
    covar_PE = np.mean((P - mu_P) * (E - mu_E))
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    return ((2*mu_P*mu_E + C1)*(2*covar_PE + C2)) / ((mu_P**2 + mu_E**2 + C1)*(var_P + var_E + C2))

def calculate_ber(img1, img2):
    b1 = np.unpackbits(img1.astype(np.uint8))
    b2 = np.unpackbits(img2.astype(np.uint8))
    return np.sum(b1 != b2) / b1.size

# --- NHÓM 2: CÁC HÀM HỖ TRỢ TRỰC QUAN (HISTOGRAM) ---

def get_histogram_image(img_np, title="Histogram", size=(4, 3)):
    """
    Tạo biểu đồ Histogram từ mảng NumPy và trả về đối tượng PhotoImage
    """
    plt.figure(figsize=size, dpi=80)
    plt.hist(img_np.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.8)
    plt.title(title, fontsize=10)
    plt.xlabel("Giá trị Pixel", fontsize=8)
    plt.ylabel("Tần suất", fontsize=8)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    pil_img = Image.open(buf)
    return ImageTk.PhotoImage(pil_img)