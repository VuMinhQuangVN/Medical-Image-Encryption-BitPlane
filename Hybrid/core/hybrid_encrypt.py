# core/hybrid_encrypt.py
import numpy as np
from Hybrid.core.hybrid_utils import get_sha512_hex, get_sha512_ints, bit_plane_slice, bit_plane_rejoin, arnold_transform
from Hybrid.core.key_generator import get_all_keys_hybrid
from Hybrid.core.hybrid_diffusion import hybrid_diffusion_forward

def run_hybrid_logic(img_np, pwd):
    """Hàm lõi thực hiện thuật toán Lai ghép (Đã tách ra khỏi UI)"""
    m, n = img_np.shape
    
    # 1. SHA-512 (Custom) -> Hex
    sha_hex = get_sha512_hex(pwd)
    sha_ints = get_sha512_ints(sha_hex)
    
    # 2. Sinh khóa Hyper-Lorenz RK4
    keys = get_all_keys_hybrid(sha_hex, m, n)
    
    # 3. GIAI ĐOẠN 1: Scrambling (Xáo trộn Bit-plane Arnold)
    planes = bit_plane_slice(img_np)
    scrambled_planes = []
    for p in range(1, 9):
        start, end = (p-1)*8, p*8
        sum_h = sum(sha_ints[start:end])
        # Tham số a, b, iterations dựa trên SHA-512
        a, b, it = 1 + (sum_h % 10), 1 + (sum_h % 10), 5 + (sum_h % 5)
        scrambled_planes.append(arnold_transform(planes[p-1], a, b, it))
    
    scrambled_img = bit_plane_rejoin(scrambled_planes)
    
    # 4. GIAI ĐOẠN 2: Diffusion (Khuếch tán Hyper-chaos)
    cipher = hybrid_diffusion_forward(scrambled_img, keys['K2'], keys['K3'], keys['Q'])
    
    return cipher, keys['Q']

def run_hybrid_logic_with_intermediate(img_np, pwd):
    m, n = img_np.shape
    sha_hex = get_sha512_hex(pwd)
    sha_ints = get_sha512_ints(sha_hex) # list 64 bytes

    # --- TÍNH TOÁN CÁC GIÁ TRỊ THỰC ĐỂ HIỂN THỊ LÊN UI ---
    x0_val = (sum(sha_ints[0:8]) / 2040) + 0.1234
    y0_val = (sum(sha_ints[8:16]) / 2040) + 0.5678
    z0_val = (sum(sha_ints[16:24]) / 2040) + 0.9101
    w0_val = (sum(sha_ints[24:32]) / 2040) + 0.1121
    
    # Sinh khóa (bên trong hàm này cũng tính lại các giá trị trên)
    keys = get_all_keys_hybrid(sha_hex, m, n)
    
    # Scrambling
    planes = bit_plane_slice(img_np)
    scrambled_planes = []
    for p in range(1, 9):
        start, end = (p-1)*8, p*8
        sum_h = sum(sha_ints[start:end])
        a, b, it = 1 + (sum_h % 10), 1 + (sum_h % 10), 5 + (sum_h % 5)
        scrambled_planes.append(arnold_transform(planes[p-1], a, b, it))
    
    scrambled_img = bit_plane_rejoin(scrambled_planes)
    
    # Diffusion
    cipher = hybrid_diffusion_forward(scrambled_img, keys['K2'], keys['K3'], keys['Q'])
    
    # Đóng gói các tham số để UI hiển thị
    params = {
        'x0': x0_val, 'y0': y0_val, 'z0': z0_val, 'w0': w0_val,
        'sha_hex': sha_hex
    }
    
    return scrambled_img, cipher, keys['Q'], params