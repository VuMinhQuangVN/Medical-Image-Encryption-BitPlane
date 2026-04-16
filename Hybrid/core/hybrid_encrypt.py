# core/hybrid_encrypt.py
import numpy as np
from core.hybrid_utils import get_sha512_hex, get_sha512_ints, bit_plane_slice, bit_plane_rejoin, arnold_transform
from core.key_generator import get_all_keys_hybrid
from core.hybrid_diffusion import hybrid_diffusion_forward

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