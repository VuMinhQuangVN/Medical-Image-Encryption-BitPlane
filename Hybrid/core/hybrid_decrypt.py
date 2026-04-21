# core/hybrid_decrypt.py
from Hybrid.core.hybrid_utils import bit_plane_rejoin, bit_plane_slice, compute_N0_Q_hybrid, get_sha512_hex, get_sha512_ints, inverse_arnold
from Hybrid.core.key_generator import get_all_keys_hybrid
from Hybrid.core.hybrid_diffusion import hybrid_diffusion_backward

def run_hybrid_decrypt_logic(cipher_np, pwd):
    """Hàm logic Giải mã (Vừa tách từ UI ra)"""
    m, n = cipher_np.shape
    sha_hex = get_sha512_hex(pwd)
    sha_ints = get_sha512_ints(sha_hex)
    
    # Khôi phục Q và Khóa
    _, _, q_val = compute_N0_Q_hybrid(sha_hex)
    keys = get_all_keys_hybrid(sha_hex, m, n)
    
    # 1. Giải mã khuếch tán (Diffusion Backward)
    diff_inv = hybrid_diffusion_backward(cipher_np, keys['K2'], keys['K3'], q_val)
    
    # 2. Giải mã xáo trộn Bit-plane (Inverse Arnold)
    planes = bit_plane_slice(diff_inv)
    orig_planes = []
    for p in range(1, 9):
        start, end = (p-1)*8, p*8
        sum_h = sum(sha_ints[start:end])
        a, b, it = 1 + (sum_h % 10), 1 + (sum_h % 10), 5 + (sum_h % 5)
        orig_planes.append(inverse_arnold(planes[p-1], a, b, it))
    
    decrypted_np = bit_plane_rejoin(orig_planes)
    return decrypted_np