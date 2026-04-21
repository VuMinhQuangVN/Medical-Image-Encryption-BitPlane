import numpy as np
import hashlib
from Hybrid.core.SHA_512 import sha512_tu_tinh

# --- NHÓM 1: XỬ LÝ SHA-512 & THAM SỐ HỖN ĐỘN ---

def get_sha512_hex(key_str):
    """
    Tạo mã băm SHA-512 dạng Hex (128 ký tự) từ chuỗi mật khẩu (Key).
    Sử dụng thuật toán SHA-512 tự triển khai (Custom Implementation).
    """
    hash_hex = sha512_tu_tinh(key_str)
    
    return hash_hex

def sha512_to_decimal(sha512_hex):
    """Chia SHA-512 thành 4 phần và XOR lại để lấy giá trị H thập phân khổng lồ"""
    h1 = int(sha512_hex[0:32], 16)
    h2 = int(sha512_hex[32:64], 16)
    h3 = int(sha512_hex[64:96], 16)
    h4 = int(sha512_hex[96:128], 16)
    return h1 ^ h2 ^ h3 ^ h4

def compute_N0_Q_hybrid(sha512_hex):
    """Kế thừa logic giáo sư: Tính số bước bỏ qua (N0) và pixel mồi (Q)"""
    H = sha512_to_decimal(sha512_hex)
    N0 = 1000 + (H % 500)  
    Q = H % 256           
    return H, N0, Q

# --- NHÓM 2: XỬ LÝ BIT-PLANE & ARNOLD (PHẦN XÁO TRỘN) ---

def bit_plane_slice(img):
    """Cắt ảnh xám thành 8 mặt phẳng bit (từ bit 0 đến bit 7)"""
    return [(img >> i) & 1 for i in range(8)]

def bit_plane_slice1(img):
    """Cắt ảnh xám thành 8 mặt phẳng bit (từ bit 0 đến bit 7)"""
    return [((img >> i) & 1) * 255 for i in range(8)]

def bit_plane_rejoin(planes):
    """Gộp 8 mặt phẳng bit lại thành ảnh xám 8-bit hoàn chỉnh"""
    res = np.zeros_like(planes[0], dtype=np.uint8)
    for i in range(8):
        res |= (planes[i] << i)
    return res

def arnold_transform(matrix, a, b, iterations):
    """Thuật toán Arnold Scrambling để hoán vị vị trí pixel/bit"""
    N = matrix.shape[0]
    current = matrix.copy()
    for _ in range(iterations):
        res = np.zeros_like(matrix)
        for x in range(N):
            for y in range(N):
                nx = (x + a * y) % N
                ny = (b * x + (a * b + 1) * y) % N
                res[nx, ny] = current[x, y]
        current = res.copy()
    return current

def inverse_arnold(matrix, a, b, iterations):
    """Arnold nghịch đảo để khôi phục lại vị trí ban đầu"""
    N = matrix.shape[0]
    current = matrix.copy()
    for _ in range(iterations):
        res = np.zeros_like(matrix)
        for x in range(N):
            for y in range(N):
                nx = ((a * b + 1) * x - a * y) % N
                ny = (-b * x + y) % N
                res[nx, ny] = current[x, y]
        current = res.copy()
    return current

# --- NHÓM 3: TIỆN ÍCH CHUYỂN ĐỔI ---

def get_sha512_ints(sha512_hex):
    """Chuyển chuỗi Hex 128 ký tự thành danh sách 64 số nguyên (0-255)"""
    return [int(sha512_hex[i:i+2], 16) for i in range(0, 128, 2)]