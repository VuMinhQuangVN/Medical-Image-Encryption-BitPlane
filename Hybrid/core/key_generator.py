# core/key_generator.py
from core.chaos_engine import generate_hybrid_keys
from core.hybrid_utils import compute_N0_Q_hybrid

def get_all_keys_hybrid(sha512_hex, m, n):
    """
    sha512_hex: Chuỗi 128 ký tự hex từ SHA-512
    m, n: Kích thước ảnh
    """
    # 1. Chuyển Hex sang list số nguyên (0-255) để tính toán x0, y0, z0, w0
    # Mỗi 2 ký tự hex là 1 byte số nguyên -> 128 hex chars = 64 bytes
    sha512_ints = [int(sha512_hex[i:i+2], 16) for i in range(0, 128, 2)]

    # 2. Ánh xạ sang các giá trị thực khởi tạo cho hệ Hyper-Lorenz
    # Chia cho 2040 (255*8) để đưa về khoảng [0, 1] rồi cộng offset để tránh điểm kỳ dị
    x0 = (sum(sha512_ints[0:8]) / 2040) + 0.1234
    y0 = (sum(sha512_ints[8:16]) / 2040) + 0.5678
    z0 = (sum(sha512_ints[16:24]) / 2040) + 0.9101
    w0 = (sum(sha512_ints[24:32]) / 2040) + 0.1121
    
    # 3. Tính H, N0, Q từ chuỗi Hex (Kế thừa giáo sư)
    H, N0, Q = compute_N0_Q_hybrid(sha512_hex)

    # 4. Sinh khóa K2, K3 từ động cơ Chaos (độ dài m*n)
    K2, K3 = generate_hybrid_keys(x0, y0, z0, w0, m, n, N0)

    return {
        "K2": K2,
        "K3": K3,
        "Q": Q,
        "H": H  
    }