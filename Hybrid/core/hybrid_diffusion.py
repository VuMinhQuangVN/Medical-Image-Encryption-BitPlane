import numpy as np

def hybrid_diffusion_forward(D, K2, K3, Q):
    """
    Giai đoạn Khuếch tán thuận (Mã hóa)
    D: Ảnh đã được xáo trộn (m x n)
    K2, K3: Các chuỗi khóa từ hệ Hyper-Lorenz
    Q: Giá trị pixel mồi (sinh ra từ SHA-512)
    """
    m, n = D.shape
    N = m * n
    D_flat = D.flatten().astype(np.int32)
    C = np.zeros(N, dtype=np.uint8)

    # --- Xử lý pixel đầu tiên ---
    part1 = (int(K2[0]) + int(K3[0])) % 256
    part2 = (D_flat[0] + int(Q)) % 256
    C[0] = part1 ^ part2

    # --- Xử lý các pixel tiếp theo ---
    for i in range(1, N):
        part1 = (int(K2[i]) + int(K3[i])) % 256
        part2 = (D_flat[i] + int(C[i-1])) % 256
        C[i] = part1 ^ part2

    return C.reshape(m, n)

def hybrid_diffusion_backward(C, K2, K3, Q):
    """
    Giai đoạn Khuếch tán nghịch (Giải mã)
    C: Ảnh bản mã (m x n)
    K2, K3: Chuỗi khóa Hyper-Lorenz (phải khớp với lúc mã hóa)
    Q: Giá trị pixel mồi (phải khớp với lúc mã hóa)
    """
    m, n = C.shape
    N = m * n
    C_flat = C.flatten().astype(np.int32)
    D_inv = np.zeros(N, dtype=np.uint8)

    # --- Giải mã các pixel từ cuối lên đầu (trừ pixel 0) ---
    for i in range(N - 1, 0, -1):
        part1 = (int(K2[i]) + int(K3[i])) % 256
        temp = C_flat[i] ^ part1
        D_inv[i] = (temp - C_flat[i-1]) % 256

    # --- Giải mã pixel đầu tiên ---
    part1 = (int(K2[0]) + int(K3[0])) % 256
    temp = C_flat[0] ^ part1
    D_inv[0] = (temp - int(Q)) % 256

    return D_inv.reshape(m, n)