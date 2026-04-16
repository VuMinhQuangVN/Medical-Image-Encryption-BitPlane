# diffusion_inverse.py
import numpy as np

def diffusion_inverse(C, K2, K3, Q):
    m, n = C.shape
    N = m * n
    C = C.flatten().astype(np.int32) # Dùng int32 để tính toán không bị lỗi uint8
    D_inv = np.zeros(N, dtype=np.uint8)

    for i in range(N - 1, 0, -1):
        
        part1 = (int(K2[i]) + int(K3[i])) % 256
        
        temp = int(C[i]) ^ part1
        
        D_inv[i] = (temp - C[i-1]) % 256

    
    part1 = (int(K2[0]) + int(K3[0])) % 256
    temp = int(C[0]) ^ part1
    D_inv[0] = (temp - int(Q)) % 256

    return D_inv.reshape(m, n)