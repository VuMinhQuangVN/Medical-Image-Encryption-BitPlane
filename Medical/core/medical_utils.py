import numpy as np
import hashlib

def get_sha512_params(key_str):
    return hashlib.sha512(key_str.encode()).hexdigest()

def get_sha512_params_analysis(key_str, img_np):
    combined_data = key_str.encode() + img_np.tobytes()
    return hashlib.sha512(combined_data).hexdigest()

def arnold_transform(matrix, a, b, iterations):
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

def get_logistic_map(r, size, x0=0.5):
    x = x0
    key = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x = r * x * (1 - x)
        key[i] = int(x * 255) % 256
    return key

def get_sha512_ints(sha512_hex):
    """Chuyển chuỗi Hex 128 ký tự thành danh sách 64 số nguyên (0-255)"""
    return [int(sha512_hex[i:i+2], 16) for i in range(0, 128, 2)]