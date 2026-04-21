import numpy as np
import cv2
from Medical.core.medical_utils import get_sha512_ints, get_sha512_params, get_sha512_params_analysis, arnold_transform, get_logistic_map

def encrypt_medical(img_input, key_str, is_analysis=False):
    # Chuẩn hóa đầu vào
    img = img_input.copy() if isinstance(img_input, np.ndarray) else cv2.imread(img_input, 0)
    if img.shape != (256, 256):
        img = cv2.resize(img, (256, 256))
    N = img.shape[0]

    # Chọn mode băm SHA-512
    raw_hex = get_sha512_params_analysis(key_str, img) if is_analysis else get_sha512_params(key_str)
    hash_ints = get_sha512_ints(raw_hex)
    # 1. Bit plane slicing
    bit_planes = [(img >> i) & 1 for i in range(8)]
    
    # 2. Arnold Scrambling
    scrambled_planes = []
    for p in range(1, 9):
        start, end = (p-1)*8, p*8
        sum_h = sum(hash_ints[start:end])
        a, b, it = 1 + (sum_h % 10), 1 + (sum_h % 10), 5 + (sum_h % 5)
        scrambled_planes.append(arnold_transform(bit_planes[p-1], a, b, it))
    
    # 3. Rejoin
    c1 = np.zeros_like(img)
    for i in range(8): c1 |= (scrambled_planes[i] << i)
        
    # 4. Diffusion
    std_h = np.std(hash_ints)
    k2 = 3.5 + ((std_h - np.floor(std_h)) * 0.4999)
    logistic_key = get_logistic_map(k2, img.size).reshape(N, N)
    
    cipher = cv2.bitwise_xor(c1, logistic_key)
    return cipher, k2

def encrypt_medical_with_intermediate(img_input, key_str, is_analysis=False):
    """Hàm mã hóa Y tế trả về thêm ảnh trung gian c1 để mô phỏng luồng kỹ thuật"""
    # Chuẩn hóa đầu vào
    img = img_input.copy() if isinstance(img_input, np.ndarray) else cv2.imread(img_input, 0)
    if img.shape != (256, 256):
        img = cv2.resize(img, (256, 256))
    N = img.shape[0]

    # Chọn mode băm SHA-512
    raw_hex = get_sha512_params_analysis(key_str, img) if is_analysis else get_sha512_params(key_str)
    hash_ints = get_sha512_ints(raw_hex)
    
    # 1. Bit plane slicing
    bit_planes = [(img >> i) & 1 for i in range(8)]
    
    # 2. Arnold Scrambling (Confusion)
    scrambled_planes = []
    for p in range(1, 9):
        start, end = (p-1)*8, p*8
        sum_h = sum(hash_ints[start:end])
        a, b, it = 1 + (sum_h % 10), 1 + (sum_h % 10), 5 + (sum_h % 5)
        scrambled_planes.append(arnold_transform(bit_planes[p-1], a, b, it))
    
    # 3. Rejoin -> Đây là ảnh trung gian c1 (Confusion Result)
    c1 = np.zeros_like(img)
    for i in range(8): 
        c1 |= (scrambled_planes[i] << i)
        
    # 4. Diffusion (Khuếch tán)
    std_h = np.std(hash_ints)
    k2 = 3.5 + ((std_h - np.floor(std_h)) * 0.4999)
    logistic_key = get_logistic_map(k2, img.size).reshape(N, N)
    
    cipher = cv2.bitwise_xor(c1, logistic_key)
    
    # TRẢ VỀ 3 THÀNH PHẦN
    return c1, cipher, k2