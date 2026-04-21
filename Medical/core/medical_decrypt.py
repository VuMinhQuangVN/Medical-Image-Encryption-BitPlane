import numpy as np
import cv2
from Medical.core.medical_utils import get_sha512_ints, get_sha512_params, get_sha512_params_analysis, inverse_arnold, get_logistic_map

def decrypt_medical(cipher_input, key_str, k2, original_ref=None):
    cipher = cipher_input.copy() if isinstance(cipher_input, np.ndarray) else cv2.imread(cipher_input, 0)
    if cipher.shape != (256, 256):
        cipher = cv2.resize(cipher, (256, 256))
    N = cipher.shape[0]

    # Nếu có original_ref thì dùng mode analysis
    if original_ref is not None:
        if original_ref.shape != (256, 256):
            original_ref = cv2.resize(original_ref, (256, 256), interpolation=cv2.INTER_LINEAR)
        raw_hex = get_sha512_params_analysis(key_str, original_ref)
    else:
        raw_hex = get_sha512_params(key_str)
    
    hash_ints = get_sha512_ints(raw_hex)
    # 1. Reverse Diffusion
    logistic_key = get_logistic_map(k2, cipher.size).reshape(N, N)
    c1 = cv2.bitwise_xor(cipher, logistic_key)
    
    # 2. Bit plane slicing
    bit_planes = [(c1 >> i) & 1 for i in range(8)]
    
    # 3. Inverse Arnold
    original_planes = []
    for p in range(1, 9):
        start, end = (p-1)*8, p*8
        sum_h = sum(hash_ints[start:end])
        a, b, it = 1 + (sum_h % 10), 1 + (sum_h % 10), 5 + (sum_h % 5)
        original_planes.append(inverse_arnold(bit_planes[p-1], a, b, it))
        
    # 4. Rejoin
    decrypted = np.zeros_like(cipher)
    for i in range(8): decrypted |= (original_planes[i] << i)
        
    return decrypted