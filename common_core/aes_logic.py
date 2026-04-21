# core/aes_logic.py
from Crypto.Cipher import AES
from Crypto.Util import Counter
import hashlib
import numpy as np

def run_aes_benchmark(img_np, password, key_size=16): # key_size=16 là AES-128, 32 là AES-256
    h, w = img_np.shape
    data = img_np.tobytes()
    # Dùng MD5 để tạo key 128-bit (16 bytes) cho AES-128
    if key_size == 16:
        key = hashlib.md5(password.encode()).digest()
    else:
        key = hashlib.sha256(password.encode()).digest()
        
    ctr = Counter.new(128)
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
    processed_data = cipher.encrypt(data)
    return np.frombuffer(processed_data, dtype=np.uint8).reshape((h, w))