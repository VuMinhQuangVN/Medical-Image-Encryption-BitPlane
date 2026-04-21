# core/SHA_512.py
import struct
from decimal import Decimal, getcontext

# Thiết lập độ chính xác cao để tính toán căn bậc 2 và 3
getcontext().prec = 100 

def get_first_n_primes(n):
    """Tìm n số nguyên tố đầu tiên"""
    primes = []
    num = 2
    while len(primes) < n:
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                break
        else:
            primes.append(num)
        num += 1
    return primes

def generate_constants():
    """Tự động tính toán H_INIT và K cho SHA-512"""
    primes = get_first_n_primes(80)
    

    h_init = []
    for p in primes[:8]:
        root = Decimal(p).sqrt()
        frac = root - int(root) 
        h_init.append(int(frac * Decimal(2**64)))
        

    k_const = []
    for p in primes:

        root = Decimal(p) ** (Decimal(1) / Decimal(3))
        frac = root - int(root) 
        k_const.append(int(frac * Decimal(2**64)))
        
    return h_init, k_const

# --- KHỞI TẠO HẰNG SỐ TỰ ĐỘNG ---
H_INIT, K = generate_constants()

mask = 0xFFFFFFFFFFFFFFFF

def rotr(x, n):
    return ((x >> n) | (x << (64 - n))) & mask

def sha512_tu_tinh(message_str):
    # 1. Padding
    if isinstance(message_str, str):
        data = bytearray(message_str.encode('utf-8'))
    else:
        data = bytearray(message_str)

    bit_len = len(data) * 8
    data.append(0x80)
    while (len(data) + 16) % 128 != 0:
        data.append(0x00)
    data += struct.pack('>QQ', 0, bit_len)

    h = list(H_INIT)

    # 2. Xử lý khối 1024-bit
    for i in range(0, len(data), 128):
        w = list(struct.unpack('>16Q', data[i:i+128]))
        for t in range(16, 80):
            s0 = rotr(w[t-15], 1) ^ rotr(w[t-15], 8) ^ (w[t-15] >> 7)
            s1 = rotr(w[t-2], 19) ^ rotr(w[t-2], 61) ^ (w[t-2] >> 6)
            w.append((w[t-16] + s0 + w[t-7] + s1) & mask)

        a, b, c, d, e, f, g, hh = h
        for t in range(80):
            S1 = rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41)
            ch = (e & f) ^ ((e ^ mask) & g)
            t1 = (hh + S1 + ch + K[t] + w[t]) & mask
            
            S0 = rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (S0 + maj) & mask

            # Cập nhật biến tạm
            hh, g, f, e, d, c, b, a = g, f, e, (d + t1) & mask, c, b, a, (t1 + t2) & mask

        # Cập nhật hash state
        v = [a, b, c, d, e, f, g, hh]
        for j in range(8):
            h[j] = (h[j] + v[j]) & mask

    return "".join(f"{x:016x}" for x in h)

if __name__ == "__main__":
    msg = "Hello Worldd"
    import hashlib
    print("--- HẰNG SỐ ĐÃ TÍNH ---")
    print(f"H[0] tính được: {hex(H_INIT[0])} (Chuẩn: 0x6a09e667f3bcc908)")
    print(f"K[0] tính được: {hex(K[0])} (Chuẩn: 0x428a2f98d728ae22)")
    print("\n--- KẾT QUẢ BĂM ---")
    print(f"Code tự tính hằng số: {sha512_tu_tinh(msg)}")
    print(f"Hashlib chuẩn:        {hashlib.sha512(msg.encode()).hexdigest()}")