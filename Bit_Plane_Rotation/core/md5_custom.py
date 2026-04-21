# MD5 Custom Implementation in Python
import math
import hashlib

A_INIT = 0x67452301
B_INIT = 0xefcdab89
C_INIT = 0x98badcfe
D_INIT = 0x10325476


K = [int(0x100000000 * abs(math.sin(i + 1))) & 0xFFFFFFFF for i in range(64)]


S = [7, 12, 17, 22] * 4 + [5,  9, 14, 20] * 4 + \
    [4, 11, 16, 23] * 4 + [6, 10, 15, 21] * 4


def F(b, c, d): return (b & c) | (~b & d)
def G(b, c, d): return (b & d) | (c & ~d)
def H(b, c, d): return b ^ c ^ d
def I(b, c, d): return c ^ (b | ~d)

def padding(message):
    msg_bytes = bytearray(message)
    orig_len_bits = (len(msg_bytes) * 8) & 0xffffffffffffffff
    
    msg_bytes.append(0x80)
    
    while len(msg_bytes) % 64 != 56:
        msg_bytes.append(0)
        
    msg_bytes += orig_len_bits.to_bytes(8, byteorder='little')
    
    return msg_bytes

def left_rotate(x, amount):
    return ((x << amount) | (x >> (32 - amount))) & 0xFFFFFFFF

def main_loop(padded_msg):
    A, B, C, D = A_INIT, B_INIT, C_INIT, D_INIT

    for chunk_offset in range(0, len(padded_msg), 64):
        chunk = padded_msg[chunk_offset:chunk_offset+64]
        
        M = [int.from_bytes(chunk[i:i+4], byteorder='little') for i in range(0, 64, 4)]
        
        AA, BB, CC, DD = A, B, C, D

        for i in range(64):
            if 0 <= i <= 15:
                f = F(B, C, D)
                g = i
            elif 16 <= i <= 31:
                f = G(B, C, D)
                g = (5 * i + 1) % 16
            elif 32 <= i <= 47:
                f = H(B, C, D)
                g = (3 * i + 5) % 16
            else:
                f = I(B, C, D)
                g = (7 * i) % 16

            total = (A + f + K[i] + M[g]) & 0xFFFFFFFF
            new_B = (B + left_rotate(total, S[i])) & 0xFFFFFFFF
            
            A, B, C, D = D, new_B, B, C

        A = (A + AA) & 0xFFFFFFFF
        B = (B + BB) & 0xFFFFFFFF
        C = (C + CC) & 0xFFFFFFFF
        D = (D + DD) & 0xFFFFFFFF

    return A, B, C, D

def format_result(A, B, C, D):
    res = b""
    for x in [A, B, C, D]:
        res += x.to_bytes(4, byteorder='little')
    return res.hex()


def my_md5_string_tool(input_string):

    data_bytes = input_string.encode('utf-8')

    data = padding(data_bytes)
    a, b, c, d = main_loop(data)

    md5_hex = format_result(a, b, c, d)

    H, N0, Q = compute_N0_Q(md5_hex)

    return {
        "md5_hex": md5_hex,
        "H": H,
        "N0": N0,
        "Q": Q
    }


def md5_to_decimal(md5_hex):
    return int(md5_hex, 16)

def compute_N0_Q(md5_hex):

    H = md5_to_decimal(md5_hex)

    N0 = H % 1500
    Q = H % 255

    return H, N0, Q

def derive_initial_values(md5_hex):
    """
    Trích xuất 10 tham số x0, y0, z0, w0, x10-x60 từ chuỗi MD5
    """
    values = []
    for i in range(10):
        start = i * 3
        chunk = md5_hex[start:start+3]
        val = (int(chunk, 16) / 4095.0) * 2.0 - 1.0
        if abs(val) < 0.001: val = 0.5 
        values.append(val)
        
    return {
        "x0": values[0], "y0": values[1], "z0": values[2], "w0": values[3],
        "x10": values[4], "x20": values[5], "x30": values[6],
        "x40": values[7], "x50": values[8], "x60": values[9]
    }

if __name__ == "__main__":
    test_str = "Hệ thống mã hóa hỗn loạn 2026"
    result = my_md5_string_tool(test_str)
    print(f"Chuỗi đầu vào: {test_str}")
    print(f"Mã băm MD5   : {result['md5_hex']}")
    print(f"Giá trị H     : {result['H']}")
    print(f"Giá trị N0    : {result['N0']}")
    print(f"Giá trị Q     : {result['Q']}")
    
    params = derive_initial_values(result['md5_hex'])
    print(f"Tham số x0    : {params['x0']}")