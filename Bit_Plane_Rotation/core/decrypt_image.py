# decrypt_image.py
import numpy as np
import cv2

from core.encrypt_image import (
    bit_plane_decomposition, 
    combine_bit_planes, 
    select_plane, 
    insert_plane, 
    compute_submatrix_position,
    # compute_rotation_angle
)

def bit_plane_rotation_inverse(img_diffused, keys):
    
    K1 = keys["K1"]
    K5 = keys["K5"]
    K6 = keys["K6"]
    K7 = keys["K7"]
    K8 = keys["K8"]
    K9 = keys["K9"]
    K10 = keys["K10"]

    m, n = img_diffused.shape
    bit_cube = bit_plane_decomposition(img_diffused)
    L = m * n * 8

    
    idx8 = np.argsort(K8)
    idx9 = np.argsort(K9)
    idx10 = np.argsort(K10)

    
    for i in range(L - 1, -1, -1):
        direction = K1[i] % 3

        if direction == 0:

            key2 = K5[i]
            key3 = K8[i]
            key0 = idx8[i]
            plane_count = 8
            plane_m = m
            plane_n = n
            
        elif direction == 1:

            key2 = K6[i]
            key3 = K9[i]
            key0 = idx9[i]
            plane_count = n
            plane_m = m
            plane_n = 8

        else:

            key2 = K7[i]
            key3 = K10[i]
            key0 = idx10[i]
            plane_count = m
            plane_m = n
            plane_n = 8

        # Chọn mặt phẳng và lấy ra
        plane_idx = key2 % plane_count
        plane = select_plane(bit_cube, direction, plane_idx)

        # Tính vị trí sub-matrix (giống hệt lúc mã hóa)
        row, col, size = compute_submatrix_position(key0, plane_m, plane_n)
        # TÍNH GÓC QUAY NGƯỢC
        # Nếu mã hóa quay k*90 độ, thì giải mã quay (4-k)*90 độ
        k_forward = key3 % 4
        k_inverse = (4 - k_forward) % 4
        
        # Xoay ngược ma trận con
        sub = plane[row:row+size, col:col+size]
        plane[row:row+size, col:col+size] = np.rot90(sub, k=k_inverse)

        # Trả lại mặt phẳng vào khối 3D
        bit_cube = insert_plane(bit_cube, direction, plane_idx, plane)

    return combine_bit_planes(bit_cube)

def decrypt_full(cipher_img, keys, Q):

    # Bước 1: Giải khuếch tán trước
    from core.diffusion_inverse import diffusion_inverse
    D_inv = diffusion_inverse(cipher_img, keys["K2"], keys["K3"], Q)
    
    # Bước 2: Xoay ngược bit-plane sau
    P_inv = bit_plane_rotation_inverse(D_inv, keys)
    
    return P_inv