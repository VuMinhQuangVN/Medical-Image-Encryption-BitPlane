import numpy as np
from Bit_Plane_Rotation.core.diffusion_phase import diffusion_phase
import cv2

# =========================================================
# Step 1: Bit-plane decomposition
# =========================================================

def bit_plane_decomposition(img):

    m, n = img.shape

    bit_planes = np.zeros((m, n, 8), dtype=np.uint8)

    for k in range(8):
        bit_planes[:, :, k] = (img >> k) & 1

    return bit_planes


# =========================================================
# Combine bit-planes back to image
# =========================================================

def combine_bit_planes(bit_planes):

    m, n, _ = bit_planes.shape

    img = np.zeros((m, n), dtype=np.uint8)

    for k in range(8):
        img = img | (bit_planes[:, :, k].astype(np.uint8) << k)

    return img


# =========================================================
# Rotate sub-matrix
# =========================================================

def rotate_submatrix(matrix, row, col, size, angle):

    sub = matrix[row:row+size, col:col+size]

    k = angle // 90

    sub_rot = np.rot90(sub, k=k)

    matrix[row:row+size, col:col+size] = sub_rot

    return matrix


# =========================================================
# Compute sub-matrix position (Step 4)
# =========================================================

def compute_submatrix_position(key0, m, n):

    temp = key0 % (m*n)

    if temp == 0:
        temp = m*n

    row = int(np.ceil(temp / n))

    col = temp % n
    if col == 0:
        col = n

    r = m - row + 1
    c = n - col + 1

    size = min(r, c)

    return row-1, col-1, size


# =========================================================
# Compute rotation angle (Step 5)
# =========================================================

def compute_rotation_angle(key3):

    if key3 == 0:
        return 0
    elif key3 == 1:
        return 90
    elif key3 == 2:
        return 180
    else:
        return 270


# =========================================================
# Select plane depending on direction
# =========================================================

def select_plane(bit_cube, direction, plane_index):

    if direction == 0:      # x-y
        return bit_cube[:, :, plane_index]

    elif direction == 1:    # x-z
        return bit_cube[:, plane_index, :]

    else:                   # y-z
        return bit_cube[plane_index, :, :]


def insert_plane(bit_cube, direction, plane_index, plane):

    if direction == 0:
        bit_cube[:, :, plane_index] = plane

    elif direction == 1:
        bit_cube[:, plane_index, :] = plane

    else:
        bit_cube[plane_index, :, :] = plane

    return bit_cube


# =========================================================
# Bit-plane rotation phase (Section 3.2)
# =========================================================

def bit_plane_rotation(img, keys):

    K1 = keys["K1"]
    K5 = keys["K5"]
    K6 = keys["K6"]
    K7 = keys["K7"]
    K8 = keys["K8"]
    K9 = keys["K9"]
    K10 = keys["K10"]

    bit_cube = bit_plane_decomposition(img)

    m, n = img.shape

    L = m*n*8

    idx8 = np.argsort(K8)
    idx9 = np.argsort(K9)
    idx10 = np.argsort(K10)

    for i in range(L):

        # ---------------------------------
        # Step 2: rotation direction
        # ---------------------------------

        direction = K1[i] % 3

        if direction == 0:

            key2 = K5
            key3 = K8
            key0 = idx8[i]
            plane_count = 8
            plane_m = m
            plane_n = n

        elif direction == 1:

            key2 = K6
            key3 = K9
            key0 = idx9[i]
            plane_count = n
            plane_m = m
            plane_n = 8

        else:

            key2 = K7
            key3 = K10
            key0 = idx10[i]
            plane_count = m
            plane_m = n
            plane_n = 8


        # ---------------------------------
        # Step 3: select plane
        # ---------------------------------

        plane_index = key2[i] % plane_count

        plane = select_plane(bit_cube, direction, plane_index)


        # ---------------------------------
        # Step 4: compute submatrix
        # ---------------------------------


        row, col, size = compute_submatrix_position(
            key0,
            plane_m,
            plane_n
        )


        # ---------------------------------
        # Step 5: rotation
        # ---------------------------------

        angle = compute_rotation_angle(key3[i] % 4)

        plane = rotate_submatrix(
            plane,
            row,
            col,
            size,
            angle
        )


        bit_cube = insert_plane(
            bit_cube,
            direction,
            plane_index,
            plane
        )


    encrypted = combine_bit_planes(bit_cube)

    return encrypted

def encrypt_image(img, keys, Q):


    # phase 1
    D = bit_plane_rotation(img, keys)

    # phase 2
    C = diffusion_phase(
        D,
        keys["K2"],
        keys["K3"],
        Q
    )

    return C

def encrypt_image_with_intermediate(img, keys, Q):
    """Hàm mã hóa trả về cả ảnh trung gian để làm Dashboard"""
    D = bit_plane_rotation(img, keys)
    

    C = diffusion_phase(D, keys["K2"], keys["K3"], Q)
    
    return D, C  