# Diffusion phase of the image encryption algorithm

import numpy as np


def diffusion_phase(D, K2, K3, Q):

    m, n = D.shape

    N = m * n

    D = D.flatten()

    C = np.zeros(N, dtype=np.uint8)


    # -----------------------------
    # pixel đầu tiên
    # -----------------------------

    part1 = (int(K2[0]) + int(K3[0])) % 256
    part2 = (int(D[0]) + int(Q)) % 256

    C[0] = part1 ^ part2


    # -----------------------------
    # pixel tiếp theo
    # -----------------------------

    for i in range(1, N):

        part1 = (int(K2[i]) + int(K3[i])) % 256
        part2 = (int(D[i]) + int(C[i-1])) % 256

        C[i] = part1 ^ part2


    C = C.reshape(m, n)

    return C