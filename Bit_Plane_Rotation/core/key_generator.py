# Key Generation using Hyper-chaotic Systems

from Bit_Plane_Rotation.core.hyper_lorenz import generate_keys
from Bit_Plane_Rotation.core.hyper6d import generate_keys_6d


def generate_all_keys(x0,y0,z0,w0,
                      x1,x2,x3,x4,x5,x6,
                      N0,
                      m,n):

    L = m * n * 8

    # K1 K2 K3 từ Hyper Lorenz
    K1,K2,K3 = generate_keys(
        x0,y0,z0,w0,
        L,
        N0
    )

    # K5..K10 từ hệ 6D
    K5,K6,K7,K8,K9,K10 = generate_keys_6d(
        x1,x2,x3,x4,x5,x6,
        L,
        N0,
        m,
        n
    )

    return {
        "K1":K1,
        "K2":K2,
        "K3":K3,
        "K5":K5,
        "K6":K6,
        "K7":K7,
        "K8":K8,
        "K9":K9,
        "K10":K10
    }