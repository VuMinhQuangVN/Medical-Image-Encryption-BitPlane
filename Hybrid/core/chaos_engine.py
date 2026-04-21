import math
import numpy as np

def lorenz_deriv(x, y, z, w, a=10, b=8/3, c=28, r=-1):
    dx = a * (y - z) + w
    dy = c * x - y - x * z
    dz = x * y - b * z
    dw = -y * z + r * w
    return dx, dy, dz, dw

def hyper_lorenz_step_rk4(x, y, z, w, a, b, c, r, h):
    k1x, k1y, k1z, k1w = lorenz_deriv(x, y, z, w, a, b, c, r)
    k2x, k2y, k2z, k2w = lorenz_deriv(x + h/2*k1x, y + h/2*k1y, z + h/2*k1z, w + h/2*k1w, a, b, c, r)
    k3x, k3y, k3z, k3w = lorenz_deriv(x + h/2*k2x, y + h/2*k2y, z + h/2*k2z, w + h/2*k2w, a, b, c, r)
    k4x, k4y, k4z, k4w = lorenz_deriv(x + h*k3x, y + h*k3y, z + h*k3z, w + h*k3w, a, b, c, r)
    
    new_x = x + (h/6) * (k1x + 2*k2x + 2*k3x + k4x)
    new_y = y + (h/6) * (k1y + 2*k2y + 2*k3y + k4y)
    new_z = z + (h/6) * (k1z + 2*k2z + 2*k3z + k4z)
    new_w = w + (h/6) * (k1w + 2*k2w + 2*k3w + k4w)
    
    if math.isinf(new_x) or math.isnan(new_x) or abs(new_x) > 1e10:
        return 0.1234, 0.5678, 0.9101, 0.1121
        
    return new_x, new_y, new_z, new_w

def generate_hybrid_keys(x0, y0, z0, w0, m, n, N0, h=0.001):
    """
    Sinh khóa K2, K3 có độ dài m*n phục vụ cho Diffusion
    """
    a, b, c, r = 10, 8/3, 28, -1
    x, y, z, w = x0, y0, z0, w0
    
    L = m * n
    
    K2 = np.zeros(L, dtype=np.uint8)
    K3 = np.zeros(L, dtype=np.uint8)

    for _ in range(N0):
        x, y, z, w = hyper_lorenz_step_rk4(x, y, z, w, a, b, c, r, h)

    for i in range(L):
        x, y, z, w = hyper_lorenz_step_rk4(x, y, z, w, a, b, c, r, h)
        
        K2[i] = int(abs(y) * 10**13) % 256
        K3[i] = int(abs(z) * 10**13) % 256

    return K2, K3