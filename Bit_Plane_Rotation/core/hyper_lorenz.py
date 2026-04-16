import math

# 1. Định nghĩa đạo hàm của hệ (Vector field)
def lorenz_deriv(x, y, z, w, a=10, b=8/3, c=28, r=-1):
    dx = a * (y - z) + w
    dy = c * x - y - x * z
    dz = x * y - b * z
    dw = -y * z + r * w
    return dx, dy, dz, dw

# 2. Giải bằng thuật toán Runge-Kutta bậc 4 (RK4) - Cực kỳ ổn định
def hyper_lorenz_step_rk4(x, y, z, w, a, b, c, r, h):
    # k1
    k1x, k1y, k1z, k1w = lorenz_deriv(x, y, z, w, a, b, c, r)
    
    # k2
    k2x, k2y, k2z, k2w = lorenz_deriv(x + h/2*k1x, y + h/2*k1y, z + h/2*k1z, w + h/2*k1w, a, b, c, r)
    
    # k3
    k3x, k3y, k3z, k3w = lorenz_deriv(x + h/2*k2x, y + h/2*k2y, z + h/2*k2z, w + h/2*k2w, a, b, c, r)
    
    # k4
    k4x, k4y, k4z, k4w = lorenz_deriv(x + h*k3x, y + h*k3y, z + h*k3z, w + h*k3w, a, b, c, r)
    
    # Kết hợp lại theo trọng số RK4
    new_x = x + (h/6) * (k1x + 2*k2x + 2*k3x + k4x)
    new_y = y + (h/6) * (k1y + 2*k2y + 2*k3y + k4y)
    new_z = z + (h/6) * (k1z + 2*k2z + 2*k3z + k4z)
    new_w = w + (h/6) * (k1w + 2*k2w + 2*k3w + k4w)
    
    # Kiểm tra an toàn: Nếu bắt đầu có dấu hiệu vọt quá xa, ép nó lại
    if math.isinf(new_x) or math.isnan(new_x) or abs(new_x) > 1e10:
        return 0.1, 0.1, 0.1, 0.1 # Reset về giá trị nhỏ nếu bị nổ
        
    return new_x, new_y, new_z, new_w

def generate_chaos_sequence(x0, y0, z0, w0, steps, N0, h=0.001):
    # Khởi tạo tham số chuẩn của hệ Hyper-Lorenz
    a, b, c, r = 10, 8/3, 28, -1
    x, y, z, w = x0, y0, z0, w0

    xs, ys, zs, ws = [], [], [], []

    for i in range(steps + N0):
        # Dùng RK4 thay vì Euler
        x, y, z, w = hyper_lorenz_step_rk4(x, y, z, w, a, b, c, r, h)

        if i >= N0:
            xs.append(x)
            ys.append(y)
            zs.append(z)
            ws.append(w)

    return xs, ys, zs, ws

def generate_keys(x0, y0, z0, w0, L, N0):
    # Tăng độ ổn định bằng cách dùng h nhỏ và RK4
    xs, ys, zs, ws = generate_chaos_sequence(x0, y0, z0, w0, L, N0, h=0.001)

    K1, K2, K3 = [], [], []
    for i in range(L):
        # Dùng trị tuyệt đối để tránh lỗi floor với số âm
        val1 = abs(xs[i] + ws[i]) * 1e13
        K1.append(math.floor(val1 % 3))

    for i in range(L // 8):
        val2 = abs(ys[i]) * 1e13
        val3 = abs(zs[i]) * 1e13
        K2.append(math.floor(val2 % 256))
        K3.append(math.floor(val3 % 256))

    return K1, K2, K3