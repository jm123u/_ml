import numpy as np

def hill_climb_minimize(alpha=0.1, tol=1e-6, max_iters=1000):
    xyz = np.random.rand(3) * 10
    for _ in range(max_iters):
        grad = 2 * xyz - np.array([2, 4, 6])
        if np.linalg.norm(grad) < tol:
            break
        xyz -= alpha * grad
    return *xyz, xyz @ xyz - 2 * xyz[0] - 4 * xyz[1] - 6 * xyz[2] + 8

min_x, min_y, min_z, min_f = hill_climb_minimize()

print(f"minimum value point: x={min_x:.6f}, y={min_y:.6f}, z={min_z:.6f}")
print(f"function minimum: f={min_f:.6f}")