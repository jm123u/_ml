import random
from micrograd.engine import Value

def function(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

def gradient_descent(learning_rate=0.1, max_iterations=100, tolerance=1e-5):
    x = Value(random.uniform(-5, 5))
    y = Value(random.uniform(-5, 5))
    z = Value(random.uniform(-5, 5))

    for iteration in range(max_iterations):
        loss = function(x, y, z)
        
        x.grad = 0
        y.grad = 0
        z.grad = 0
        
        loss.backward()
        
        x.data -= learning_rate * x.grad
        y.data -= learning_rate * y.grad
        z.data -= learning_rate * z.grad
        
        if abs(loss.data) < tolerance:
            break

    return x.data, y.data, z.data, loss.data

x_opt, y_opt, z_opt, f_opt = gradient_descent()

print(f"x={x_opt:.3f}, y={y_opt:.3f}, z={z_opt:.3f}, f(x,y,z)={f_opt:.3f}")