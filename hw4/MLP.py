import numpy as np

seven_segment_truth_table = {
    0:  (1, 1, 1, 1, 1, 1, 0),
    1:  (0, 1, 1, 0, 0, 0, 0),
    2:  (1, 1, 0, 1, 1, 0, 1),
    3:  (1, 1, 1, 1, 0, 0, 1),
    4:  (0, 1, 1, 0, 0, 1, 1),
    5:  (1, 0, 1, 1, 0, 1, 1),
    6:  (1, 0, 1, 1, 1, 1, 1),
    7:  (1, 1, 1, 0, 0, 0, 0),
    8:  (1, 1, 1, 1, 1, 1, 1),
    9:  (1, 1, 1, 1, 0, 1, 1)
}
binary_outputs = {
    0: (0, 0, 0, 0),
    1: (0, 0, 0, 1),
    2: (0, 0, 1, 0),
    3: (0, 0, 1, 1),
    4: (0, 1, 0, 0),
    5: (0, 1, 0, 1),
    6: (0, 1, 1, 0),
    7: (0, 1, 1, 1),
    8: (1, 0, 0, 0),
    9: (1, 0, 0, 1)
}
X = np.array([seven_segment_truth_table[i] for i in range(10)])  
Y = np.array([binary_outputs[i] for i in range(10)])             

input_size = 7
hidden_size = 5
output_size = 4
lr = 1.0      
h = 1e-4        
max_epochs = 1000
early_stop_threshold = 1e-6


W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros(output_size)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pack(W1, b1, W2, b2):
    return np.concatenate([W1.flatten(), b1, W2.flatten(), b2])

def unpack(vec):
    W1 = vec[0:35].reshape(7, 5)
    b1 = vec[35:40]
    W2 = vec[40:60].reshape(5, 4)
    b2 = vec[60:]
    return W1, b1, W2, b2

def forward_all(X, W1, b1, W2, b2):
    H = sigmoid(X @ W1 + b1)      
    O = sigmoid(H @ W2 + b2)     
    return H, O

def loss(W1, b1, W2, b2):
    _, O = forward_all(X, W1, b1, W2, b2)
    return np.mean((O - Y) ** 2)


def numerical_gradient(f, param, h=1e-4):
    grad = np.zeros_like(param)
    for idx in np.ndindex(*param.shape):
        orig = param[idx]
        param[idx] = orig + h
        fxh1 = f()
        param[idx] = orig - h
        fxh2 = f()
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        param[idx] = orig
    return grad

params = pack(W1, b1, W2, b2)
def loss_wrapper():
    W1_, b1_, W2_, b2_ = unpack(params)
    return loss(W1_, b1_, W2_, b2_)


for epoch in range(max_epochs):
    grad = numerical_gradient(loss_wrapper, params, h=h)
    grad_norm = np.linalg.norm(grad)
    if grad_norm < early_stop_threshold:
        print(f"Early stop at epoch {epoch}: gradient norm {grad_norm:.8f}")
        break
    params -= lr * grad
    if epoch % 100 == 0 or epoch == max_epochs - 1:
        print(f"Epoch {epoch:04d}, Loss = {loss_wrapper():.6f}, Grad Norm = {grad_norm:.6f}")

W1, b1, W2, b2 = unpack(params)
_, predictions = forward_all(X, W1, b1, W2, b2)
predictions_binary = np.round(predictions).astype(int)

for i in range(10):
    pred = predictions_binary[i]
    print(f"{i}: Forecast = {pred.tolist()}  accurate = {Y[i].tolist()}")
