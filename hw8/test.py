import torch

x = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

optimizer = torch.optim.SGD([x], lr=0.1)

for i in range(100):
    optimizer.zero_grad()
    
    f = x[0]**2 + x[1]**2 + x[2]**2 - 2*x[0] - 4*x[1] - 6*x[2] + 8
    
    f.backward() 
    optimizer.step() 

    if i % 10 == 0:
        print(f"Step {i}: x = {x.data.numpy()}, f(x) = {f.item():.4f}")
