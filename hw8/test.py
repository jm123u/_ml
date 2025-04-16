import torch

# 初始變數設定（可自由選擇起始點）
x = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

# 優化器設定（使用 SGD 優化器）
optimizer = torch.optim.SGD([x], lr=0.1)

# 優化迴圈
for i in range(100):
    optimizer.zero_grad()
    
    # 函數定義
    f = x[0]**2 + x[1]**2 + x[2]**2 - 2*x[0] - 4*x[1] - 6*x[2] + 8
    
    f.backward()  # 計算梯度
    optimizer.step()  # 更新變數

    if i % 10 == 0:
        print(f"Step {i}: x = {x.data.numpy()}, f(x) = {f.item():.4f}")
