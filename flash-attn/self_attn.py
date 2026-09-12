import torch
import math
import numpy as np
import os

# -------------------------------
# .bin 文件统一存放目录
# -------------------------------
BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(BIN_DIR, exist_ok=True)

# -------------------------------
# PyTorch 实现注意力机制
# -------------------------------
def self_attention_pytorch(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / math.sqrt(Q.size(-1))
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output

# -------------------------------
# 参数设置
# -------------------------------
m, n = 64, 128  # 可根据你的测试需求修改
dtype = torch.float32

# 设置随机种子以确保可复现性（便于和 CUDA 对比）
torch.manual_seed(42)

# 生成 Q, K, V (m, n)
Q = torch.randn(m, n, dtype=dtype)
K = torch.randn(m, n, dtype=dtype)
V = torch.randn(m, n, dtype=dtype)

# 可选：移动到 GPU 计算（结果仍可保存为 CPU numpy）
Q, K, V = Q.cuda(), K.cuda(), V.cuda()

# -------------------------------
# 执行自注意力
# -------------------------------
with torch.no_grad():
    O_torch = self_attention_pytorch(Q, K, V)


# -------------------------------
# 保存 Q, K, V, O 到 .bin 文件
# 使用 numpy 保存为 float32 格式
# -------------------------------

def save_tensor_bin(tensor, filename):
    """Save a PyTorch tensor as binary file (float32)."""
    tensor_np = tensor.cpu().numpy()
    tensor_np.astype(np.float32).tofile(filename)
    print(f"Saved {filename} with shape {tensor.shape}, dtype {tensor_np.dtype}")

save_tensor_bin(Q, os.path.join(BIN_DIR, "Q.bin"))
save_tensor_bin(K, os.path.join(BIN_DIR, "K.bin"))
save_tensor_bin(V, os.path.join(BIN_DIR, "V.bin"))
save_tensor_bin(O_torch, os.path.join(BIN_DIR, "O_torch.bin"))

print("✅ All tensors saved as .bin files.")
