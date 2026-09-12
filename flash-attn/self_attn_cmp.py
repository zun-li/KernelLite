import numpy as np
import os

# -------------------------------
# .bin 文件统一存放目录
# -------------------------------
BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

m, n = 64, 128

# Load PyTorch output and CUDA output
O_torch = np.fromfile(os.path.join(BIN_DIR, "O_torch.bin"), dtype=np.float32).reshape(m, n)
O_cuda  = np.fromfile(os.path.join(BIN_DIR, "O_cuda.bin"), dtype=np.float32).reshape(m, n)

# Compute error
diff = O_torch - O_cuda
max_error = np.abs(diff).max()
mse = (diff ** 2).mean()

print("🔍 Comparison Result:")
print(f"Max absolute error: {max_error:.6e}")
print(f"MSE: {mse:.6e}")

# Optional: print matrices
# print("\nPyTorch Output O:")
# print(O_torch)

# print("\nCUDA Output O:")
# print(O_cuda)

assert max_error < 1e-5, "❌ CUDA output differs too much from PyTorch!"
print("\n✅ PASSED: CUDA and PyTorch outputs are numerically close.")
