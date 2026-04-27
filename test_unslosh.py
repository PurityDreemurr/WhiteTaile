import torch
import sys

print("=" * 40)
# 1. 基础环境
print(f"Python 版本: {sys.version.split()[0]}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 驱动版本: {torch.version.cuda}")

# 2. 显卡硬件确认 (修复之前的报错)
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"找到显卡: {torch.cuda.get_device_name(0)}")
    print(f"计算能力: {props.major}.{props.minor}")
    print(f"显存总量: {props.total_memory / 1024**3:.2f} GB")
else:
    print("错误: 未检测到 GPU，请检查驱动！")

# 3. Unsloth 挂载测试 (不加载模型，只测补丁能否初始化)
try:
    from unsloth import FastLanguageModel
    print("Unsloth 补丁: 挂载成功！")
except Exception as e:
    print(f"Unsloth 补丁: 挂载失败，原因: {e}")

# 4. 算力测试 (做一个简单的矩阵乘法，验证 CUDA 调用链)
try:
    x = torch.randn(100, 100).to("cuda")
    y = torch.randn(100, 100).to("cuda")
    z = torch.matmul(x, y)
    print("CUDA 算力测试: 成功 (矩阵运算正常)")
except Exception as e:
    print(f"CUDA 算力测试: 失败，原因: {e}")
print("=" * 40)