import torch
import matplotlib.pyplot as plt
import numpy as np

# 加载 AWQ 搜索结果文件
# awq_results_path = "awq_cache/llama3-8b-w4-g128.pt"
awq_results_path = "awq_cache/opt-w4-g128.pt"
awq_results = torch.load(awq_results_path, map_location="cpu")

# 查看搜索结果的内容
print("AWQ 搜索结果内容:")
for key, value in awq_results.items():
    print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")

# 假设 'important_weights' 是我们要绘制的权重重要性
# 由于示例中没有明确的 'important_weights' 键，我们使用示例中的权重数据
# 这里我们使用 'model.decoder.layers.31.fc2' 作为示例
important_weights_key = 'model.decoder.layers.31.fc2'
important_weights = awq_results.get(important_weights_key, None)

if important_weights is not None:
    # 将张量转换为 NumPy 数组
    important_weights_np = important_weights.numpy()
    
    # 检查张量的形状
    print(f"重要权重形状: {important_weights_np.shape}")

    # 假设我们要绘制每个权重的平均值
    avg_weights = np.mean(important_weights_np, axis=1)
    
    # 为了绘制，我们将权重展平成一维
    avg_weights_flat = avg_weights.flatten()

    # 创建绘图
    plt.figure(figsize=(10, 6))
    plt.plot(avg_weights_flat, marker='o', linestyle='-', color='b')
    plt.title('重要权重图')
    plt.xlabel('权重索引')
    plt.ylabel('权重值')
    plt.grid(True)
    plt.show()
else:
    print("没有找到重要权重索引。")