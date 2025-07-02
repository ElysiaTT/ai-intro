import torch
from torch.nn.functional import scaled_dot_product_attention
# 第一维：批次大小（Batch Size），这里为 1。
# 第二维：序列长度（Sequence Length），这里为 256。
# 第三维：头的数量（Number of Heads），这里为 8。
# 第四维：每个头的特征维度（Head Dimension），这里为 64。

# shapes: (batch_size, seq_len, num_heads, head_dim) 
query = torch.randn(1, 256, 8, 64)
key = torch.randn(1, 256, 8, 64)
value = torch.randn(1, 256, 8, 64)
output = scaled_dot_product_attention(query, key, value)
print(output.shape)