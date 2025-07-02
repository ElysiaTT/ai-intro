import d2l
import torch
# shapes: (batch_size, seq_len, num_heads, head_dim) 

value=torch.rand(1, 256, 4, 64)
key=torch.rand(1, 256, 4, 64)
query=torch.rand(1, 256, 8, 64)
num_head_groups = query.shape[2]//key.shape[2] 
print(num_head_groups) # each group is of size 4 since there are 4 kv_heads

# 为了提高效率，交换seq_len和num_heads维度，einops可以像下面这样简单地完成:
from einops import rearrange 

query = rearrange(query, "b n h d -> b h n d") 
key = rearrange(key, "b s h d -> b h s d") 
value = rearrange(value, "b s h d -> b h s d")


#然后就是需要在查询矩阵中引入”分组“的概念
# 也就是将query矩阵按照kv_heads进行分组，然后分别与key和value进行点积运算
# 最后将每个组的结果进行拼接，得到最终的输出
query= rearrange(query, "b (h g) n d -> b g h n d",g=num_head_groups )#之前我们计算的一组内多少个kv_heads

from einops import einsum, rearrange 
scores = einsum(query, key, "b g h n d, b h s d -> b h n s") 
print(scores.shape) # torch.Size([1, 4, 256, 256])


import torch.nn.functional as F 
scale = query.size(-1) ** 0.5 
attention = F.softmax(scores / scale, dim=-1) 
# here we do just a standard matrix multiplication 
out = einsum(attention, value, "b h n s, b h s d -> b h n d") 
# finally, just reshape back to the (batch_size, seq_len, num_kv_heads, hidden_dim) 
out = rearrange(out, "b h n d -> b n h d") 
print(out.shape) # torch.Size([1, 256, 4, 64])