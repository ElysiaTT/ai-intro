import numpy as np
import d2l
import torch
import torch.nn as nn
class Qwen2RMSNorm(nn.Module):  # 标准化层
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))#初始化
        self.variance_epsilon = eps#一个很小的浮点数值，用于防止除零操作，通常称为epsilon
    def forward(self, hidden_states):
        input_dtype=hidden_states.dtype
        hidden_states-hidden_states.to(torch.float32)
        variance=hidden_states.pow(2).mean(-1,keepdim=True)#计算输入张量在最后一个维度（特征维度）上的方差。具体来说，先对每个元素平方（.pow(2)），然后在最后一个维度上求平均（.mean(-1, keepdim=True)）。keepdim=True保持维度数目不变，方便后续广播操作。
        hidden_states=hidden_states*torch.rsqrt(variance+self.variance_epsilon)
        return self.weight*hidden_states.to(input_dtype)