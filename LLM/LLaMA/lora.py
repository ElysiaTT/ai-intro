import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def _init_(self, in_features,out_features,merge,rank=16,lora_alpha=16,dropout=0.5):
        super(LoRALinear,self)._init_()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dropout = dropout

        self.Linear = nn.Linear(in_features,out_features)
        if rank >0 :
            self.lora_b=nn.Parameter(torch.zeros(out_features,rank))
            self.lora_a = nn.Parameter(torch.randn(rank,in_features))
            self.scale = self.lora_alpha/self.rank
            self.linear.weight.requires_grad = False
        
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(dropout)
        else: 
            self.dropout = nn.Identity()
        
        self.initial_weight()

    def initial_weight(self):
        nn.init.kaiming_uniform_(self.Linear.weight)
        nn.init.zeros_(self.Linear.bias)
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_a)
            nn.init.zeros_(self.lora_b)

    def forward(self,x):
        if self.rank > 0 and self.merge >0:
            output = F.linear(x,self.Linear.weight+self.lora_b@self.lora_a*self.scale,self.Linear.bias)
            output = self.dropout(output)
            return output
        else:
            output = self.Linear(x)
            return self.dropout(output)