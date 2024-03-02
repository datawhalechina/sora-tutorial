import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        assert config.hidden_dim % config.num_heads == 0
        
        self.wq = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.wk = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.wv = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.att_dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        q = q.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        k = k.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        v = v.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # (b, nh, ql, hd) @ (b, nh, hd, kl) => b, nh, ql, kl
        att = torch.matmul(q, k.transpose(2, 3))
        att /= math.sqrt(self.config.head_dim)
        score = F.softmax(att.float(), dim=-1)
        score = self.att_dropout(score)
        
        # (b, nh, ql, kl) @ (b, nh, kl, hd) => b, nh, ql, hd
        attv = torch.matmul(score, v)
        attv = attv.view(batch_size, seq_len, -1)
        return score, attv