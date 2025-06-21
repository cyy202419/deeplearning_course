import torch
import torch.nn as nn
from einops import rearrange

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class WordEmbedding(BaseModel):
    def __init__(self, vocab_size, h_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, h_dim)
    
    def forward(self, x):
        return self.embedding(x.to(self.device))
    
class PositionEmbedding(BaseModel):
    def __init__(self, max_seq_len, h_dim):
        super().__init__()
        self.pe = torch.zeros(max_seq_len, h_dim).to(self.device)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, h_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / h_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
    
    def forward(self, x):
        return self.pe[:x.size(1), :].unsqueeze(0).repeat(x.size(0), 1, 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0):
        super().__init__()
        dim_head = dim // heads  # 每个头的维度
        inner_dim = dim_head * heads  # QKV的总维度
        project_out = not (heads == 1 and dim_head == dim)  # 是否需要输出投影
        
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子
        
        self.norm = nn.LayerNorm(dim)  # 输入归一化
        self.attend = nn.Softmax(dim=-1)  # 注意力分数归一化
        self.dropout = nn.Dropout(dropout)  # 注意力Dropout
        
        # QKV生成线性层
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        # 输出投影层（条件判断）
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v, mask=None):
        q = self.norm(q)
        k = self.norm(k)
        v = self.norm(v)
        

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))  # 重排为多头格式
        
        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            if mask.dim() == 2:  # (batch, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
            elif mask.dim() == 3:  # (batch, seq_len, seq_len)
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            elif mask.dim() == 4:  # (batch, heads, seq_len, seq_len)
                pass  # Already in correct format
            else:
                raise ValueError(f"Mask must have 2, 3 or 4 dimensions (got {mask.dim()})")
            
            dots = dots.masked_fill(mask == 0, float('-inf'))
            
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        # 加权求和并合并多头
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class PositionwiseFeedforwardLayer(BaseModel):
    def __init__(self, h_dim, pf_dim, dropout):
        super().__init__()
        # 定义两个全连接层
        self.fc_1 = nn.Linear(h_dim, pf_dim)  # 第一层：h_dim -> pf_dim
        self.fc_2 = nn.Linear(pf_dim, h_dim)  # 第二层：pf_dim -> h_dim
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # 第一层全连接 + ReLU激活
        inputs = torch.relu(self.fc_1(inputs))
        # Dropout
        inputs = self.dropout(inputs)
        # 第二层全连接
        inputs = self.fc_2(inputs)
        return inputs