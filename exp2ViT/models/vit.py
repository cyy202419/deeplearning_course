import torch
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads  # QKV的总维度
        project_out = not (heads == 1 and dim_head == dim)  # 是否需要输出投影
        
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子
        
        self.norm = nn.LayerNorm(dim)  # 输入归一化
        self.attend = nn.Softmax(dim=-1)  # 注意力分数归一化
        self.dropout = nn.Dropout(dropout)  # 注意力Dropout
        
        # QKV生成线性层
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # 输出投影层（条件判断）
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        
        # 生成QKV并分头
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 拆分为Q/K/V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        # 加权求和并合并多头
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        
        # 构建depth层的编码器堆叠
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(self, *, 
                image_size=224, 
                patch_size=16, 
                num_classes=10,
                dim=768, 
                depth=12, 
                heads=12, 
                mlp_dim=3072, 
                pool='cls',
                channels=3, 
                dim_head=64, 
                dropout=0,
                emb_dropout=0.):
        super().__init__()
        
        # 图像分块验证
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
               'Image dimensions must be divisible by the patch size.'
        
        # 计算块数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        # 池化方式验证
        assert pool in {'cls', 'mean'}, \
               'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # 块嵌入层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        
        # 位置编码和CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # 输出处理
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)  # [b, n, dim]
        b, n, _ = x.shape
        
        # 添加CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # [b, n+1, dim]
        
        # 添加位置编码
        #x += self.pos_embedding[:, :(n + 1)]
        x += self.pos_embedding 
        x = self.dropout(x)
        
        # Transformer处理
        x = self.transformer(x)
        
        # 池化方式选择
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        # 分类头
        x = self.to_latent(x)
        return self.mlp_head(x)