from typing import OrderedDict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_separate_mask(num_keypoints, num_joints, device="cpu"):
    total_len = num_keypoints + num_joints + 2
    mask = torch.zeros((total_len, total_len), device=device)
    mask[:num_keypoints, :num_keypoints] = 1
    mask[num_keypoints : num_keypoints+num_joints, num_keypoints : num_keypoints+num_joints] = 1
    mask[-2, -2] = 1 
    mask[-1, -1] = 1 
    return mask

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):
    """
    embed_dim: output dimension for each position
    grid_sizes: the grids sizes in each dimension (K,).
    out: (grid_sizes[0], ..., grid_sizes[K-1], D)
    """
    num_sizes = len(grid_sizes)
    # For grid size of 1, we do not need to add any positional embedding
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    emb = np.zeros(grid_sizes + (embed_dim,))
    # Uniformly divide the embedding dimension for each grid size
    dim_for_each_grid = embed_dim // num_valid_sizes
    # To make it even
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        emb[..., valid_size_idx * dim_for_each_grid:(valid_size_idx + 1) * dim_for_each_grid] += \
            get_1d_sincos_pos_embed_from_grid(dim_for_each_grid, pos).reshape(posemb_shape)
        valid_size_idx += 1
    return emb


def get_multimodal_cond_pos_embed(embed_dim, mm_cond_lens: OrderedDict, 
                                  embed_modality=True):
    """
    Generate position embeddings for multimodal conditions. 
    
    mm_cond_lens: an OrderedDict containing 
        (modality name, modality token length) pairs.
        For `"image"` modality, the value can be a multi-dimensional tuple.
        If the length < 0, it means there is no position embedding for the modality or grid.
    embed_modality: whether to embed the modality information. Default is True.
    """
    num_modalities = len(mm_cond_lens)
    modality_pos_embed = np.zeros((num_modalities, embed_dim))
    if embed_modality:
        # Get embeddings for various modalites
        # We put it in the first half
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, torch.arange(num_modalities))
        modality_pos_embed[:, :embed_dim // 2] = modality_sincos_embed
        # The second half is for position embeddings
        pos_embed_dim = embed_dim // 2
    else:
        # The whole embedding is for position embeddings
        pos_embed_dim = embed_dim
    
    # Get embeddings for positions inside each modality
    c_pos_emb = np.zeros((0, embed_dim))
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(
                pos_embed_dim, embed_grid_sizes)
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim,))
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(
                pos_embed_dim, torch.arange(cond_len if cond_len > 0 else 1))
            cond_pos_embed = np.zeros((abs(cond_len), embed_dim))
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed
        cond_pos_embed += modality_pos_embed[idx]
        c_pos_emb = np.concatenate([c_pos_emb, cond_pos_embed], axis=0)
    
    return c_pos_emb

class DepthHead(nn.Module):
    def __init__(self):
        super(DepthHead, self).__init__()
        self.conv_1 = self._make_conv_block(4, 64, 4, 2, 3)
        self.conv_2 = self._make_conv_block(64, 128, 4, 2, 1)
        self.conv_3 = self._make_conv_block(128, 256, 4, 2, 1)
        self.conv_4 = self._make_conv_block(256, 512, 4, 2, 1)
        self.conv_5 = self._make_conv_block(512, 1024, 4, 2, 1)
        self.conv_6 = self._make_conv_block(1024, 2048, 4, 2, 1)
        self.max_pooling = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

    def _make_conv_block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),  
            nn.LeakyReLU() 
        )
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.max_pooling(x)
        return x
    
class EmbedMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_blocks=3, dropout=0.1):
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LeakyReLU())
        
        for _ in range(num_blocks-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
        
class SegmentedLN(nn.Module):

    def __init__(self, num_keypoints, num_joints, embed_dim) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        part1 = self.ln1(x[:, :self.num_keypoints])
        part2 = self.ln2(x[:, self.num_keypoints : self.num_keypoints + self.num_joints])
        # part3 = x[:, self.num_keypoints + self.num_joints :]  # 保留剩余部分不变
        return torch.cat([part1, part2], dim=1)
        

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 合并QKV的线性投影
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, N, D = x.shape
        # 合并生成QKV [B, N, 3*D]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)  # 各[B, N, D]
        
        # 分头处理 [B, H, N, D/H]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim) ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, v)  # [B, H, N, D/H]
        output = output.transpose(1, 2).contiguous().view(B, N, D)  # 合并多头
        return self.out_proj(output)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2  #& dim = 256, half = 128
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        ) #& shape: [128]
        args = t[:, None].float() * freqs[None]   #& t[:, None].float() * freqs[None] : (B, 1) * (1, 128)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_keypoints = kwargs.get("num_keypoints", None)
        self.num_joints = kwargs.get("num_joints", None)

        self.attention = MultiHeadAttention(self.embed_dim, self.num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, 2 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.Dropout(dropout)
        )
        # Post-LayerNorm
        if self.num_keypoints is not None and self.num_joints is not None:
            #^ num_keypoints 和 num_joints 两部分分开进行 LN！因为一个是 keypoint, 一个是 joint。
            self.norm_ln1 = SegmentedLN(self.num_keypoints, self.num_joints, self.embed_dim)
            self.norm_ln2 = SegmentedLN(self.num_keypoints, self.num_joints, self.embed_dim)
        else:
            print(f"Num of keypoints or num of joints is None, use default LayerNorm!")
            self.norm_ln1 = nn.LayerNorm(self.embed_dim)
            self.norm_ln2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x, mask=None):
        # MHA部分 (Post-Norm)
        attn_output = self.attention(x, mask)
        x = x + attn_output
        x = self.norm_ln1(x)
        
        # FFN部分 (Post-Norm)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm_ln2(x)
        return x

if __name__ == "__main__":

    num_keypoints = 7
    num_joints = 8
    block = TransformerBlock(embed_dim=384, num_heads=8, num_keypoints=num_keypoints, num_joints=num_joints)
    mask = create_separate_mask(num_keypoints, num_joints, device="cpu")
    x = torch.randn(2, num_keypoints+num_joints+2, 384)
    out = block(x, mask)
    print(out.shape)