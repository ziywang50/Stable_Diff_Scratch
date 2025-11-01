import torch
from torch import nn
from torch.nn import functional as F

import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias = in_proj_bias) #W_q, W_k, W_v combined
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias) # W_o, output projection
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    #Attention = softmax(QK^T/sqrt(d_head))V
    #head_i = Attention(QW^q_i, KW^k_i, VW^v_i)
    def forward(self, x:torch.Tensor, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interm_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        #(batch_size, sequence_length, dim) -> (batch_size, seq_len, dim*3) -> 3 tensors of the same shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)

        #batch_size, H, Seq_len, Seq_len
        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            #Mask where the upper triangular part(the ones above the diagonal) that is made of 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        #(Batch_size, H, Seq_len) @ (Batch_size, H, Seq_len, Dim/H) -> (Batch_size, H, Seq_len, Dim/h)

        output = weight @ v
        output = output.transpose(1, 2).reshape(input_shape)
        output = self.out_proj(output)

        return output
class CrossAttention(nn.Module):
    def __init__(self, n_heads:int, d_embed:int, d_cross:int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    def forward(self, x, y):
        #x: (latent) (Batch_Size, Seq_Len_Q, Dim_Q)
        #y: (context) (Batch_Size, Seq_Len_KV, Dim_KV
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interm_shape = (batch_size, -1, self.n_heads, self.d_head)
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)
        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim = -1)
        output = weight @ v
        output = output.transpose(1,2).contiguous().view(input_shape)
        output = self.out_proj(output)
        return output
