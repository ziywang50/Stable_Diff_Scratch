import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))
    
    def forward(self, tokens):
        #(Batch_Size, Seq_Len) -> (Batch_Size, Seq_len, Dim)
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #(Batch_size, seq_len, dim)
        residue = x
        #(batch_size, seq_len, n_embd)
        x = self.layernorm_1(x)
        #(batch_size, seq_len, n_embd)
        x = self.attention(x, causal_mask = True)
        x += residue

        #FEEDFORWARD
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        #(batch_size, seq_len, 4 * n_embd)
        x = F.gelu(x)
        x = self.linear_2(x)
        #(batch_size, seq_len, n_embd)
        x += residue
        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
    
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        #(Batch_Size, Seq_Len) -> (Batch_Size, Seq_len, Dim)
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)
        return output