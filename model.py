import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import cos, sin

def get_frequency(i, d):
    return 10000**(-2*(i-1)/d)
    
def get_rotary_embedding(cfg, embd_dim):
    d = embd_dim//2
    R = np.zeros([cfg.block_size,embd_dim,embd_dim])
      
    for m in range(cfg.block_size):
          for i in range(d):
            theta_i = get_frequency(i,d)
            rot_2d = np.array([[cos(m*theta_i), -sin(m*theta_i)],[sin(m*theta_i),cos(m*theta_i)]])
            R[m,2*i:2*i+2,2*i:2*i+2] = rot_2d
    return torch.tensor(R,device=cfg.device).to(torch.float)
    
class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key_proj = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query_proj = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value_proj = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("rotary_matrix",get_rotary_embedding(cfg, head_size))
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query_proj(x), self.key_proj(x), self.value_proj(x)  # B,T, head_size
        q = torch.bmm(q.transpose(0,1),self.rotary_matrix[:T]).transpose(0,1)
        k = torch.bmm(k.transpose(0,1),self.rotary_matrix[:T]).transpose(0,1)
        d_k = k.shape[-1]
        h = q @ k.transpose(-2, -1)  # B, T, T
        h = h.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        A = F.softmax(h * d_k ** -0.5, dim=-1)
        o = A @ v
        return o


class MultiheadAttention(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.block_size = cfg.block_size
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)
        self.mha = MultiheadAttention(cfg, head_size=head_size)

    def forward(self, x):
        x = self.mha(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()

        self.text_embd = nn.Embedding(vocab_size, cfg.n_embd)
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.out = nn.Linear(cfg.n_embd, vocab_size)
        self.ln = nn.LayerNorm(cfg.n_embd)

    def forward(self, encoded_text, target=None):
        B, T = encoded_text.shape
        text_embd = self.text_embd(encoded_text)
        x = text_embd 
        x = self.transformer_layers(x)
        x = self.ln(x)
        logits = self.out(x)

        if target != None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
            return logits, loss
        return logits

    def generate(self, x, max_new_token):
        B, T = x.shape
        generated = []
        for i in range(max_new_token):
            x = x[:, -self.block_size:]
            y = self(x)
            pred_char = y[:, -1, :]

            pred_char_prob = F.softmax(pred_char, dim=-1)
            sampled_char = torch.multinomial(pred_char_prob, 1)
            generated.append(sampled_char)
            x = torch.cat([x, sampled_char], dim=1)
        return torch.tensor(generated, dtype=torch.long)
