import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import cos, sin
from config import get_config
from dataset import vocab_size

config = get_config("config.yaml")

def get_frequency(i, d):
    return 10000**(-2*(i-1)/d)
    
def get_rotary_embedding(token_length, embd_dim):
    d = embd_dim//2
    R = []
    for m in range(token_length):
        R_m = np.zeros([embd_dim,embd_dim])
        for i in range(d):
            theta_i = get_frequency(i,d)
            sub_mtrx = np.array([[cos(m*theta_i), -sin(m*theta_i)],[sin(m*theta_i),cos(m*theta_i)]])
            R_m[2*i:2*i+2,2*i:2*i+2]=sub_mtrx
        R.append(R_m)
    return torch.tensor(R,device=config.device)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key_proj = nn.Linear(config.n_embd, head_size, bias=False)
        self.query_proj = nn.Linear(config.n_embd, head_size, bias=False)
        self.value_proj = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("rotary_matrix",get_rotary_embedding(config.block_size, head_size))
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

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
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)
        self.mha = MultiheadAttention(n_head, head_size=head_size)

    def forward(self, x):
        x = self.mha(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_embd = nn.Embedding(vocab_size, config.n_embd)
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.out = nn.Linear(config.n_embd, vocab_size)
        self.ln = nn.LayerNorm(config.n_embd)

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
            x = x[:, -config.block_size:]
            y = self(x)
            pred_char = y[:, -1, :]

            pred_char_prob = F.softmax(pred_char, dim=-1)
            sampled_char = torch.multinomial(pred_char_prob, 1)
            generated.append(sampled_char)
            x = torch.cat([x, sampled_char], dim=1)
        return torch.tensor(generated, dtype=torch.long)
