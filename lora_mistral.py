import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from mistral import Attention, ModelArgs

class LoraMistralAttention(Attention):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.query_A = nn.Parameter(torch.randn(self.wq.weight.shape[0], args.r))
        self.query_B = nn.Parameter(torch.zeros(args.r, self.wq.weight.shape[1]))
        self.key_A = nn.Parameter(torch.randn(self.wk.weight.shape[0], args.r))
        self.key_B = nn.Parameter(torch.zeros(args.r, self.wq.weight.shape[1]))
        self.scaling_factor = args.lora_alpha/math.sqrt(args.r)

    def lora_wq(self, x):
        delta = torch.matmul(self.query_A*self.scaling_factor, self.query_B)
        return self.wq(x) + F.linear(x, delta) 

    def lora_wk(self, x):
        delta = torch.matmul(self.key_A*self.scaling_factor, self.key_B)
        return self.wk(x) + F.linear(x, delta)

    def forward(self, x):
        query = rearrange(
            self.lora_wq(x),
            "B seq_len (n_head h_dim) -> B seq_len n_head h_dim",
            h_dim=self.args.head_dim,
        )
        key = rearrange(
            self.lora_wk(x),
            "B seq_len (n_head h_dim) -> B seq_len n_head h_dim",
            h_dim=self.args.head_dim,
        )
        value = rearrange(
            self.wv(x),
            "B seq_len (n_head h_dim) -> B seq_len n_head h_dim",
            h_dim=self.args.head_dim,
        )