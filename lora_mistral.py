from pathlib import Path
from typing import List
import torch
import json
from mistral import ModelArgs, Attention, EncoderBlock, MistralTransformer
from lora import LoraLinear
from sentencepiece import SentencePieceProcessor
import torch.nn as nn
from mistral import Mistral


class LoraAttention(Attention):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.wq = LoraLinear(
            args.dim,
            args.n_heads * args.head_dim,
            bias=False,
            lora_alpha=0.8,
            lora_rank=32,
        )
        self.wk = LoraLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False,
            lora_alpha=0.8,
            lora_rank=32,
        )


class LoraEncoderBlock(EncoderBlock):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.attention = LoraAttention(args)


class LoraMistralTransformer(MistralTransformer):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.layers = torch.nn.ModuleList(
            [LoraEncoderBlock(args=args) for _ in range(args.n_layers)]
        )



