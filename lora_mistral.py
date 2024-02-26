from pathlib import Path
from typing import List
import torch
import json
from mistral import ModelArgs, Attention, EncoderBlock, MistralTransformer
from lora import LoraLinear
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

class LoraMistral(Mistral):
    def __init__(self, max_batch_size: int, device: str):
        super().__init__(max_batch_size, device)

    def from_pretrained(self, checkpoints_dir, tokenizer_path, lora_path):
        with open(checkpoints_dir / "params.json", "r") as f:
            self.args = ModelArgs(
                **json.loads(f.read()),
            )
        if self.device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        self.args.device = self.device
        self.args.max_batch_size = self.max_batch_size
        self.tokenizer.load(str(tokenizer_path))
        self.args.vocab_size = self.tokenizer.vocab_size()
        self.model = LoraMistralTransformer(self.args).to(device=self.device)
        state_dict = torch.load(checkpoints_dir / "consolidated.00.pth")
        self.model.load_state_dict(state_dict, strict=False)
