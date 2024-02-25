import torch
import torch.nn as nn
import torch.nn.functional as F

class LoraLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_alpha: float = 1.0,
        lora_rank=int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.A = nn.parameter(torch.randn(in_features, lora_rank))
        self.B = nn.parameter(torch.zeros(lora_rank, out_features))
        self.scale = lora_alpha/lora_rank
        self.weight.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        delta = torch.matmul(self.A*self.scale, self.B)
        return super().forward(input) + F.linear(input, delta)
    


