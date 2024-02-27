from pathlib import Path
import torch
import torch.nn as nn
import json
import torch.nn.functional as F

from lora_mistral import LoraMistral
from dataset import SquadDataset

save_checkpoint = True
eval_interval = 100
block_size = 32
batch_size = 4
device = "cuda"
tokenizer_path = "/home/ubuntu/mistral-7B-v0.1/tokenizer.model"

mistral = LoraMistral(max_batch_size=32, device=device)

mistral.from_pretrained(
    Path("/home/ubuntu/mistral-7B-v0.1/"),
    Path(tokenizer_path),
    Path("/home/ubuntu/lora"),
)
mistral.freeze_model_except_lora()
mistralTransformer = mistral.model

print("Only the following layers requires grad :")
print("=" * 32)
for name, param in mistralTransformer.named_parameters():
    if param.requires_grad:
        print(name)


dataset = SquadDataset(tokenizer_path, block_size, batch_size)
x_data, y_data = dataset.get_batch("train")
optimizer = torch.optim.AdamW(mistralTransformer.parameters(), 0.00001)

iter = 0
for x, y in zip(x_data, y_data):
    x = x.cuda()
    y = y.cuda()

    positions = torch.arange(0, x.shape[1])
    logits = mistralTransformer(x, positions)

    B, T, C = logits.shape

    logits = logits.view(B * T, C)
    y = y.view(B * T)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iter % eval_interval == 0:
        print(print("Losss ", loss.item()))
        
        if save_checkpoint:
            lora_state_dict = {name: param for name, param in mistralTransformer.state_dict().items() if "lora" in name}
            torch.save(lora_state_dict, f'lora_{iter}_{loss.item():.2f}.pth')
