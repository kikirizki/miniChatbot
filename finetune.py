from pathlib import Path
import torch
import torch.nn as nn
import json
import torch.nn.functional as F

from lora_mistral import LoraMistral
from dataset import SquadDataset

eval_interval = 10
block_size = 32
batch_size = 8
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
print("="*32)  
for name, param in mistralTransformer.named_parameters():
    if param.requires_grad : print(name)     


dataset = SquadDataset(tokenizer_path, block_size, batch_size)
x_data, y_data = dataset.get_batch("train")
optimizer = torch.optim.AdamW(mistralTransformer.parameters(),0.00001)

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
        # logprob = F.softmax(logits,dim=-1)
        # logprob = logprob.detach()
        # logprob = torch.argmax(logprob, dim=-1) 
        # logprob = logprob.cpu().numpy()
        # print(logprob)
        # print(mistral.tokenizer.decode(logprob.tolist()))

        # if iter > 100:
        #     break

        # break
    #     torch.save(
    #         {
    #             "model_state_dict": mistralTransformer.state_dict(),
    #             "optimizer_state_dict": optimizer.state_dict(),
    #             "loss": loss,
    #             "iteration": iter,
    #         },
    #         "latest.pt",
    #     )
