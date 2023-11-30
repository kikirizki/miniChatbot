import torch
from model import TransformerDecoder
from config import get_config
from dataset import get_batch

config = get_config("config.yaml")
model = TransformerDecoder()
m = model.to(config.device)

optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)
x_data, y_data = get_batch("train")
iter = 0
for x, y in zip(x_data, y_data):
    logits, loss = m(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter % config.eval_interval == 0:
        print(print("Losss ", loss.item()))
