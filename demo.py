import torch
from dataset import encode, get_config, vocab
from model import TransformerDecoder
config = get_config("config.py")

model = TransformerDecoder().to(config.device)
model.load_state_dict(torch.load('latest.pt')['model_state_dict'])
sample = torch.tensor(encode("<question>:are you beyonce?", vocab=vocab), dtype=torch.long).to(config.device)
y = model.generate(sample.unsqueeze(dim=0), 50)