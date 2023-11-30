import torch
from dataset import encode, get_config, vocab, decode
from model import TransformerDecoder
import re

config = get_config("config.py")

def filter_generated_response(text):
    pattern = r'<answer>:(.*?)(?=\n<)'
    match = re.search(pattern, text)

    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    elif '<answer>:' in text and '\n<' in text:
        # If the pattern is not found but the markers are present, filter the text
        filtered_text = re.sub(r'<answer>:', '', text)
        filtered_text = filtered_text.split('\n<')[0].strip()
        return filtered_text
    else:
        raise AssertionError("Pattern not found.")

model = TransformerDecoder().to(config.device)
model.load_state_dict(torch.load('latest.pt')['model_state_dict'])
sample = torch.tensor(encode("<question>:are you beyonce?", vocab=vocab), dtype=torch.long).to(config.device)
y = model.generate(sample.unsqueeze(dim=0), 50)
y = y.squeeze().cpu().numpy()
generated_text = decode(y, vocab=vocab)
answer = filter_generated_response(generated_text)
