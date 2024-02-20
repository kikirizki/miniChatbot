import torch
import fire
from model import LLaMA
from pathlib import Path

def chat(checkpoints_dir:str, tokenizer_path:str):
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    model = LLaMA(
        checkpoints_dir=Path(checkpoints_dir),
        tokenizer_path=Path(tokenizer_path),
        max_seq_len=1024,
        max_batch_size=1,
        device=device,
    )

    while True:
        message = input("user: ")
        prompt_with_template = [
            f"<s>[INST] <<SYS>>you are a young scientist<</SYS>>[/INST]</s><s>[INST] {{{message}}} [/INST]"
        ]
        input_length = len(prompt_with_template[0])
        _, completed_texts = model.text_completion(prompt_with_template, max_gen_len=64)
        print(f"Bot:{completed_texts[0][input_length:]}")

if __name__ == "__main__":
    fire.Fire(chat)
    