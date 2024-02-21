import torch
import fire
from llama import LLaMA
from mistral import Mistral
from pathlib import Path

def chat(model_name: str, checkpoints_dir: str, tokenizer_path: str):
    torch.manual_seed(0)

    allow_cuda = True
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"
    
    if model_name not in ["mistral", "llama"]:
        print("Choose model either mistral or llama")
        return

    model_architecture = Mistral if model_name == "mistral" else LLaMA
    max_seq_len = 1024 if model_name == "llama" else None

    model = model_architecture(
        checkpoints_dir=Path(checkpoints_dir),
        tokenizer_path=Path(tokenizer_path),
        max_seq_len=max_seq_len,
        max_batch_size=1,
        device=device,
    )

    while True:
        message = input("user: ")
        if model_name == "mistral":
            prompt_with_template = [f"{message}"]
        else:
            prompt_with_template = [
                f"<s>[INST] <<SYS>>you are a young scientist<</SYS>>[/INST]</s><s>[INST] {{{message}}} [/INST]"
            ]

        input_length = len(prompt_with_template[0])
        _, completed_texts = model.text_completion(prompt_with_template, max_gen_len=64)
        print(f"Bot:{completed_texts[0][input_length:]}")

if __name__ == "__main__":
    fire.Fire(chat)
