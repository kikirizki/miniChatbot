import torch
import fire
from llama import LLaMA
from mistral import Mistral
from pathlib import Path


def chat(model_name: str, checkpoints_dir: str, tokenizer_path: str):
    torch.manual_seed(0)

    allow_cuda = True
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    if model_name == "mistral":
        model = Mistral(
            checkpoints_dir=Path(checkpoints_dir),
            tokenizer_path=Path(tokenizer_path),
            max_batch_size=1,
            device=device,
        )
        while True:
            message = input("user: ")
            prompt_with_template = [
                f"{message}",
            ]
            input_length = len(prompt_with_template[0])
            _, completed_texts = model.text_completion(prompt_with_template, 35)
            print(f"Bot:{completed_texts[0][input_length:]}")
    elif model_name == "llama":
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
            _, completed_texts = model.text_completion(
                prompt_with_template, max_gen_len=64
            )
            print(f"Bot:{completed_texts[0][input_length:]}")
    else:
        print("Choose model either mistral or llama")


if __name__ == "__main__":
    fire.Fire(chat)
