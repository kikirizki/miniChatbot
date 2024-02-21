import torch
import fire
from llama import LLaMA
from mistral import Mistral
from pathlib import Path

def create_llama_template(message, system):
    return f"<s>[INST] <<SYS>>{system}<</SYS>>[/INST]</s><s>[INST] {{{message}}} [/INST]"

def chat(model_name: str, checkpoints_dir: str, tokenizer_path: str, allow_cuda: bool=True):

    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    if model_name == "mistral":
        model = Mistral(
            checkpoints_dir=Path(checkpoints_dir),
            tokenizer_path=Path(tokenizer_path),
            max_batch_size=1,
            device=device,
        )
    
    elif model_name == "llama":
        model = LLaMA(
            checkpoints_dir=Path(checkpoints_dir),
            tokenizer_path=Path(tokenizer_path),
            max_seq_len=1024,
            max_batch_size=1,
            device=device,
        )
    else:
        print("Please choose model architecture, either mistral or llama")  
        return

    while True:
        message = input("user: ")
        prompt_with_template = (
            [
               create_llama_template(message, "you are a young scientist") 
            ]
            if model == "llama"
            else [message]
        )
        input_length = len(prompt_with_template[0])
        _, completed_texts = model.text_completion(prompt_with_template, max_gen_len=64)
        print(f"Bot:{completed_texts[0][input_length:]}")

if __name__ == "__main__":
    fire.Fire(chat)
