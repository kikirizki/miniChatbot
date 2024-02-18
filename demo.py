import torch
from model import LLaMA


if __name__ == '__main__':
    import argparse
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    
    parser = argparse.ArgumentParser(description="Argument parser for model paths")
    parser.add_argument("--checkpoints_path", type=str, help="Path to the checkpoints file")
    parser.add_argument("--parameter_path", type=str, help="Path to the parameters file")
    parser.add_argument("--tokenizer_path", type=str, help="Path to the tokenizer file")

    args = parser.parse_args()

    model = LLaMA(
        checkpoints_path=args.checkpoints_path,
        parameter_path=args.parameter_path,
        tokenizer_path=args.tokenizer_path,
        load_model=True,
        max_seq_len=1024,
        max_batch_size=1,
        device=device
    )

    while True:
        message = input("user: ")
        prompt_with_template = [f"<s>[INST] <<SYS>>you are a young scientist<</SYS>>[/INST]</s><s>[INST] {{{message}}} [/INST]"]
        input_length = len(prompt_with_template[0])
        _, completed_texts = model.text_completion(prompt_with_template, max_gen_len=64)
        print(f"Bot:{completed_texts[0][input_length:]}")
    