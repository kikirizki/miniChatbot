import os
import torch
import argparse
from model import TransformerDecoder
from config import get_config
from dataset import get_batch

def train(config):
    x_data, y_data, vocab_size = get_batch("train", config)
    model = TransformerDecoder(config, vocab_size)
    m = model.to(config.device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)
    
    iter = 0

    # Specify the directory to save checkpoints
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for x, y in zip(x_data, y_data):
        logits, loss = m(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % config.eval_interval == 0:
            print("Loss: ", loss.item())

        if iter % config.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'{config.checkpoint_name}_{iter}.pt')
            torch.save({
                'model_state_dict': m.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'iteration': iter
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Delete older checkpoints, keeping only the last 5
            all_checkpoints = sorted(os.listdir(checkpoint_dir))
            checkpoints_to_keep = 5

            if len(all_checkpoints) > checkpoints_to_keep:
                checkpoints_to_delete = all_checkpoints[:-checkpoints_to_keep]

                for checkpoint in checkpoints_to_delete:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                    os.remove(checkpoint_path)
                    print(f"Deleted checkpoint: {checkpoint_path}")

        iter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', type=bool, default=False, help='Set --pretrain if you want to finetune to question answer dataset')
    args = parser.parse_args()
    
    config = get_config("finetune_config.yaml") if args.finetune else get_config("train_config.yaml")
    train(config)
        
