import yaml
import torch
from dataclasses import dataclass

@dataclass
class ModelArgs:
    def __init__(self, batch_size, block_size, max_iters, eval_interval, learning_rate, eval_iters, n_embd, n_head, n_layer, dropout, save_interval, checkpoint_name, task):
        self.batch_size = batch_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = float(learning_rate)
        self.eval_iters = eval_iters
        self.save_interval = save_interval
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_name = checkpoint_name
        self.task = task

def get_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = ModelArgs(**config_dict)
    return config