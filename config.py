import yaml
import torch

class Config:
    def __init__(self, batch_size, block_size, max_iters, eval_interval, learning_rate, eval_iters, n_embd, n_head, n_layer, dropout, save_interval):
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

def get_config(config_path):
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config