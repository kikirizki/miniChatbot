import numpy as np
import tiktoken
import torch
import os
import json
import wget

from config import get_config

# Constants
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
DOWNLOADED_FILE = "train-v2.0.json"

enc = tiktoken.get_encoding("cl100k_base")

vocab_size = 0

def download_file(url, downloaded_file):
    if not os.path.exists(downloaded_file):
        def bar_progress(current, total, width=80):
            progress = int(width * current / total)
            return "[" + "=" * progress + " " * (width - progress) + "]"

        wget.download(url, downloaded_file, bar=bar_progress)
        print("File downloaded successfully.")
    else:
        print("File already exists.")

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_qa_pairs():
    download_file(SQUAD_URL, DOWNLOADED_FILE)
    data = read_json_file(DOWNLOADED_FILE)

    qa_pairs = []
    for item in data['data']:
        for paragraph in item['paragraphs']:
            for question_answer in paragraph['qas']:
                if len(question_answer['answers']) > 0:
                    line = f"<question>:{question_answer['question']}\n<answer>:{question_answer['answers'][0]['text']}"
                    qa_pairs.append(line)
                    print(line)
    return "\n".join(qa_pairs)

def get_context():
    download_file(SQUAD_URL, DOWNLOADED_FILE)
    data = read_json_file(DOWNLOADED_FILE)

    contexts = []
    for item in data['data']:
        for paragraph in item['paragraphs']:
            contexts.append(paragraph['context'])
    return "\n".join(contexts)

def encode(text):
    return enc.encode(text)

def decode(idxs):
    return enc.decode(idxs)

def get_batch(split, config):
    train_data, val_data, vocab_size = get_dataset(config.task)
    data = train_data if split == "train" else val_data
    data_len = len(data)
    n_batch = data_len // config.batch_size
    rnd_shape = (n_batch, config.batch_size)

    min_value, max_value = 0, data_len - config.block_size
    rnd_idx = np.random.randint(min_value, max_value, size=rnd_shape)

    x = [torch.tensor([data[i:i + config.block_size] for i in batch_idx], dtype=torch.long, device=config.device) for
         batch_idx in rnd_idx]
    y = [torch.tensor([data[(i + 1) % data_len:(i + 1) % data_len + config.block_size] for i in batch_idx],
                      dtype=torch.long, device=config.device) for batch_idx in rnd_idx]
    return x, y, vocab_size

def get_dataset(dataset_type):
    raw_text = get_context() if dataset_type == "train" else get_qa_pairs()
    raw_text_length = len(raw_text)
    n = int(0.9 * raw_text_length)
    vocab_size = enc.n_vocab
    train_data = enc.encode(raw_text[:n])
    val_data = enc.encode(raw_text[n:])
    return train_data, val_data, vocab_size


