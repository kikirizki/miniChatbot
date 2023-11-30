from config import get_config
import numpy as np
import torch
import wget
import json
import os

config = get_config("config.yaml")


import os
import json
import wget

import os
import json
import wget
def get_raw_string():
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    downloaded_file = "train-v2.0.json"

    # Check if the file already exists
    if not os.path.exists(downloaded_file):
        # Define the download progress bar
        def bar_progress(current, total, width=80):
            progress = int(width * current / total)
            return "[" + "=" * progress + " " * (width - progress) + "]"

        # Download the file with a progress bar
        wget.download(url, downloaded_file, bar=bar_progress)
        print("File downloaded successfully.")
    else:
        print("File already exists.")

    # Read the JSON file
    with open(downloaded_file, 'r') as f:
        data = json.load(f)

    # Now 'data' contains the parsed JSON data
    # You can access the content of the JSON file using dictionary-like syntax
    qa_pairs = []
    for i in data['data']:
        for p in i['paragraphs']:
            for i in p['qas']:

                if len(i['answers']) > 0:
                    line = f"<question>:{i['question']}\n<answer>:{i['answers'][0]['text']}"
                    qa_pairs.append(line)
                    print(line)
    return "\n".join(qa_pairs)


def encode(text, vocab):
    return [vocab.index(c) for c in text]


def decode(idxs, vocab):
    return "".join([vocab[idx] for idx in idxs])


def get_batch(split):
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
    return x, y


raw_text = get_raw_string()
raw_text_length = len(raw_text)
n = int(0.9 * raw_text_length)
vocab = "".join(sorted(list(set(raw_text))))
vocab_size = len(vocab)

train_data = encode(raw_text[:n],vocab)
val_data = encode(raw_text[n:],vocab)



