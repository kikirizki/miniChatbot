import numpy as np
import torch
import wget
import json
import os
from sentencepiece import SentencePieceProcessor


class SquadDataset:
    def __init__(
        self, tokenizer_path: str, block_size: int, batch_size: int
    ):
        self.block_size = block_size
        self.batch_size = batch_size
        self.tokenizer = SentencePieceProcessor(tokenizer_path)
        self.tokenizer.load(tokenizer_path)
        self.raw_string = self.get_raw_string()
        print("================ Dataset loaded ===============")
        _encoded_data = self.tokenizer.encode(self.raw_string)
        print("================ Dataset encoded ==============")
        split_index = int(0.9 * len(_encoded_data))
        self.train_data = _encoded_data[:split_index]
        self.val_data = _encoded_data[split_index:]

    def get_raw_string(self):
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
            
        else:
            print("File already exists.")

        # Read the JSON file
        with open(downloaded_file, "r") as f:
            data = json.load(f)

        # Now 'data' contains the parsed JSON data
        # You can access the content of the JSON file using dictionary-like syntax
        qa_pairs = []
        for i in data["data"]:
            for p in i["paragraphs"]:
                for i in p["qas"]:

                    if len(i["answers"]) > 0:
                        line = f"<question>:{i['question']}\n<answer>:{i['answers'][0]['text']}"
                        qa_pairs.append(line)
                        
        return "\n".join(qa_pairs)

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        data_len = len(data)
        n_batch = data_len // self.batch_size
        rnd_shape = (n_batch, self.batch_size)

        min_value, max_value = 0, data_len - self.block_size
        rnd_idx = np.random.randint(min_value, max_value, size=rnd_shape)

        x = [
            torch.tensor(
                [data[i : i + self.block_size] for i in batch_idx],
                dtype=torch.long,
            )
            for batch_idx in rnd_idx
        ]
        y = [
            torch.tensor(
                [
                    data[(i + 1) % data_len : (i + 1) % data_len + self.block_size]
                    for i in batch_idx
                ],
                dtype=torch.long,
            )
            for batch_idx in rnd_idx
        ]
        return x, y