from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from einops import rearrange


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


def get_complex_rotary_matrix(head_dim: int, seq_len: int, theta: float = 10000.0):
    # theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # let k = 2(i-1) then expression -2(i-1)/dim for i = [1, 2, ... dim/2] equivalent to
    # -k/dim k=[-2(1-1), 2(2-1),2(3-1),...,2(dim/2 -1)]=[0,2,...,dim-2]
    # or in python will be range(0,dim,2).
    # Please recall that the range() function will only include dim-2 in the end of the series
    theta = 1.0 / 10000.0 ** (
        torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim
    )
    m = torch.arange(seq_len)
    angular_component = torch.outer(m, theta).float()
    radial_component = torch.ones_like(angular_component)
    freqs_complex = torch.polar(radial_component, angular_component)
    return freqs_complex


def apply_rotary_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_matrix_complex: torch.Tensor,
    device: str,
):

    query_complex = torch.view_as_complex(
        rearrange(
            query.float(),
            "B seq_len n_head (half_hdim two) -> B seq_len n_head half_hdim two",
            two=2,
        )
    )

    key_complex = torch.view_as_complex(
        rearrange(
            key.float(),
            "B seq_len n_head (half_hdim two) -> B seq_len n_head half_hdim two",
            two=2,
        )
    )

    rotary_matrix_complex = rearrange(
        rotary_matrix_complex, "seq_len half_hdim -> 1 seq_len 1 half_hdim"
    )
    query_rotated_complex = query_complex * rotary_matrix_complex
    key_rotated_complex = key_complex * rotary_matrix_complex
    query_rotated = torch.view_as_real(query_rotated_complex).reshape_as(query)
    key_rotated = torch.view_as_real(key_rotated_complex).reshape_as(key)
    return query_rotated.type_as(query).to(device), key_rotated.type_as(key).to(device)


def repeat_kv_head(keys: torch.Tensor, values: torch.Tensor, repeats: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_key = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )
        self.cache_value = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )

    def update_cache(self, key, value, start_pos):
        batch_size, seq_len, _, _ = key.shape
        self.cache_key[:batch_size, start_pos : start_pos + seq_len] = key
        self.cache_value[:batch_size, start_pos : start_pos + seq_len] = value
        updated_key = self.cache_key[:batch_size, : start_pos + seq_len]
        updated_value = self.cache_value[:batch_size, : start_pos + seq_len]
        return updated_key, updated_value

    def forward(
        self, x: torch.Tensor, start_pos: int, rotary_matrix_complex: torch.Tensor
    ):

        query = rearrange(
            self.wq(x),
            "B seq_len (n_head h_dim) -> B seq_len n_head h_dim",
            h_dim=self.head_dim,
        )
        key = rearrange(
            self.wk(x),
            "B seq_len (n_head h_dim) -> B seq_len n_head h_dim",
            h_dim=self.head_dim,
        )

        value = rearrange(
            self.wv(x),
            "B seq_len (n_head h_dim) -> B seq_len n_head h_dim",
            h_dim=self.head_dim,
        )
        query, key = apply_rotary_embeddings(
            query, key, rotary_matrix_complex, device=x.device
        )

        key, value = self.update_cache(key, value, start_pos)

        key, value = repeat_kv_head(key, value, self.n_rep)

        key = rearrange(key, "B seq_len n_head head_dim -> B n_head seq_len head_dim")
        value = rearrange(
            value, "B seq_len n_head head_dim -> B n_head seq_len head_dim"
        )
        query = rearrange(query, "B 1 n_head head_dim -> B n_head 1 head_dim")

        scores = torch.einsum("bhqd,bhkd -> bhqk", query, key) / math.sqrt(
            self.head_dim
        )
        scores = F.softmax(scores.float(), dim=-1).type_as(query)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.einsum("bhqk,bhkv -> bhqv", scores, value)

        output = rearrange(output, " B n_head 1 head_dim -> B 1 n_head head_dim")
        output = rearrange(output, "B 1 n_head head_dim -> B 1 (n_head head_dim)")

        return self.wo(output)  # (B, 1, Dim) -> (B, 1, Dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(args.n_layers)])

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.complex_rotary_matrix = get_complex_rotary_matrix(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2
        ).to(device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        complex_rotary_matrix = self.complex_rotary_matrix[
            start_pos : start_pos + seq_len
        ]
        for layer in self.layers:
            h = layer(h, start_pos, complex_rotary_matrix)
        h = self.norm(h)
        output = self.output(h).float()
        return output


class LLaMA:

    def __init__(
        self,
        checkpoints_dir: Path,
        tokenizer_path: Path,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):

        with open(checkpoints_dir/'params.json', "r") as f:
            self.args = ModelArgs(
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                **json.loads(f.read()),
            )
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.load(str(tokenizer_path))
        self.args.vocab_size = self.tokenizer.vocab_size()

        self.model = Transformer(self.args).to(device)

        state_dict = torch.load(checkpoints_dir/'consolidated.00.pth', map_location="cpu")
        del state_dict["rope.freqs"]
        self.model.load_state_dict(state_dict, strict=True)

    def text_completion(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        encoded_prompts = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]

        batch_size = len(encoded_prompts)
        assert (
            batch_size <= self.args.max_batch_size
        ), f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in encoded_prompts)

        assert (
            max_prompt_len <= self.args.max_seq_len
        ), f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full(
            (batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device
        )
        for i, encoded in enumerate(encoded_prompts):
            # Populate the initial tokens with the prompt tokens
            tokens[i, : len(encoded)] = torch.tensor(
                encoded, dtype=torch.long, device=self.args.device
            )

        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_tokens_mask = tokens != pad_id
        for cur_pos in range(1, total_len):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:

                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            next_token = torch.where(
                prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            if all(eos_reached):
                break

        token_list = []
        text_list = []
        for current_prompt_tokens in tokens.tolist():

            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            token_list.append(current_prompt_tokens)
            decoded_text = self.tokenizer.decode(current_prompt_tokens)
            text_list.append(decoded_text)
        return (token_list, text_list)

    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
