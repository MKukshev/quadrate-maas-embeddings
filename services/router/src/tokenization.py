from __future__ import annotations

from typing import Dict

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TokenizerProvider:
    """Provide and cache tokenizers by name."""

    def __init__(self) -> None:
        self._cache: Dict[str, PreTrainedTokenizerBase] = {}

    def get(self, name: str) -> PreTrainedTokenizerBase:
        if name not in self._cache:
            self._cache[name] = AutoTokenizer.from_pretrained(name)
        return self._cache[name]


def count_text_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Count tokens for a single text using the provided tokenizer."""

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_length=True,
        is_split_into_words=False,
    )
    length = encoded.get("length")
    if isinstance(length, list) and length:
        return int(length[0])
    if isinstance(length, int):
        return int(length)

    input_ids = encoded.get("input_ids") or []
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def truncate_text_to_tokens(tokenizer: PreTrainedTokenizerBase, text: str, max_tokens: int) -> str:
    """Truncate text to the given token budget and return the decoded string."""

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        return_attention_mask=False,
        return_token_type_ids=False,
        is_split_into_words=False,
    )
    input_ids = encoded.get("input_ids") or []
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return tokenizer.decode(input_ids, skip_special_tokens=True)
