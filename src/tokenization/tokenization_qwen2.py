# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for Qwen2."""

import json
import os
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple

import regex as re

# Minimal implementations of required base classes:

class AddedToken(str):
    def __new__(cls, content, lstrip=False, rstrip=False, special=False, normalized=False):
        obj = str.__new__(cls, content)
        obj.lstrip = lstrip
        obj.rstrip = rstrip
        obj.special = special
        obj.normalized = normalized
        return obj

class PreTrainedTokenizer:
    def __init__(self, **kwargs):
        self.added_tokens_encoder = {}
        self._pad_token = kwargs.get("pad_token", None)
        self._bos_token = kwargs.get("bos_token", None)
        self._eos_token = kwargs.get("eos_token", None)
        self._unk_token = kwargs.get("unk_token", None)
        self.clean_up_tokenization_spaces = kwargs.get("clean_up_tokenization_spaces", False)
        self.split_special_tokens = kwargs.get("split_special_tokens", False)
    
    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False, **kwargs):
        # Minimal implementation: join tokens
        tokens = [self._convert_id_to_token(id) for id in token_ids]
        return " ".join(tokens)
    
    def _convert_id_to_token(self, index):
        raise NotImplementedError("Subclasses should implement _convert_id_to_token")
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

# Minimal logging
class Logger:
    def warning_once(self, message):
        print("Warning:", message)
    def info(self, message):
        print("Info:", message)

def get_logger(name):
    return Logger()

logger = get_logger(__name__)

# Constants
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

@lru_cache()
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Qwen2Tokenizer(PreTrainedTokenizer):
    """
    Construct a Qwen2 tokenizer based on byte-level Byte-Pair-Encoding.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        bos_token = (AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
                     if isinstance(bos_token, str) else bos_token)
        eos_token = (AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
                     if isinstance(eos_token, str) else eos_token)
        unk_token = (AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
                     if isinstance(unk_token, str) else unk_token)
        pad_token = (AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
                     if isinstance(pad_token, str) else pad_token)

        with open(vocab_file, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = []
        with open(merges_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if (i == 0 and line.startswith("#version:")) or not line:
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.pat = re.compile(PRETOKENIZE_REGEX)
        if kwargs.get("add_prefix_space", False):
            logger.warning_once(f"{self.__class__.__name__} does not support `add_prefix_space`; setting it to True has no effect.")
        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self._unk_token))

    def _convert_id_to_token(self, index):
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    # def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False, spaces_between_special_tokens=False, **kwargs):
    #     return super().decode(token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces, spaces_between_special_tokens=spaces_between_special_tokens, **kwargs)

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False, spaces_between_special_tokens=False, **kwargs):
        # Convert each token id to its token string; replace None with an empty string.
        for i in token_ids:
            print(i,'->',self._convert_id_to_token(int(i)))
        tokens = [self._convert_id_to_token(int(token_id)) for token_id in token_ids]

        tokens = [token if token is not None else "" for token in tokens]
        return " ".join(tokens)




    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"])
        merge_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"])
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive. Please check that the tokenizer is not corrupted!")
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1
        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, **kwargs):
        text = unicodedata.normalize("NFC", text)
        return (text, kwargs)
