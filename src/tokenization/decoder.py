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
"""Standalone Qwen2 Tokenizer with full functionality and perfect match with reference.
   This version reads vocab.json and merges.txt from the 'model' directory.
"""

import json
import os
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple, List, Union
import regex as re  # Use the 'regex' module to support Unicode property escapes.
import numpy as np
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


##############################
# Helper Classes and Functions
##############################

class AddedToken:
    """
    Minimal implementation of AddedToken to wrap token strings.
    """
    def __init__(self, content: str, lstrip: bool = False, rstrip: bool = False, special: bool = False, normalized: bool = True):
        self.content = content
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.special = special
        self.normalized = normalized

    def __str__(self):
        return self.content

    def __repr__(self):
        return f"AddedToken({self.content})"


def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, dict):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.tolist()
    else:
        return obj


@lru_cache()
def bytes_to_unicode():
    """
    Returns a mapping from byte values to unicode strings.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple[str]) -> set:
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def clean_up_tokenization(out_string: str) -> str:
    """
    Clean up tokenization artifacts by removing spaces before punctuation.
    """
    out_string = (
        out_string.replace(" .", ".")
                  .replace(" ?", "?")
                  .replace(" !", "!")
                  .replace(" ,", ",")
                  .replace(" ' ", "'")
                  .replace(" n't", "n't")
                  .replace(" 'm", "'m")
                  .replace(" 's", "'s")
                  .replace(" 've", "'ve")
                  .replace(" 're", "'re")
    )
    return out_string


##############################
# Qwen2Tokenizer Class
##############################

class Qwen2TokenizerDecoder:
    """
    Standalone implementation of the Qwen2 tokenizer based on byte-level Byte-Pair-Encoding.
    It reads vocab.json and merges.txt during initialization.
    """
    VOCAB_FILES_NAMES = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }
    PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        errors: str = "replace",
        unk_token: str = "<|endoftext|>",
        bos_token: Optional[str] = None,
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        clean_up_tokenization_spaces: bool = False,
        split_special_tokens: bool = False,
        **kwargs,
    ):
        # Wrap tokens with AddedToken if provided as strings.
        if isinstance(bos_token, str) and bos_token is not None:
            bos_token = AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
        if isinstance(eos_token, str):
            eos_token = AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
        if isinstance(unk_token, str):
            unk_token = AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
        if isinstance(pad_token, str):
            pad_token = AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.split_special_tokens = split_special_tokens
        self.errors = errors

        # Load the vocabulary from vocab.json.
        with open(vocab_file, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Build byte-to-unicode mapping.
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Read and process merges.txt to build BPE merge ranks.
        bpe_merges = []
        with open(merges_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if (i == 0 and line.startswith("#version:")) or not line:
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        self.pat = re.compile(self.PRETOKENIZE_REGEX)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self):
        return self.encoder

    def bpe(self, token: str) -> str:
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
                if word[i] == first and i < len(word) - 1 and word[i+1] == second:
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

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string using the pre-tokenization regex and BPE."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token (str) to an id using the vocabulary."""
        return self.encoder.get(token, self.encoder.get(str(self.unk_token)))

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an id (int) to a token (str) using the reverse vocabulary."""
        return self.decoder.get(index, str(self.unk_token))

    def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """
        Convert a token id or a list of token ids to a list of tokens.
        If skip_special_tokens is True, tokens equal to unk, bos, eos, or pad are skipped.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        tokens = []
        special_tokens = {str(self.unk_token), str(self.eos_token), str(self.bos_token), str(self.pad_token)}
        for idx in token_ids:
            token = self._convert_id_to_token(idx)
            if skip_special_tokens and token in special_tokens:
                continue
            tokens.append(token)
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a sequence of tokens to a single string.
        Joins tokens then decodes the byte-level encoding using byte_decoder.
        """
        text = "".join(tokens)
        # Instead of ord(), use the character itself as key in byte_decoder.
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = False,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of token IDs to a string.
        This function replicates the exact functionality of the reference Qwen2Tokenizer.decode().

        Args:
            token_ids (int or List[int]): The token IDs to decode.
            skip_special_tokens (bool): Whether to remove special tokens (unk, bos, eos, pad).
            clean_up_tokenization_spaces (bool): If True, cleans up extra spaces using clean_up_tokenization.
            spaces_between_special_tokens (bool): Forced to False for Qwen2Tokenizer.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            str: The decoded text.
        """
        token_ids = to_py_obj(token_ids)
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        text = self.convert_tokens_to_string(tokens)
        if clean_up_tokenization_spaces:
            text = clean_up_tokenization(text)
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        """
        Save the vocabulary and BPE merge files to the specified directory.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + self.VOCAB_FILES_NAMES["vocab_file"])
        merge_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + self.VOCAB_FILES_NAMES["merges_file"])
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

    def prepare_for_tokenization(self, text: str, **kwargs) -> Tuple[str, dict]:
        """
        Normalize the text using NFC normalization.
        """
        text = unicodedata.normalize("NFC", text)
        return text, kwargs


##############################
# Test Case Using Real Files
##############################

if __name__ == "__main__":
    # Use actual vocab and merges files from the "model" directory.
    vocab_path = "model/vocab.json"
    merges_path = "model/merges.txt"
    
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        logger.error("vocab.json and/or merges.txt not found in the model directory.")
    else:
        # Initialize the tokenizer using the actual files.
        tokenizer = Qwen2TokenizerDecoder(vocab_path, merges_path)
        
        # A sample list of valid token ids.
        sample_token_ids = [    27,     91,   7265,  10417,    223,   1055,  10417,    223,  51889,
             91,     29,   2610,    525,   1207,  16948,     11,   3465,    553,
          54364,  14817,     13,   1446,    525,    264,  10950,  17847,  15757,
             91,   1474,     91,     29,   3838,    374,    279,   2629,    315,
            220,     16,    323,    220,     17,  75414,     91,  71703,     91,
             29 ,    1249,   1477,    279,   2629,    315    ] 

        # sample_token_ids = [
        #     27, 91, 7265, 10417, 223, 1055, 10417, 223, 51889, 91, 29, 2610, 525, 1207, 16948, 
        #     11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 15757, 91, 1474, 
        #     91, 29, 3838, 374, 279, 2629, 315, 220, 16, 323, 220, 17, 75414, 91, 71703, 91, 
        #     29, 1249, 1477, 279, 2629, 315, 323, 323, 220, 17, 11, 582, 1184, 311, 912, 1105, 
        #     3786, 13, 576, 5109, 525, 323, 220, 17, 13, 6771, 594, 912, 1105, 3019, 553, 3019, 
        #     382, 16, 13, 9645, 1495, 279, 5109, 25, 323, 220, 17, 624, 17, 13, 2691, 279, 
        #     5109, 3786, 25, 488, 220, 17, 382, 785, 2629, 315, 323, 220, 17, 374, 220, 18, 
        #     382, 54815, 11, 279, 4226, 374, 1124, 79075, 90, 18, 7810, 151645
        # ]
        
        decoded_text = tokenizer.decode(sample_token_ids)
        print("Decoded text:", decoded_text)
