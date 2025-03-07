"""Standalone Qwen2 Tokenizer

This file combines the functionality previously split between decoder.py and tokenization_qwen2.py.
It implements a byte-level BPE tokenizer for Qwen2 with full functionality and no changes in behavior.
"""

import json
import os
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple, List, Union
import regex as re  # Use the 'regex' module for Unicode property escapes.
import numpy as np
import logging

# Setup logging.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def to_py_obj(obj):
    """
    Convert a TensorFlow/PyTorch tensor, NumPy array, or list to a pure Python object.
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
    Returns a mapping from byte values (0-255) to unicode strings.
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
    Return set of adjacent symbol pairs in a word (represented as a tuple of symbols).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def clean_up_tokenization(out_string: str) -> str:
    """
    Clean up tokenization artifacts by removing extra spaces before punctuation.
    """
    out_string = (out_string.replace(" .", ".")
                             .replace(" ?", "?")
                             .replace(" !", "!")
                             .replace(" ,", ",")
                             .replace(" ' ", "'")
                             .replace(" n't", "n't")
                             .replace(" 'm", "'m")
                             .replace(" 's", "'s")
                             .replace(" 've", "'ve")
                             .replace(" 're", "'re"))
    return out_string


class AddedToken(str):
    """
    A minimal token wrapper that subclasses str, so that the token can be used as a regular string
    while carrying additional attributes.
    """
    def __new__(cls, content, lstrip=False, rstrip=False, special=False, normalized=False):
        obj = str.__new__(cls, content)
        obj.lstrip = lstrip
        obj.rstrip = rstrip
        obj.special = special
        obj.normalized = normalized
        return obj


class PreTrainedTokenizer:
    """
    Minimal base class for tokenizers.
    """
    def __init__(self, **kwargs):
        self.added_tokens_encoder = {}
        self._pad_token = kwargs.get("pad_token", None)
        self._bos_token = kwargs.get("bos_token", None)
        self._eos_token = kwargs.get("eos_token", None)
        self._unk_token = kwargs.get("unk_token", None)
        self.clean_up_tokenization_spaces = kwargs.get("clean_up_tokenization_spaces", False)
        self.split_special_tokens = kwargs.get("split_special_tokens", False)

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False, **kwargs):
        tokens = [self._convert_id_to_token(id) for id in token_ids]
        return " ".join(tokens)

    def _convert_id_to_token(self, index):
        raise NotImplementedError("Subclasses should implement _convert_id_to_token")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class Qwen2Tokenizer(PreTrainedTokenizer):
    """
    Qwen2 Tokenizer based on byte-level BPE.
    Reads vocabulary (vocab.json) and merge rules (merges.txt) during initialization.
    """
    VOCAB_FILES_NAMES = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }
    PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    def __init__(self,
                 vocab_file: str,
                 merges_file: str,
                 errors: str = "replace",
                 unk_token: str = "<|endoftext|>",
                 bos_token: Optional[str] = None,
                 eos_token: str = "<|endoftext|>",
                 pad_token: str = "<|endoftext|>",
                 clean_up_tokenization_spaces: bool = False,
                 split_special_tokens: bool = False,
                 **kwargs):
        # Wrap tokens if provided as plain strings.
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

        # Load the vocabulary.
        with open(vocab_file, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Build byte-to-unicode mappings.
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Process merges.txt to build BPE merge ranks.
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
        if kwargs.get("add_prefix_space", False):
            logger.warning("`add_prefix_space` is not supported; setting it to True has no effect.")
        super().__init__(errors=errors,
                         bos_token=bos_token,
                         eos_token=eos_token,
                         pad_token=pad_token,
                         unk_token=unk_token,
                         clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                         split_special_tokens=split_special_tokens,
                         **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

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

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string using the pre-tokenization regex and BPE."""
        bpe_tokens = []
        for token in re.findall(self.pretokenize_regex(), text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.bpe(token).split(" "))
        return bpe_tokens

    def pretokenize_regex(self):
        return self.PRETOKENIZE_REGEX

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token (str) to its corresponding id."""
        return self.encoder.get(token, self.encoder.get(str(self.unk_token)))

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an id (int) to its corresponding token (str)."""
        return self.decoder.get(index, str(self.unk_token))

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a sequence of tokens to a single string.
        """
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def convert_ids_to_tokens(self, token_ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """
        Convert a token id or list of token ids to tokens.
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

    def decode(self, token_ids: Union[int, List[int]], skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: Optional[bool] = False, **kwargs) -> str:
        """
        Convert token ids to a decoded string.
        """
        token_ids = to_py_obj(token_ids)
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        text = self.convert_tokens_to_string(tokens)
        if clean_up_tokenization_spaces:
            text = clean_up_tokenization(text)
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        """
        Save the vocabulary and merges files to the specified directory.
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
                    logger.warning("Saving vocabulary: BPE merge indices are not consecutive.")
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


if __name__ == "__main__":
    # Test case.
    vocab_path = "model/vocab.json"
    merges_path = "model/merges.txt"
    
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        logger.error("vocab.json and/or merges.txt not found in the model directory.")
    else:
        tokenizer = Qwen2Tokenizer(vocab_path, merges_path)
        sample_token_ids = [27, 91, 7265, 10417, 223, 1055, 10417, 223, 51889,
                            91, 29, 2610, 525, 1207, 16948, 11, 3465, 553,
                            54364, 14817, 13, 1446, 525, 264, 10950, 17847, 15757,
                            91, 1474, 91, 29, 3838, 374, 279, 2629, 315,
                            220, 16, 323, 220, 17, 75414, 91, 71703, 91,
                            29, 1249, 1477, 279, 2629, 315]
        decoded_text = tokenizer.decode(sample_token_ids)
        print("Decoded text:", decoded_text)
