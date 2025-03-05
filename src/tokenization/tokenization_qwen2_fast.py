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
"""Tokenization classes for Qwen2 (fast version)."""

from typing import Optional, Tuple

# Here we assume that a fast tokenizer base class is available.
# For this flat file, we provide a minimal stub for PreTrainedTokenizerFast.
class PreTrainedTokenizerFast:
    def __init__(self, **kwargs):
        self.vocab_file = kwargs.get("vocab_file")
        self.merges_file = kwargs.get("merges_file")
        self.tokenizer_file = kwargs.get("tokenizer_file")
        self.unk_token = kwargs.get("unk_token")
        self.bos_token = kwargs.get("bos_token")
        self.eos_token = kwargs.get("eos_token")
        self.pad_token = kwargs.get("pad_token")
    def save_vocabulary(self, save_directory: str, name: Optional[str] = None):
        # Minimal placeholder: in a real scenario this would save the fast tokenizer’s model files.
        return [os.path.join(save_directory, "tokenizer.json")]
        
import os
from ...tokenization_utils import AddedToken
from ...utils import logging
from .tokenization_qwen2 import Qwen2Tokenizer

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}

class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Qwen2 tokenizer (backed by HuggingFace's tokenizers library) based on byte-level BPE.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = Qwen2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
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
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
