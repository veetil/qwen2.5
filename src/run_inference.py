#!/usr/bin/env python
# coding=utf-8
# -----------------------------------------------------------------------------
# run_inference.py
#
# A simple script to run inference using Qwen2. This file loads a Qwen2ForCausalLM model
# and its fast tokenizer, then generates text from a given prompt using a minimal
# greedy autoregressive decoding loop.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# -----------------------------------------------------------------------------

import argparse
import os
import json
import math
import re
import unicodedata
from collections import namedtuple
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

# -----------------------------------------------------------------------------
# Minimal Logging
# -----------------------------------------------------------------------------
import logging as py_logging
logger = py_logging.getLogger(__name__)
logger.setLevel(py_logging.INFO)
if not logger.handlers:
    logger.addHandler(py_logging.StreamHandler())

# -----------------------------------------------------------------------------
# Minimal Utility Functions and No-Op Decorators
# -----------------------------------------------------------------------------
def is_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False

def add_start_docstrings(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def add_start_docstrings_to_model_forward(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def add_code_sample_docstrings(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def replace_return_docstrings(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def deprecate_kwarg(old_name, version, new_name=None):
    def decorator(func):
        return func
    return decorator

# -----------------------------------------------------------------------------
# Minimal PretrainedConfig and PreTrainedModel
# -----------------------------------------------------------------------------
class PretrainedConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if "use_return_dict" not in kwargs:
            self.use_return_dict = True
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()
    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        return cls(**config_dict, **kwargs)

class PreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def post_init(self):
        pass

# -----------------------------------------------------------------------------
# GenerationMixin with Greedy Autoregressive Decoding
# -----------------------------------------------------------------------------
class GenerationMixin:
    def generate(self, input_ids, max_length=50, **kwargs):
        """
        Greedy autoregressive decoding.
        
        Args:
            input_ids (torch.LongTensor): Initial input token IDs of shape (batch_size, seq_len)
            max_length (int): Maximum total length (in tokens) of the generated sequence.
            **kwargs: Additional keyword arguments passed to forward().
        
        Returns:
            torch.LongTensor: Generated token IDs.
        """
        self.eval()
        generated = input_ids
        past_key_values = None
        with torch.no_grad():
            while generated.shape[1] < max_length:
                outputs = self.forward(
                    input_ids=generated,
                    past_key_values=past_key_values,
                    use_cache=True,
                    **kwargs
                )
                # Compute logits from the last hidden state
                logits = self.lm_head(outputs.last_hidden_state)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                # If an EOS token is defined and produced for all sequences, stop decoding.
                if hasattr(self.config, "eos_token_id") and (next_token == self.config.eos_token_id).all():
                    generated = torch.cat([generated, next_token], dim=1)
                    break
                generated = torch.cat([generated, next_token], dim=1)
                past_key_values = outputs.past_key_values
        return generated

# -----------------------------------------------------------------------------
# Minimal Cache Implementations
# -----------------------------------------------------------------------------
class Cache:
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states

class DynamicCache(Cache):
    def __init__(self):
        self.key_cache = {}
        self.value_cache = {}
        self._seen_tokens = {}
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            self._seen_tokens[layer_idx] = key_states.shape[-2]
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            self._seen_tokens[layer_idx] += key_states.shape[-2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    def get_seq_length(self, layer_idx=0):
        return self._seen_tokens.get(layer_idx, 0)

# -----------------------------------------------------------------------------
# Minimal AttentionMaskConverter
# -----------------------------------------------------------------------------
class AttentionMaskConverter:
    @staticmethod
    def _ignore_causal_mask_sdpa(attention_mask, inputs_embeds, past_key_values_length, sliding_window=None, is_training=False):
        if attention_mask is None:
            return True
        return torch.all(attention_mask == 1)
    @staticmethod
    def _unmask_unattended(causal_mask, min_dtype):
        return causal_mask

# -----------------------------------------------------------------------------
# Placeholders for FlashAttentionKwargs and LossKwargs
# -----------------------------------------------------------------------------
FlashAttentionKwargs = dict
class LossKwargs:
    pass

# -----------------------------------------------------------------------------
# Minimal Modeling Outputs
# -----------------------------------------------------------------------------
BaseModelOutputWithPast = namedtuple("BaseModelOutputWithPast", ["last_hidden_state", "past_key_values", "hidden_states", "attentions"])
CausalLMOutputWithPast = BaseModelOutputWithPast
QuestionAnsweringModelOutput = BaseModelOutputWithPast
SequenceClassifierOutputWithPast = BaseModelOutputWithPast
TokenClassifierOutput = BaseModelOutputWithPast

# -----------------------------------------------------------------------------
# Minimal Unpack for typing
# -----------------------------------------------------------------------------
Unpack = Any

# -----------------------------------------------------------------------------
# Minimal ACT2FN
# -----------------------------------------------------------------------------
ACT2FN = {
    "silu": lambda x: x * torch.sigmoid(x),
    "gelu": F.gelu,
}

# -----------------------------------------------------------------------------
# Minimal Modeling Rope Utils (using functions from configuration)
# -----------------------------------------------------------------------------
# (Assume ROPE_INIT_FUNCTIONS is defined as in configuration; for brevity, we reuse the minimal versions below.)
def _compute_default_rope_parameters(config: Optional[PretrainedConfig] = None,
                                     device: Optional[torch.device] = None,
                                     seq_len: Optional[int] = None,
                                     **rope_kwargs) -> Tuple[torch.Tensor, float]:
    if config is None:
        raise ValueError("A configuration must be provided.")
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, attention_factor

# For minimality, we set only the "default" rope type.
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
}

def rope_config_validation(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    if getattr(config, "rope_scaling", None) is None:
        return
    rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
    if rope_type not in ROPE_INIT_FUNCTIONS:
        logger.warning(f"Unrecognized rope_type: {rope_type}. Defaulting to 'default'.")
        config.rope_scaling["rope_type"] = "default"

# -----------------------------------------------------------------------------
# Qwen2 Configuration
# -----------------------------------------------------------------------------
class Qwen2Config(PretrainedConfig):
    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    def __init__(self,
                 vocab_size: int = 151936,
                 hidden_size: int = 4096,
                 intermediate_size: int = 22016,
                 num_hidden_layers: int = 32,
                 num_attention_heads: int = 32,
                 num_key_value_heads: Optional[int] = 32,
                 hidden_act: Union[str, Callable] = "silu",
                 max_position_embeddings: int = 32768,
                 initializer_range: float = 0.02,
                 rms_norm_eps: float = 1e-6,
                 use_cache: bool = True,
                 tie_word_embeddings: bool = False,
                 rope_theta: float = 10000.0,
                 rope_scaling: Optional[Dict] = None,
                 use_sliding_window: bool = False,
                 sliding_window: int = 4096,
                 max_window_layers: int = 28,
                 attention_dropout: float = 0.0,
                 **kwargs):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        super().__init__(**kwargs)
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "Qwen2Config":
        return cls(**config_dict, **kwargs)
    @classmethod
    def from_json_file(cls, json_file: str) -> "Qwen2Config":
        with open(json_file, "r", encoding="utf-8") as reader:
            config_dict = json.load(reader)
        return cls.from_dict(config_dict)

# -----------------------------------------------------------------------------
# Minimal Model Components
# -----------------------------------------------------------------------------
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, self.scaling,
            dropout=self.attention_dropout, **kwargs
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen2MLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

class Qwen2Attention(LlamaAttention):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len
    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_sliding_window and getattr(config, "_attn_implementation", "eager") != "flash_attention_2":
            logger.warning("Sliding Window Attention is enabled but not implemented for this attn implementation.")
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, attn_weights = self.self_attn(hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

# -----------------------------------------------------------------------------
# Minimal Model Classes
# -----------------------------------------------------------------------------
class MistralModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()
    def post_init(self):
        pass
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None, **kwargs):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = None  # Minimal implementation; for inference we assume causal mask is not needed.
        hidden_states = inputs_embeds
        pos_emb = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = [] if output_hidden_states else None
        all_self_attns = [] if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            layer_out = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                              past_key_value=past_key_values, output_attentions=output_attentions,
                              use_cache=use_cache, cache_position=cache_position, position_embeddings=pos_emb, **kwargs)
            hidden_states = layer_out[0]
            if output_attentions:
                all_self_attns.append(layer_out[1])
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values,
                                       hidden_states=all_hidden_states, attentions=all_self_attns)

class Qwen2Model(MistralModel):
    pass

# -----------------------------------------------------------------------------
# Qwen2 PreTrained Model and Inference Classes with GenerationMixin
# -----------------------------------------------------------------------------
class Qwen2PreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = Qwen2Config
    base_model_prefix = "model"
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    def get_input_embeddings(self):
        return self.model.embed_tokens
    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs) -> BaseModelOutputWithPast:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                             output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                             return_dict=return_dict, cache_position=cache_position, **kwargs)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            return (loss, logits) if loss is not None else logits
        return CausalLMOutputWithPast(last_hidden_state=logits, past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# (Other classes for classification or QA are omitted for brevity)

# -----------------------------------------------------------------------------
# run_inference.py Main Script
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen2")
    parser.add_argument("--model_path", default="model", type=str, required=True,
                        help="Path or identifier of the pretrained Qwen2 model (e.g., a local folder or model hub id)")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Input text prompt for generation")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of the generated sequence")
    parser.add_argument("--use_fast_tokenizer", action="store_true",
                        help="If set, uses the fast tokenizer version; otherwise uses the slow tokenizer")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model configuration.
    # config = Qwen2Config.from_pretrained(args.model_path)
    config_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(config_path):
        config = Qwen2Config.from_json_file(config_path)
    else:
        config = Qwen2Config.from_pretrained(args.model_path)

    # Load the model. Qwen2ForCausalLM now has a generate() method from GenerationMixin.
    model = Qwen2ForCausalLM.from_pretrained(args.model_path, config=config).to(device)
    model.eval()

    # Load the fast tokenizer.
    from tokenization.tokenization_qwen2_fast import Qwen2TokenizerFast
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_path)

    # Tokenize the input prompt.
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate text using the autoregressive loop.
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=args.max_length)

    # Decode and print the generated text.
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== Generated Text ===")
    print(generated_text)

if __name__ == "__main__":
    main()
