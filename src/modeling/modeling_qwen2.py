# qwen2_inference.py
# This file is a fully functional flattened version for inference only,
# matching the reference Qwen2-2-7B-hf model output.
# It inlines minimal implementations of required functions and utilities.

import math
import json
import os
import re
import unicodedata
from collections import namedtuple
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

# ------------------------------------------------------------------
# Minimal Logging and Utils
# ------------------------------------------------------------------

import logging as py_logging
logger = py_logging.getLogger(__name__)
logger.setLevel(py_logging.INFO)  # Set to DEBUG for detailed logs
if not logger.handlers:
    handler = py_logging.StreamHandler()
    formatter = py_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def is_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False

# Minimal no-op decorators for docstrings and deprecation.
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

# ------------------------------------------------------------------
# Minimal PretrainedConfig and PreTrainedModel classes
# ------------------------------------------------------------------

class PretrainedConfig:
    # Minimal configuration base; actual Qwen2Config will override fields.
    def __init__(self, **kwargs):
        logger.debug(f"Initializing PretrainedConfig with kwargs: {kwargs}")
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.use_return_dict = kwargs.pop("return_dict", True)
        self.torchscript = kwargs.pop("torchscript", False)
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        # Additional model-specific attributes should be provided in derived configs.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        logger.debug("Converting PretrainedConfig to dictionary")
        return self.__dict__.copy()

class PreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        logger.debug(f"Initialized PreTrainedModel with config: {config}")

    def post_init(self):
        logger.debug("Post initialization of PreTrainedModel")
        # Minimal post initialization.
        pass




class GenerationMixin:
    def generate(self, input_ids, max_length=50, **kwargs):
        """
        Greedy autoregressive decoding that uses only the new token in subsequent iterations
        and avoids reapplying the lm_head if the forward pass already outputs logits.
        
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
        iteration = 0
        logger.info(f"Starting generation. Initial sequence length: {generated.shape[1]}")
        
        with torch.no_grad():
            while generated.shape[1] < max_length:
                if not self.config.use_cache:
                    past_key_values = None  # force it off, just to be safe


                iteration += 1
                logger.debug(f"Iteration {iteration}: generated sequence shape: {generated.shape}")
                # On first iteration, pass full sequence; afterwards, pass only the last token.
                # current_input = generated if not self.config.use_cache else generated[:, -1:]
                
                if self.config.use_cache:
                    # If caching is enabled, pass the full sequence on the first iteration,
                    # then only the last token thereafter.
                    current_input = generated if past_key_values is None else generated[:, -1:]
                else:
                    # When caching is disabled, always pass the full sequence.
                    current_input = generated


                outputs = self.forward(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=self.config.use_cache,  # Use the configuration flag
                    return_dict=True,
                    **kwargs
                )
                
                # The forward pass returns a structured output with last_hidden_state and past_key_values.
                last_hidden_state = outputs.last_hidden_state

                if not self.config.use_cache:
                    past_key_values = None
                else:
                    past_key_values = outputs.past_key_values


                # past_key_values = outputs.past_key_values
                
                # Check if the output is already logits (vocab dimension) or raw hidden states.
                if last_hidden_state.size(-1) == self.config.vocab_size:
                    logits = last_hidden_state
                else:
                    raise ValueError(f"Unexpected output shape: {last_hidden_state.shape}")
                    # logits = self.lm_head(last_hidden_state)
                
                logger.debug(f"[Generation] Iteration {iteration}: Logits shape: {logits.shape}")
                next_token_logits = logits[:, -1, :]
                logger.debug(f"[Generation] Iteration {iteration}: Next token logits shape: {next_token_logits.shape}")
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                logger.debug(f"[Generation] Iteration {iteration}: Next token: {next_token.squeeze().tolist()}")
                
                if hasattr(self.config, "eos_token_id") and (next_token == self.config.eos_token_id).all():
                    logger.info(f"EOS token encountered at iteration {iteration}. Stopping generation.")
                    generated = torch.cat([generated, next_token], dim=1)
                    break
                
                generated = torch.cat([generated, next_token], dim=1)
                logger.debug(f"[Generation] Iteration {iteration}: updated generated shape: {generated.shape}")
            
            logger.info(f"Finished generation. Final sequence length: {generated.shape[1]}")
        return generated




# ------------------------------------------------------------------
# Minimal Cache implementations
# ------------------------------------------------------------------

class Cache:
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        logger.debug(f"Updating cache for layer {layer_idx}")
        # Minimal: no caching, simply return the states.
        return key_states, value_states

class DynamicCache(Cache):
    def __init__(self):
        self.key_cache = {}
        self.value_cache = {}
        self._seen_tokens = {}
        logger.debug("Initialized DynamicCache")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        logger.debug(f"Updating DynamicCache for layer {layer_idx}")
        # For minimal inference, simply concatenate along sequence dimension.
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
        logger.debug(f"Getting sequence length for layer {layer_idx}")
        return self._seen_tokens.get(layer_idx, 0)

# ------------------------------------------------------------------
# Minimal AttentionMaskConverter
# ------------------------------------------------------------------

class AttentionMaskConverter:
    @staticmethod
    def _ignore_causal_mask_sdpa(attention_mask, inputs_embeds, past_key_values_length, sliding_window=None, is_training=False):
        # Minimal: if attention_mask is all ones, return True.
        if attention_mask is None:
            return True
        return torch.all(attention_mask == 1)

    @staticmethod
    def _unmask_unattended(causal_mask, min_dtype):
        # Minimal: return mask unchanged.
        return causal_mask

# ------------------------------------------------------------------
# Minimal FlashAttentionKwargs and LossKwargs placeholders
# ------------------------------------------------------------------

FlashAttentionKwargs = dict
class LossKwargs:
    pass

# ------------------------------------------------------------------
# Minimal Modeling Outputs as NamedTuples
# ------------------------------------------------------------------

BaseModelOutputWithPast = namedtuple("BaseModelOutputWithPast", ["last_hidden_state", "past_key_values", "hidden_states", "attentions"])
CausalLMOutputWithPast = BaseModelOutputWithPast
QuestionAnsweringModelOutput = BaseModelOutputWithPast
SequenceClassifierOutputWithPast = BaseModelOutputWithPast
TokenClassifierOutput = BaseModelOutputWithPast

# ------------------------------------------------------------------
# Minimal Unpack for typing
# ------------------------------------------------------------------

Unpack = Any

# ------------------------------------------------------------------
# Minimal ACT2FN dictionary
# ------------------------------------------------------------------

ACT2FN = {
    "silu": lambda x: x * torch.sigmoid(x),
    "gelu": F.gelu,
}


# ------------------------------------------------------------------
# Additional Helper Functions for Attention
# ------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
    It converts hidden states from shape (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ------------------------------------------------------------------
# Modeling Rope Utils (from modeling_rope_utils.py)
# ------------------------------------------------------------------

def _compute_default_rope_parameters(config: Optional[PretrainedConfig] = None,
                                     device: Optional[torch.device] = None,
                                     seq_len: Optional[int] = None,
                                     **rope_kwargs) -> Tuple[torch.Tensor, float]:
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError("Unexpected arguments: **rope_kwargs and config are mutually exclusive")
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor

# For minimality, we set only the "default" rope type.

def _compute_linear_scaling_rope_parameters(config: Optional[PretrainedConfig] = None,
                                            device: Optional[torch.device] = None,
                                            seq_len: Optional[int] = None,
                                            **rope_kwargs) -> Tuple[torch.Tensor, float]:
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError("Unexpected arguments: **rope_kwargs and config are mutually exclusive")
    if len(rope_kwargs) > 0:
        factor = rope_kwargs["factor"]
    elif config is not None:
        factor = config.rope_scaling["factor"]
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)
    inv_freq /= factor
    return inv_freq, attention_factor

def _compute_dynamic_ntk_parameters(config: Optional[PretrainedConfig] = None,
                                    device: Optional[torch.device] = None,
                                    seq_len: Optional[int] = None,
                                    **rope_kwargs) -> Tuple[torch.Tensor, float]:
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError("Unexpected arguments: **rope_kwargs and config are mutually exclusive")
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
        max_position_embeddings = rope_kwargs["max_position_embeddings"]
        factor = rope_kwargs["factor"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        max_position_embeddings = config.max_position_embeddings
        factor = config.rope_scaling["factor"]
    attention_factor = 1.0
    seq_len = seq_len if seq_len is not None and seq_len > max_position_embeddings else max_position_embeddings
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor

def _compute_yarn_parameters(config: PretrainedConfig, device: torch.device, seq_len: Optional[int] = None, **rope_kwargs) -> Tuple[torch.Tensor, float]:
    if len(rope_kwargs) > 0:
        raise ValueError("Unexpected arguments: **rope_kwargs should be unset in _compute_yarn_parameters")
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"]
    attention_factor = config.rope_scaling.get("attention_factor", 0.1 * math.log(factor) + 1.0)
    beta_fast = config.rope_scaling.get("beta_fast", 32)
    beta_slow = config.rope_scaling.get("beta_slow", 1)
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)
    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)
    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings)
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).float().to(device)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_extrapolation_factor) + inv_freq_extrapolation * inv_freq_extrapolation_factor
    return inv_freq, attention_factor

def _compute_longrope_parameters(config: PretrainedConfig, device: torch.device, seq_len: Optional[int] = None, **rope_kwargs) -> Tuple[torch.Tensor, float]:
    if len(rope_kwargs) > 0:
        raise ValueError("Unexpected arguments: **rope_kwargs should be unset in _compute_longrope_parameters")
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    long_factor = config.rope_scaling["long_factor"]
    short_factor = config.rope_scaling["short_factor"]
    factor = config.rope_scaling.get("factor")
    attention_factor = config.rope_scaling.get("attention_factor")
    if hasattr(config, "original_max_position_embeddings"):
        original_max_position_embeddings = config.original_max_position_embeddings
        factor = config.max_position_embeddings / config.original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings
    if attention_factor is None:
        attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings)) if factor > 1.0 else 1.0
    ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device) if (seq_len and seq_len > original_max_position_embeddings) else torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)
    return inv_freq, attention_factor

def _compute_llama3_parameters(config: PretrainedConfig, device: torch.device, seq_len: Optional[int] = None, **rope_kwargs) -> Tuple[torch.Tensor, float]:
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)
    factor = config.rope_scaling["factor"]
    low_freq_factor = config.rope_scaling["low_freq_factor"]
    high_freq_factor = config.rope_scaling["high_freq_factor"]
    old_context_len = config.rope_scaling["original_max_position_embeddings"]
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > high_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama, attention_factor

ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}

def _check_received_keys(rope_type: str, received_keys: set, required_keys: set, optional_keys: Optional[set] = None, ignore_keys: Optional[set] = None):
    if "type" in received_keys:
        received_keys -= {"type"}
        required_keys.add("rope_type")
    if ignore_keys is not None:
        received_keys -= ignore_keys
    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in rope_scaling for rope_type='{rope_type}': {missing_keys}")
    if optional_keys is not None:
        unused_keys = received_keys - required_keys - optional_keys
    else:
        unused_keys = received_keys - required_keys
    if unused_keys:
        logger.warning(f"Unrecognized keys in rope_scaling for rope_type='{rope_type}': {unused_keys}")

def _validate_default_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
    required_keys = {"rope_type"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

def _validate_linear_scaling_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
    required_keys = {"rope_type", "factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)
    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"rope_scaling factor must be a float >= 1, got {factor}")

def _validate_dynamic_scaling_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
    required_keys = {"rope_type", "factor"}
    optional_keys = {"original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)
    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"rope_scaling factor must be a float >= 1, got {factor}")

def _validate_yarn_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
    required_keys = {"rope_type", "factor"}
    optional_keys = {"attention_factor", "beta_fast", "beta_slow"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)
    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"rope_scaling factor must be a float >= 1, got {factor}")
    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        logger.warning(f"rope_scaling attention_factor must be a float > 0, got {attention_factor}")
    beta_fast = rope_scaling.get("beta_fast")
    if beta_fast is not None and not isinstance(beta_fast, float):
        logger.warning(f"rope_scaling beta_fast must be a float, got {beta_fast}")
    beta_slow = rope_scaling.get("beta_slow")
    if beta_slow is not None and not isinstance(beta_slow, float):
        logger.warning(f"rope_scaling beta_slow must be a float, got {beta_slow}")
    if (beta_fast or 32) < (beta_slow or 1):
        logger.warning(f"rope_scaling beta_fast must be greater than beta_slow, got beta_fast={beta_fast}, beta_slow={beta_slow}")

def _validate_longrope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
    required_keys = {"rope_type", "short_factor", "long_factor"}
    optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    short_factor = rope_scaling.get("short_factor")
    if not isinstance(short_factor, list) or len(short_factor) != dim // 2:
        logger.warning(f"rope_scaling short_factor must be a list of length {dim//2}, got {short_factor}")
    long_factor = rope_scaling.get("long_factor")
    if not isinstance(long_factor, list) or len(long_factor) != dim // 2:
        logger.warning(f"rope_scaling long_factor must be a list of length {dim//2}, got {long_factor}")
    if hasattr(config, "original_max_position_embeddings"):
        logger.warning_once("Please use the factor field in rope_scaling instead of original_max_position_embeddings.")
    else:
        factor = rope_scaling.get("factor")
        if factor is None or not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"rope_scaling factor must be a float >= 1, got {factor}")
        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0.0):
            logger.warning(f"rope_scaling attention_factor must be a float > 0, got {attention_factor}")

def _validate_llama3_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
    required_keys = {"rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)
    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"rope_scaling factor must be a float >= 1, got {factor}")
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    if low_freq_factor is None or not isinstance(low_freq_factor, float):
        logger.warning(f"rope_scaling low_freq_factor must be a float, got {low_freq_factor}")
    if high_freq_factor is None or not isinstance(high_freq_factor, float):
        logger.warning(f"rope_scaling high_freq_factor must be a float, got {high_freq_factor}")
    if high_freq_factor <= low_freq_factor:
        logger.warning(f"rope_scaling high_freq_factor must be greater than low_freq_factor, got {high_freq_factor} and {low_freq_factor}")
    original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
    if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
        logger.warning(f"rope_scaling original_max_position_embeddings must be an integer, got {original_max_position_embeddings}")
    if original_max_position_embeddings >= config.max_position_embeddings:
        logger.warning(f"rope_scaling original_max_position_embeddings must be less than max_position_embeddings, got {original_max_position_embeddings} and {config.max_position_embeddings}")

ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_dynamic_scaling_rope_parameters,
    "yarn": _validate_yarn_parameters,
    "longrope": _validate_longrope_parameters,
    "llama3": _validate_llama3_parameters,
}

def rope_config_validation(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    validation_fn = ROPE_VALIDATION_FUNCTIONS.get(rope_type)
    if validation_fn is not None:
        validation_fn(config, ignore_keys=ignore_keys)
    else:
        logger.warning(f"Missing validation function mapping in ROPE_VALIDATION_FUNCTIONS for rope_type='{rope_type}'")

# ------------------------------------------------------------------
# Qwen2 Configuration (configuration_qwen2.py)
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# (Below: Implement Qwen2 Model classes based on reference)
# For brevity, we implement only key classes with minimal stubs.
# ------------------------------------------------------------------

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
        logger.debug(f"[Attention] q_proj shape: {query_states.shape}, k_proj shape: {key_states.shape}, v_proj shape: {value_states.shape}")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        logger.debug(f"[Attention] After rotary embeddings: query {query_states.shape}, key {key_states.shape}")
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, self.scaling,
            dropout=self.attention_dropout, **kwargs
        )
        logger.debug(f"[Attention] Raw attention output shape: {attn_output.shape}")
        out_shape = input_shape + (self.num_key_value_groups * self.head_dim,)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        logger.debug(f"[Attention] Reshaped attention output shape: {attn_output.shape}")
        attn_output = self.o_proj(attn_output)
        logger.debug(f"[Attention] After o_proj, attention output shape: {attn_output.shape}")
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
        logger.debug(f"[DecoderLayer] Input shape: {hidden_states.shape}")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        logger.debug(f"[DecoderLayer] After input_layernorm: {hidden_states.shape}")
        attn_out, attn_weights = self.self_attn(hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
        logger.debug(f"[DecoderLayer] Attention output shape: {attn_out.shape}")
        hidden_states = residual + attn_out
        logger.debug(f"[DecoderLayer] After residual addition (post-attn): {hidden_states.shape}")
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        logger.debug(f"[DecoderLayer] After post-attention layernorm: {hidden_states.shape}")
        hidden_states = self.mlp(hidden_states)
        logger.debug(f"[DecoderLayer] After MLP: {hidden_states.shape}")
        hidden_states = residual + hidden_states
        logger.debug(f"[DecoderLayer] Final output shape: {hidden_states.shape}")
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class MistralModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # For inference, we reuse Qwen2 decoder architecture.
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
        if past_key_values is None and use_cache:
            past_key_values = DynamicCache()

        if cache_position is None:
            if use_cache and past_key_values is not None:
                past_seen = past_key_values.get_seq_length()
            else:
                past_seen = 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)


        # if cache_position is None:
        #     past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        # causal_mask = None  # For minimal inference, assume causal mask is computed externally or not needed.
        # hidden_states = inputs_embeds
        # pos_emb = self.rotary_emb(hidden_states, position_ids)

        # If use_cache=False, build a causal mask that is [batch_size, 1, seq_len, seq_len]
        # so each token only attends up to itself.
        if not use_cache:
            # seq_len = inputs_embeds.shape[1]
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]
            # The standard “triu” style mask sets attention_mask[i,j] = -inf if j > i
            # so tokens cannot attend to “future” positions:
            causal_mask = torch.full(
                (1, 1, seq_len, seq_len),
                float("-inf"),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            # shape is now [1, 1, seq_len, seq_len]; expand to [batch_size, 1, seq_len, seq_len]
            causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)
        else:
            # keep same for caching
            causal_mask = None

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
    pass  # Inherits behavior from MistralModel

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

    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[Qwen2Config] = None, *args, **kwargs):
        # Minimal implementation of from_pretrained:
        if config is None:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                config = cls.config_class.from_json_file(config_path)
            else:
                raise ValueError("No configuration file found in the model path.")
        model = cls(config)
        from safetensors.torch import load_file
        state_dict = {}
        total_keys_in_files = 0

        for filename in os.listdir(model_path):
            if filename.endswith(".safetensors"):
                file_path = os.path.join(model_path, filename)
                file_state_dict = load_file(file_path)
                keys_in_file = list(file_state_dict.keys())
                num_keys_in_file = len(keys_in_file)
                logger.debug(f"Loaded {num_keys_in_file} keys from {filename}: {keys_in_file}")
                total_keys_in_files += num_keys_in_file
                state_dict.update(file_state_dict)
        logger.debug(f"Total keys in safetensors files: {total_keys_in_files}")

        # Get model's current state dict keys
        model_state = model.state_dict()
        total_model_keys = len(model_state.keys())
        logger.debug(f"Total keys in model state dict: {total_model_keys}")

        # Compute matched, missing, and unexpected keys
        pretrained_keys = set(state_dict.keys())
        model_keys = set(model_state.keys())
        matched_keys = pretrained_keys & model_keys
        missing_keys = model_keys - pretrained_keys
        unexpected_keys = pretrained_keys - model_keys
        
        logger.debug(f"Total matched keys: {len(matched_keys)}")
        logger.debug(f"Matched keys: {matched_keys}")
        logger.debug(f"Missing keys (in model but not in pretrained): {missing_keys}")
        logger.debug(f"Unexpected keys (in pretrained but not in model): {unexpected_keys}")
        if total_model_keys > 0:
            logger.debug(f"Percentage of keys matched: {len(matched_keys) / total_model_keys:.2%}")

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")
        return model



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

class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels if hasattr(config, "num_labels") else 2
        self.model = Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()
        logger.debug(f"Initialized Qwen2ForSequenceClassification with config: {config}")

    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None) -> BaseModelOutputWithPast:
        logger.debug("Forward pass of Qwen2ForSequenceClassification")
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                             use_cache=use_cache, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs.last_hidden_state
        logits = self.score(hidden_states)
        # Pool last non-pad token logits:
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            if self.config.pad_token_id is None:
                last_token = -1
            else:
                non_pad = (input_ids != self.config.pad_token_id).to(torch.int64)
                token_idx = torch.arange(input_ids.shape[-1], device=input_ids.device)
                last_token = (token_idx * non_pad).argmax(-1)
            pooled_logits = logits[torch.arange(batch_size), last_token]
        else:
            pooled_logits = logits[:, -1]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutputWithPast(last_hidden_state=pooled_logits, past_key_values=outputs.past_key_values,
                                                hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class Qwen2ForTokenClassification(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels if hasattr(config, "num_labels") else 2
        self.model = Qwen2Model(config)
        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", 0.1))
        self.score = nn.Linear(config.hidden_size, self.num_labels)
        self.post_init()

    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None) -> BaseModelOutputWithPast:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                             use_cache=use_cache, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(last_hidden_state=logits, past_key_values=outputs.past_key_values,
                                       hidden_states=outputs.hidden_states, attentions=outputs.attentions)

class Qwen2ForQuestionAnswering(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = Qwen2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                start_positions: Optional[torch.LongTensor] = None,
                end_positions: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None) -> BaseModelOutputWithPast:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                   past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                   output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        loss = None
        if start_positions is not None and end_positions is not None:
            loss = F.cross_entropy(start_logits.view(-1), start_positions.view(-1)) + F.cross_entropy(end_logits.view(-1), end_positions.view(-1))
        return QuestionAnsweringModelOutput(last_hidden_state=logits, past_key_values=outputs.past_key_values,
                                              hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# ------------------------------------------------------------------
# End of Qwen2 Inference Code
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage:
    # For actual inference, load a pretrained Qwen2Config and model weights.
    # Here, we create a dummy config for testing.
    dummy_rope_scaling = {
        "rope_type": "default",
    }
    config = Qwen2Config(vocab_size=1000, hidden_size=512, intermediate_size=2048,
                         num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=8,
                         max_position_embeddings=1024, rope_theta=10000.0, rope_scaling=dummy_rope_scaling)
    model = Qwen2ForCausalLM(config)
    # Create dummy input_ids
    input_ids = torch.randint(0, config.vocab_size, (1, 20))
    outputs = model(input_ids=input_ids)
    print("Logits shape:", outputs.last_hidden_state.shape)
