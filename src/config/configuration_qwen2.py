# configuration_qwen2.py
# This file is a flattened, self‐contained configuration file for the Qwen2 model.
# It includes only the minimal functions needed for inference of the Qwen2-7B math instruct model.

import json
import math
from typing import Any, Dict, Optional, Tuple

import torch

##############################
# Minimal Logging Definition #
##############################
class SimpleLogger:
    @staticmethod
    def get_logger(name: str):
        # A very simple logger that just prints messages.
        class Logger:
            def info(self, msg): print("[INFO]", msg)
            def warning(self, msg): print("[WARNING]", msg)
            def warning_once(self, msg): print("[WARNING]", msg)
            def error(self, msg): print("[ERROR]", msg)
        return Logger()

logger = SimpleLogger.get_logger(__name__)

##############################
# Minimal PretrainedConfig   #
##############################
class PretrainedConfig:
    """
    Minimal version of PretrainedConfig.
    This class stores configuration parameters and provides basic serialization.
    """
    def __init__(self, **kwargs):
        # Update instance dictionary with all keyword arguments.
        self.__dict__.update(kwargs)
        # Set default value for returning dict outputs.
        if "use_return_dict" not in kwargs:
            self.use_return_dict = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        return cls(**config_dict, **kwargs)

##############################
# Minimal RoPE Functions     #
##############################
# These functions are taken (and slightly simplified) from modeling_rope_utils.py.
# They are required to compute the rotary embeddings used in Qwen2.

def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation.
    """
    if config is None:
        raise ValueError("A configuration must be provided.")
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Not used here
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, attention_factor

def _compute_linear_scaling_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with linear scaling.
    """
    if config is None:
        raise ValueError("A configuration must be provided.")
    factor = config.rope_scaling["factor"] if config.rope_scaling is not None else 1.0
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)
    inv_freq = inv_freq / factor
    return inv_freq, attention_factor

def _compute_dynamic_ntk_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies with NTK scaling.
    """
    if config is None:
        raise ValueError("A configuration must be provided.")
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"] if config.rope_scaling is not None else 1.0
    attention_factor = 1.0
    seq_len = seq_len if seq_len is not None and seq_len > max_position_embeddings else max_position_embeddings
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, attention_factor

def _compute_yarn_parameters(
    config: PretrainedConfig, device: torch.device, seq_len: Optional[int] = None, **rope_kwargs
) -> Tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies using the Yarn method.
    (This implementation is minimal; details can be filled in later.)
    """
    if config is None:
        raise ValueError("A configuration must be provided.")
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    factor = config.rope_scaling["factor"] if config.rope_scaling is not None else 1.0
    attention_factor = config.rope_scaling.get("attention_factor", 0.1 * math.log(factor) + 1.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, attention_factor

def _compute_longrope_parameters(
    config: PretrainedConfig, device: torch.device, seq_len: Optional[int] = None, **rope_kwargs
) -> Tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies using LongRoPE scaling.
    (Minimal implementation; detailed behavior can be refined later.)
    """
    if config is None:
        raise ValueError("A configuration must be provided.")
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    # For simplicity, use default factors if not provided.
    ext_factors = torch.tensor([1.0] * (dim // 2), dtype=torch.float32, device=device)
    inv_freq = 1.0 / (ext_factors * (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)))
    attention_factor = config.rope_scaling.get("attention_factor", 1.0) if config.rope_scaling is not None else 1.0
    return inv_freq, attention_factor

def _compute_llama3_parameters(
    config: PretrainedConfig, device: torch.device, seq_len: Optional[int] = None, **rope_kwargs
) -> Tuple[torch.Tensor, float]:
    """
    Computes the inverse frequencies for llama3-style RoPE.
    (Minimal implementation.)
    """
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)
    factor = config.rope_scaling["factor"] if config.rope_scaling is not None else 1.0
    inv_freq_llama = inv_freq / factor  # Minimal placeholder implementation
    return inv_freq_llama, attention_factor

# Map rope_type to our minimal implementations.
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}

# Minimal validation function – in our flattened version we simply print warnings.
def rope_config_validation(config: PretrainedConfig, ignore_keys: Optional[set] = None):
    if getattr(config, "rope_scaling", None) is None:
        return
    # For our purposes, ensure there is a rope type specified.
    rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
    if rope_type not in ROPE_INIT_FUNCTIONS:
        logger.warning(f"Unrecognized rope_type: {rope_type}. Defaulting to 'default'.")
        config.rope_scaling["rope_type"] = "default"

##############################
# Qwen2Config Implementation #
##############################
class Qwen2Config(PretrainedConfig):
    """
    Qwen2 configuration class for inference.
    
    Only the parameters necessary for inference are kept.
    """
    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # Default parallel plans (placeholders)
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

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = False,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 28,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        # Set primary parameters
        self.vocab_size = vocab_size
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
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling  # Should be a dict if provided; otherwise, user must supply one
        self.attention_dropout = attention_dropout

        # Validate and normalize rope scaling settings.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            # Rename 'type' to 'rope_type' for consistency.
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        # Call the validation (minimal version)
        rope_config_validation(self)

        # Update with any additional keyword arguments.
        self.__dict__.update(kwargs)

##############################
# Optional: from_dict Method #
##############################
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "Qwen2Config":
        return cls(**config_dict, **kwargs)

    @classmethod
    def from_json_file(cls, json_file: str) -> "Qwen2Config":
        with open(json_file, "r", encoding="utf-8") as reader:
            config_dict = json.load(reader)
        return cls.from_dict(config_dict)

##############################
# End of configuration_qwen2.py
##############################
