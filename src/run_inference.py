# run_inference.py
# This file is a fully functional flattened version for inference only,
# matching the reference Qwen2-2-7B-hf model output.
# It uses only the slow tokenizer implemented in tokenization_qwen2.py.

import argparse
import os
import torch
import torch.nn.functional as F
from torch import nn
from collections import namedtuple
from modeling.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
from tokenization.tokenization_qwen2 import Qwen2Tokenizer

import logging as py_logging
logger = py_logging.getLogger(__name__)
logger.setLevel(py_logging.INFO)
if not logger.handlers:
    logger.addHandler(py_logging.StreamHandler())

# Minimal output namedtuple (for GenerationMixin compatibility).
BaseModelOutputWithPast = namedtuple("BaseModelOutputWithPast", ["last_hidden_state", "past_key_values", "hidden_states", "attentions"])

def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen2 using the slow tokenizer")
    parser.add_argument("--model_path", default="model", type=str, required=True,
                        help="Path to the pretrained Qwen2 model (directory with config.json, vocab.json, and merges.txt)")
    parser.add_argument("--prompt", type=str,
                        default="<|begin▁of▁sentence|>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|User|>Alice has 20 quarters. She wants to exchange them for nickels and so she goes to the bank. After getting back from the bank, she discovers that 20% of the nickels are iron nickels worth $3 each. What is the total value of her money now?<|Assistant|>",
                        help="Input text prompt for generation")
    parser.add_argument("--max_length", type=int, default=250,
                        help="Maximum length (in tokens) of the generated sequence")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model configuration.
    config_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(config_path):
        config = Qwen2Config.from_json_file(config_path)
    else:
        config = Qwen2Config.from_pretrained(args.model_path)
    config.use_cache = False

    # Load the model.
    model = Qwen2ForCausalLM.from_pretrained(args.model_path, config=config).to(device)
    model.eval()

    # Use the slow tokenizer.
    vocab_file = os.path.join(args.model_path, "vocab.json")
    merges_file = os.path.join(args.model_path, "merges.txt")
    tokenizer = Qwen2Tokenizer(vocab_file, merges_file)

    # Tokenize the input prompt.
    tokens = tokenizer._tokenize(args.prompt)
    input_ids = [tokenizer._convert_token_to_id(token) for token in tokens]
    input_ids_tensor = torch.tensor([input_ids]).to(device)
    logger.info(f"Tokenized input_ids: {input_ids_tensor}")

    # Generate text using the autoregressive loop.
    with torch.no_grad():
        outputs = model.generate(input_ids_tensor, max_length=args.max_length)
        logger.info(f"Generated token ids: {outputs}")

    # Decode and print the generated text.
    generated_ids = outputs[0].tolist()
    decoded_text = tokenizer.decode(generated_ids)
    print("=== Generated Text ===")
    print(decoded_text)


if __name__ == "__main__":
    main()
