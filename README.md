# Qwen 2.5 Minimal Implementation

## Overview

This repository provides a stripped-down, minimal implementation of Qwen 2.5 intended for learning, testing, and research & development. It is based on the Hugging Face transformers repo but has been simplified. Note that this implementation is not optimized for speed or performance.

## Why This Project?

- **Minimal Implementation:**  
  The codebase is minimal, ideal for experimentation and rapid prototyping.

- **Learning & Research Focus:**  
  Designed purely for learning, research, and testing new ideas, rather than for production use.

- **Accessible Alternative:**  
  Qwen 2.5 lacks accessible open-source implementations. This repository offers a simpler alternative.

## Current Features
  - Autoregressive text generation via a minimal inference script in `run_inference.py`.

- **Tested on Qwen 2.5 7B Math Instruct:**  
  For sequences smaller than 500 tokens, the model has been tested with Qwen 2.5 7B Math Instruct and produces expected results.

## Features Not Yet Implemented

- **Speed and Performance Optimizations:**  
  This implementation is not optimized for speed or performance. Future work may include:
  - Implementing caching mechanisms.
  - Optimizations for memory and computational efficiency.
  - Faster tokenization methods.

- **Sliding Window Attention:**  
  Sliding window attention is not implemented. As a result, for long sequences, the model's behavior may deviate significantly from expected results.

## Repository Structure

This minimal implementation consists of the following main files:

- **`run_inference.py`**  
  A script to load the pretrained Qwen 2.5 model, tokenize an input prompt, generate text via autoregressive inference, and decode the generated token IDs using the slow tokenizer.

- **`download_model.py`**  
  A script to download the Qwen 2.5 7B Math Instruct model from Hugging Face into the `model/` directory.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/veetil/qwen2.5.git
cd qwen2.5
pip install -r requirements.txt
python src/run_inference.py --model_path model --prompt "<|begin▁of▁sentence|>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|User|>What is the sum of 1 and 2 ?<|Assistant|>" --max_length 250

