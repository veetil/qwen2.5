# Qwen 2.5 Minimal Implementation

## Overview

This repository provides a stripped-down, minimal implementation of Qwen 2/2.5 intended for learning, testing, and research & development. It is based on the Hugging Face Transformers implementation but has been simplified to make it easier to understand, experiment with, and modify. Note that this implementation is not optimized for speed or performance.

## Why This Project?

- **Minimal Implementation:**  
  The codebase is minimal and clear, ideal for experimentation and rapid prototyping.

- **Learning & Research Focus:**  
  Designed purely for learning, research, and testing new ideas, rather than for production use.

- **Accessible Alternative:**  
  Qwen 2.5 lacks accessible open-source implementations. This repository offers a simpler alternative compared to the official, more complex transformers integration.

## Current Features

- **Basic Tokenization and Inference:**  
  - A merged, slow tokenizer implementation in `tokenization_qwen2.py` that combines all functionalities.
  - Autoregressive text generation via a minimal inference script in `run_inference.py`.

- **Tested on Qwen 2.5 7B Math Instruct:**  
  For input sequences smaller than 500 tokens, the model has been tested with Qwen 2.5 7B Math Instruct and produces expected results.

## Features Not Yet Implemented

- **Speed and Performance Optimizations:**  
  This implementation is not optimized for any speed or performance. Future work may include:
  - Implementing caching mechanisms.
  - Optimizations for memory and computational efficiency.
  - Faster tokenization methods.

- **Sliding Window Attention:**  
  Sliding window attention is not implemented. As a result, for input sequences longer than 4096 tokens, the model's behavior may deviate significantly from expected results.

## Repository Structure

This minimal implementation consists of just two main files:

- **`tokenization_qwen2.py`**  
  Contains the complete, slow tokenizer implementation (merged from the previous `decoder.py` and `tokenization_qwen2.py` files). It implements byte-level Byte-Pair-Encoding (BPE) tokenization.

- **`run_inference.py`**  
  A script to load the pretrained Qwen 2.5 model, tokenize an input prompt, generate text via autoregressive inference, and decode the generated token IDs using the slow tokenizer.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/qwen2.5-minimal.git
cd qwen2.5-minimal
pip install -r requirements.txt
