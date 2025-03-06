import os
import requests
from tqdm import tqdm

# Hugging Face model repo
HF_REPO = "https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct/resolve/main/"

# List of essential files
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json"
]

# Destination directory
DEST_DIR = "model"
os.makedirs(DEST_DIR, exist_ok=True)

def download_file(url, dest_path):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            bar.update(len(chunk))

def main():
    for file in MODEL_FILES:
        url = HF_REPO + file
        dest_path = os.path.join(DEST_DIR, file)
        
        if os.path.exists(dest_path):
            print(f"Skipping {file}, already exists.")
            continue
        
        print(f"Downloading {file}...")
        try:
            download_file(url, dest_path)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {file}: {e}")

if __name__ == "__main__":
    main()
