"""Download only inference weights/tokenizers; never copy existing Qwen or LaMa weights."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", default=None)
args = parser.parse_args()
for repo, name in [("BAAI/bge-m3", "bge-m3"), ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3")]:
    print(f"Preparing {repo}", flush=True)
    path = snapshot_download(
        repo_id=repo,
        local_dir=ROOT / "data/models" / name,
        endpoint=args.endpoint,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "pytorch_model.bin",
            "tokenizer.model",
            "sentencepiece.bpe.model",
            "modules.json",
            "1_Pooling/*",
            "README.md",
        ],
        ignore_patterns=["onnx/*", "openvino/*", "imgs/*"],
        max_workers=3,
    )
    print(path, flush=True)
