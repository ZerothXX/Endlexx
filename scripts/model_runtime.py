"""Loopback-only inference service. Run with the isolated model environment.

The application owns scheduling; model loading occurs inside the shared GPU lock.
No changes or imports of Endless's private configuration are necessary.
"""

import argparse
import gc
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
ROOT = Path(__file__).resolve().parents[1]


class ContextBudgetExceeded(ValueError):
    pass


def fit_context(processor, messages, images, limit=6500, preserve_history=False):
    messages = [dict(message) for message in messages]
    dropped = 0
    while True:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images or None, return_tensors="pt")
        if inputs["input_ids"].shape[1] <= limit:
            return inputs, dropped
        if preserve_history:
            raise ContextBudgetExceeded("模型上下文超出预算，需要先压缩对话历史。")
        # Preserve system instructions and the current request, including its images.
        index = next((i for i, m in enumerate(messages[:-1]) if m["role"] != "system"), None)
        if index is None:
            raise ContextBudgetExceeded("当前需求与必要资料超过本机上下文预算，请缩短需求或精简角色档案。")
        messages.pop(index)
        dropped += 1
        while index < len(messages) - 1 and messages[index]["role"] == "assistant":
            messages.pop(index)
            dropped += 1


class Runtime:
    def __init__(self, args):
        self.args = args
        self.model = self.processor = self.embedder = self.embed_tokenizer = self.reranker = (
            self.rerank_tokenizer
        ) = None
        self.mutex = threading.RLock()
        self.last_error = None
        self.calls = 0

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.processor = AutoProcessor.from_pretrained(
            self.args.model,
            local_files_only=True,
            use_fast=False,
            min_pixels=128 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.args.model,
            local_files_only=True,
            quantization_config=config,
            device_map="cuda:0",
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        self.model.eval()

    def unload(self):
        import torch

        self.model = self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"unloaded": True}

    def generate(self, payload):
        import torch
        from filelock import FileLock
        from PIL import Image

        with FileLock(str(self.args.lock), timeout=1700), self.mutex:
            self.load()
            messages = [dict(message) for message in payload["messages"]]
            images = []
            for source in payload.get("images", [])[:3]:
                path = Path(source).resolve()
                if not path.is_relative_to(self.args.assets.resolve()):
                    raise ValueError("Image path is outside asset storage")
                image = Image.open(path).convert("RGB")
                image.thumbnail((896, 896))
                images.append(image)
            if images:
                messages[-1]["content"] = [{"type": "image", "image": image} for image in images] + [
                    {"type": "text", "text": messages[-1]["content"]}
                ]
            inputs, dropped = fit_context(
                self.processor, messages, images, preserve_history=payload.get("preserve_history", False)
            )
            input_tokens = inputs["input_ids"].shape[1]
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=min(int(payload.get("max_tokens", 700)), 1400),
                    do_sample=False,
                    use_cache=True,
                )
            result = self.processor.batch_decode(
                output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )[0].strip()
            self.calls += 1
            return {
                "text": result,
                "model": "Qwen3-VL-4B-Instruct",
                "quantization": "nf4",
                "input_tokens": input_tokens,
                "history_messages_dropped": dropped,
            }

    def embed(self, payload):
        import torch

        with self.mutex:
            if self.embedder is None:
                from transformers import AutoModel, AutoTokenizer

                self.embed_tokenizer = AutoTokenizer.from_pretrained(
                    self.args.embedding, local_files_only=True
                )
                self.embedder = AutoModel.from_pretrained(self.args.embedding, local_files_only=True).eval()
                torch.set_num_threads(min(6, os.cpu_count() or 4))
            vectors = []
            for start in range(0, len(payload["texts"]), 4):
                inputs = self.embed_tokenizer(
                    payload["texts"][start : start + 4],
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    # BGE-M3 dense retrieval uses normalized CLS pooling.
                    embedded = self.embedder(**inputs).last_hidden_state[:, 0]
                    embedded = torch.nn.functional.normalize(embedded, p=2, dim=1)
                vectors.extend(embedded.tolist())
            return {"vectors": vectors, "model": self.args.embedding_id}

    def rerank(self, payload):
        import torch

        with self.mutex:
            if self.reranker is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                    self.args.reranker, local_files_only=True
                )
                self.reranker = AutoModelForSequenceClassification.from_pretrained(
                    self.args.reranker, local_files_only=True
                ).eval()
            pairs = [[payload["query"], item] for item in payload["texts"]]
            tokens = self.rerank_tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            with torch.inference_mode():
                scores = self.reranker(**tokens).logits.view(-1).float().tolist()
            return {"scores": scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--embedding-id", default="BAAI/bge-m3")
    parser.add_argument("--reranker", required=True)
    parser.add_argument("--lock", type=Path, default=ROOT / "data/gpu.lock")
    parser.add_argument("--assets", type=Path, default=ROOT / "data/assets")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    runtime = Runtime(args)
    token = os.environ.get("ENDLEXX_MODEL_RUNTIME_TOKEN", "")
    if not token:
        raise SystemExit("ENDLEXX_MODEL_RUNTIME_TOKEN must be configured")

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data):
            value = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(value)))
            self.end_headers()
            self.wfile.write(value)

        def do_GET(self):
            self.respond(
                200,
                {
                    "ready": args.model.is_dir(),
                    "loaded": runtime.model is not None,
                    "calls": runtime.calls,
                    "embedding_ready": Path(args.embedding).is_dir(),
                    "reranker_ready": Path(args.reranker).is_dir(),
                    "last_error": runtime.last_error,
                },
            )

        def do_POST(self):
            if self.headers.get("X-Runtime-Token") != token:
                return self.respond(403, {"error": "Forbidden"})
            length = int(self.headers.get("Content-Length", 0))
            if length > 2_000_000:
                return self.respond(413, {"error": "Payload too large"})
            try:
                payload = json.loads(self.rfile.read(length))
                if self.path == "/unload":
                    with runtime.mutex:
                        value = runtime.unload()
                elif self.path == "/generate":
                    value = runtime.generate(payload)
                elif self.path == "/embed":
                    value = runtime.embed(payload)
                elif self.path == "/rerank":
                    value = runtime.rerank(payload)
                else:
                    return self.respond(404, {"error": "Unknown endpoint"})
                self.respond(200, value)
            except ContextBudgetExceeded as exc:
                self.respond(422, {"code": "context_budget", "error": str(exc)})
            except Exception as exc:
                runtime.last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                traceback.print_exc()
                self.respond(500, {"error": runtime.last_error})

        def log_message(self, *_):
            pass

    args.assets.mkdir(parents=True, exist_ok=True)
    print(f"Endlexx model runtime http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
