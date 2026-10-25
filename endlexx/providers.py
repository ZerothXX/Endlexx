"""Three deployment profiles, with explicit network policy and capability reporting."""

import base64
import json
import re
from pathlib import Path

import httpx

from .config import ROOT, settings


class ModelContextExceeded(ValueError):
    pass


def parse_object(value: str) -> dict:
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.S).strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型没有返回有效的结构化结果")
    return json.loads(value[start : end + 1])


def runtime_call(path: str, payload: dict | None = None, timeout: int | None = None):
    cfg = settings()
    with httpx.Client(timeout=timeout or cfg.model_timeout, trust_env=False) as client:
        response = client.post(
            cfg.model_runtime_url + path,
            json=payload or {},
            headers={"X-Runtime-Token": cfg.model_runtime_token},
        )
        if response.status_code == 422 and response.json().get("code") == "context_budget":
            raise ModelContextExceeded(response.json()["error"])
        response.raise_for_status()
        return response.json()


def profile_snapshot(mode: str) -> dict:
    cfg = settings()
    if mode.startswith("m_"):
        from .model_registry import snapshot

        return snapshot(mode)
    if mode not in {"local", "api", "hybrid"}:
        raise ValueError("未知模型方案")
    controller_source = (
        cfg.hybrid_controller_source if mode == "hybrid" else ("api" if mode == "api" else "local")
    )
    vision_source = cfg.hybrid_vision_source if mode == "hybrid" else controller_source
    external = "api" in {controller_source, vision_source}
    return {
        "mode": mode,
        "controller": cfg.api_model if controller_source == "api" else "Qwen3-VL-4B-Instruct",
        "vision": cfg.vision_api_model if vision_source == "api" else "Qwen3-VL-4B-Instruct",
        "controller_source": controller_source,
        "vision_source": vision_source,
        "network": ("mixed" if mode == "hybrid" else "external_api") if external else "local_only",
        "embedding": cfg.embedding_model,
        "reranker": cfg.reranker_model,
        "transfer": "endless",
        "transfer_runtime": cfg.transfer_runtime,
        "repair": "lama",
        "max_tools": 5,
        "api_base_url": cfg.api_base_url if external else None,
    }


def generate(
    snapshot: dict,
    messages: list[dict],
    images: list[Path] | None = None,
    max_tokens: int = 700,
    structured: bool = False,
) -> str:
    cfg = settings()
    source = snapshot.get("vision_source" if images else "controller_source")
    if source == "ollama":
        return ollama_generate(
            snapshot["vision"] if images else snapshot["controller"],
            messages,
            images or [],
            max_tokens,
            structured,
        )
    if source == "local" or (source is None and snapshot["network"] == "local_only"):
        result = runtime_call(
            "/generate",
            {
                "messages": messages,
                "images": [str(p.resolve()) for p in images or []],
                "max_tokens": max_tokens,
                "preserve_history": True,
            },
        )
        from .logging import log

        log.info(
            "local_model_context",
            input_tokens=result.get("input_tokens"),
            history_messages_dropped=result.get("history_messages_dropped", 0),
        )
        if result.get("history_messages_dropped", 0):
            # Older running model services may not yet support preserve_history.
            raise ModelContextExceeded("模型上下文超出预算，需要先压缩对话历史。")
        return result["text"]
    from dotenv import dotenv_values

    api_key = dotenv_values(ROOT / ".env").get("ENDLEXX_API_KEY", cfg.api_key) or ""
    if snapshot.get("connection_id"):
        from .model_registry import credential

        api_key = credential(snapshot["connection_id"], snapshot.get("credential_ref"))
    if not api_key:
        raise ValueError("API 方案尚未配置密钥；请在本机 .env 中设置 ENDLEXX_API_KEY")
    api_messages = [dict(item) for item in messages]
    if images:
        parts = [{"type": "text", "text": api_messages[-1]["content"]}]
        for path in images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
                    },
                }
            )
        api_messages[-1]["content"] = parts
    payload = {
        "model": snapshot["vision"] if images else snapshot["controller"],
        "messages": api_messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if structured:
        payload["response_format"] = {"type": "json_object"}
    with httpx.Client(timeout=cfg.model_timeout) as client:
        response = client.post(
            snapshot["api_base_url"].rstrip("/") + "/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer " + api_key},
        )
        # Never persist upstream responses: they may contain credentials or user material.
        if response.status_code >= 400:
            error = response.text.lower() if response.status_code in {400, 413, 422} else ""
            if response.status_code in {400, 413, 422} and any(
                marker in error
                for marker in (
                    "context_length",
                    "context window",
                    "maximum context",
                    "too many tokens",
                    "context budget",
                    "input length",
                    "token limit",
                )
            ):
                raise ModelContextExceeded("模型上下文超出预算，需压缩历史或精简当前需求与资料。")
            raise RuntimeError(f"模型 API 请求失败，HTTP {response.status_code}")
        return response.json()["choices"][0]["message"]["content"]


def ollama_generate(model, messages, images, max_tokens, structured):
    from filelock import FileLock

    from .deployments import OLLAMA_URL, ensure_ollama

    ensure_ollama()
    api_messages = [dict(row) for row in messages]
    # Conservative UTF-8 byte budget keeps indispensable instructions/current input
    # inside the configured context even without a second tokenizer installation.
    if sum(len(str(row.get("content", "")).encode("utf-8")) for row in api_messages) > 5700:
        raise ModelContextExceeded("当前需求与必要资料超过本地模型上下文预算，请精简需求或角色档案。")
    if images:
        api_messages[-1]["images"] = [base64.b64encode(path.read_bytes()).decode() for path in images]
    payload = dict(
        model=model,
        messages=api_messages,
        stream=False,
        think=False,
        keep_alive=0,
        options=dict(num_ctx=8192, num_predict=min(max_tokens, 1400), temperature=0.1),
    )
    if structured:
        payload["format"] = "json"
    with FileLock(str(settings().gpu_lock), timeout=settings().task_timeout):
        try:
            runtime_call("/unload", timeout=10)
        except httpx.HTTPError:
            pass  # Fresh installations do not have the native research runtime.
        with httpx.Client(timeout=settings().model_timeout, trust_env=False) as client:
            response = client.post(OLLAMA_URL + "/api/chat", json=payload)
            if not response.is_success:
                raise ValueError(f"本地模型运行失败（HTTP {response.status_code}），请检查可用内存或重新部署")
            content = response.json()["message"]["content"]
            if not content.strip():
                raise RuntimeError("所选模型未在输出预算内生成最终答复，请改用 Instruct 标签或精简需求。")
            return content


def model_status():
    cfg = settings()
    try:
        with httpx.Client(timeout=2, trust_env=False) as client:
            runtime = client.get(cfg.model_runtime_url + "/health").json()
    except (httpx.HTTPError, ValueError):
        runtime = {"ready": False, "reason": "本地模型服务未启动"}
    return {
        "runtime": runtime,
        "profiles": [
            {
                "id": "local",
                "external": False,
                "name": "本地多模态",
                "available": runtime.get("ready", False),
                "description": "Qwen3-VL 本地理解与规划 + 本地工具，素材不外发",
            },
            {
                "id": "api",
                "external": True,
                "name": "多模态 API",
                "available": bool(cfg.api_key),
                "description": "外部 API 主控 + 本地图像工具；图片与上下文会发送至配置服务",
            },
            {
                "id": "hybrid",
                "external": "api" in {cfg.hybrid_controller_source, cfg.hybrid_vision_source},
                "name": "多模型协作",
                "available": (
                    (
                        runtime.get("ready", False)
                        or "local" not in {cfg.hybrid_controller_source, cfg.hybrid_vision_source}
                    )
                    and (
                        bool(cfg.api_key)
                        or "api" not in {cfg.hybrid_controller_source, cfg.hybrid_vision_source}
                    )
                ),
                "description": f"主控 {cfg.hybrid_controller_source} / 视觉 {cfg.hybrid_vision_source} + 本地 BGE、Endless、LaMa；API 分工会外发相应内容",
            },
        ],
        "repair": {
            "weights_present": cfg.lama_path.is_file(),
            "ppo_available": False,
            "reason": "PPO 旧权重缺少 StateEncoder，当前启用 LaMa 基线",
        },
        "storage": "PostgreSQL + pgvector"
        if cfg.database_url.startswith("postgresql")
        else "SQLite 本地开发模式",
        "queue": cfg.task_backend,
    }
