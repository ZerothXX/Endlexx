"""Durable model connections. Public metadata and credentials never share a response."""

import json
import os
import re
import secrets
import time
from contextlib import contextmanager
from urllib.parse import urlsplit

from filelock import FileLock

from .config import settings

MODEL_ID = r"^(local|api|hybrid|m_[a-z0-9]{1,16})$"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows readers briefly hold a handle without FILE_SHARE_DELETE. Progress
    # polling must not turn a completed model download into a failed deployment.
    for attempt in range(30):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.02)


@contextmanager
def registry():
    folder = settings().data_dir
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "model-registry.lock"), timeout=15):
        path = folder / "model-registry.json"
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else bootstrap()
        yield value
        atomic_json(path, value)


def bootstrap():
    cfg = settings()
    entries = []
    if (cfg.endless_root / "models/Qwen3-VL-4B-Instruct/config.json").is_file():
        entries.append(
            dict(
                id="m_native",
                name="Qwen3-VL · 已有本地部署",
                kind="local",
                engine="native",
                model="Qwen3-VL-4B-Instruct",
                status="ready",
            )
        )
        if cfg.lama_path.is_file() and (cfg.endless_root / "model_utils.py").is_file():
            entries.append(
                dict(
                    id="m_bundle",
                    name="Endless + LaMa 研究组合",
                    kind="hybrid",
                    engine="native",
                    model="Qwen3-VL-4B-Instruct",
                    status="ready",
                )
            )
    if cfg.api_key:
        entries.append(
            dict(
                id="m_importedapi",
                name="已有 API 配置",
                kind="api",
                engine="api",
                model=cfg.api_model,
                vision_model=cfg.vision_api_model,
                base_url=cfg.api_base_url,
                legacy_secret=True,
                status="ready",
            )
        )
    return {"active_id": entries[0]["id"] if entries else None, "entries": entries}


def listing():
    from .config import refresh_model_paths

    refresh_model_paths()
    with registry() as value:
        return json.loads(json.dumps(value))


def get_entry(model_id):
    with registry() as value:
        entry = next((row.copy() for row in value["entries"] if row["id"] == model_id), None)
    if entry is None:
        raise ValueError("模型配置已不存在，请在输入框下方重新选择模型")
    return entry


def credential(model_id, secret_ref=None):
    with registry() as value:
        entry = next((row for row in value["entries"] if row["id"] == model_id), {})
        path = settings().data_dir / "model-secrets.json"
        credentials = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if secret_ref:
            return credentials.get(secret_ref, "") if secret_ref.startswith(model_id + ":") else ""
        return credentials.get(entry.get("secret_ref") or model_id) or (
            settings().api_key if entry.get("legacy_secret") else ""
        )


def save_api(payload, model_id=None):
    if any(not payload[key].strip() for key in ("name", "model", "vision_model", "base_url")):
        raise ValueError("名称、模型 ID 和 API 地址不能为空")
    if payload.get("api_key") is not None and not payload["api_key"].strip():
        payload = payload | {"api_key": None}
    parsed = urlsplit(payload["base_url"])
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("请填写不带凭据和查询参数的 HTTP(S) API 基础地址")
    with registry() as value:
        entry = next((row for row in value["entries"] if row["id"] == model_id), None)
        if model_id and (entry is None or entry["kind"] != "api"):
            raise ValueError("只能编辑已有 API 配置")
        path = settings().data_dir / "model-secrets.json"
        credentials = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        model_id = model_id or "m_" + secrets.token_hex(6)
        old_ref = (entry or {}).get("secret_ref") or model_id
        key = (
            payload.get("api_key")
            or credentials.get(old_ref)
            or (settings().api_key if (entry or {}).get("legacy_secret") else "")
        )
        if not key:
            raise ValueError("请输入 API Key；编辑时留空可保留已有密钥")
        # Keep earlier credential versions for queued/retried immutable snapshots.
        secret_ref = model_id + ":" + secrets.token_hex(6)
        credentials[secret_ref] = key.strip()
        atomic_json(path, credentials)
        path.chmod(0o600)
        data = {key: payload[key].strip() for key in ("name", "base_url", "model", "vision_model")}
        data["base_url"] = data["base_url"].rstrip("/")
        if entry is None:
            entry = dict(id=model_id, kind="api", engine="api", status="ready")
            value["entries"].append(entry)
        entry.update(data)
        entry.update(secret_ref=secret_ref, legacy_secret=False)
        value["active_id"] = value["active_id"] or model_id
        return entry.copy()


def add_deployment(kind, model):
    if kind == "local" and (
        not re.fullmatch(r"[a-zA-Z0-9_.:/-]{1,120}", model)
        or "cloud" in model.lower()
        or "://" in model
        or ".." in model
    ):
        raise ValueError("请输入 Ollama 支持的本地模型标签，例如 qwen3:4b-instruct 或 qwen3-vl:4b")
    model_id = "m_" + secrets.token_hex(6)
    with registry() as value:
        existing = next(
            (
                row
                for row in value["entries"]
                if row["kind"] == kind
                and (kind == "hybrid" or (row["engine"] == "ollama" and row["model"] == model))
            ),
            None,
        )
        if existing:
            return existing.copy()
        entry = dict(
            id=model_id,
            kind=kind,
            engine="ollama" if kind == "local" else "native",
            name=model if kind == "local" else "Endless + LaMa 研究组合",
            model=model if kind == "local" else "Qwen3-VL-4B-Instruct",
            status="installing",
        )
        value["entries"].append(entry)
        return entry.copy()


def update_entry(model_id, **changes):
    with registry() as value:
        entry = next((row for row in value["entries"] if row["id"] == model_id), None)
        if not entry:
            raise ValueError("模型配置不存在")
        entry.update(changes)
        if entry["status"] == "ready" and not value["active_id"]:
            value["active_id"] = model_id


def activate(model_id):
    entry = get_entry(model_id)
    if entry["status"] != "ready":
        raise ValueError("请等待模型部署完成")
    with registry() as value:
        value["active_id"] = model_id
    return entry


def remove(model_id):
    entry = get_entry(model_id)
    if entry["status"] == "installing":
        raise ValueError("请先取消下载，等待安装停止后再删除")
    with registry() as value:
        value["entries"] = [row for row in value["entries"] if row["id"] != model_id]
        if value["active_id"] == model_id:
            value["active_id"] = next(
                (row["id"] for row in value["entries"] if row["status"] == "ready"), None
            )
        path = settings().data_dir / "model-secrets.json"
        if path.exists():
            credentials = json.loads(path.read_text(encoding="utf-8"))
            credentials = {
                key: item
                for key, item in credentials.items()
                if key != model_id and not key.startswith(model_id + ":")
            }
            atomic_json(path, credentials)


def snapshot(model_id):
    entry = get_entry(model_id)
    if entry["kind"] == "api" and not entry.get("secret_ref"):
        # Migrate a legacy connection without putting any key into the snapshot.
        save_api(
            dict(
                name=entry["name"],
                base_url=entry["base_url"],
                model=entry["model"],
                vision_model=entry.get("vision_model") or entry["model"],
                api_key=None,
            ),
            model_id,
        )
        entry = get_entry(model_id)
    if entry["status"] != "ready":
        raise ValueError("所选模型尚未部署完成，请等待或切换模型")
    if entry["kind"] == "api" and not credential(model_id):
        raise ValueError("此模型尚未配置 API Key")
    source = "api" if entry["kind"] == "api" else ("ollama" if entry["engine"] == "ollama" else "local")
    cfg = settings()
    native_vision = entry.get("vision_fallback") == "native"
    return dict(
        mode=entry["kind"],
        connection_id=model_id,
        connection_name=entry["name"],
        controller=entry["model"],
        vision="Qwen3-VL-4B-Instruct" if native_vision else (entry.get("vision_model") or entry["model"]),
        controller_source=source,
        vision_source="local" if native_vision else source,
        network="external_api" if source == "api" else "local_only",
        api_base_url=entry.get("base_url"),
        credential_ref=entry.get("secret_ref"),
        embedding=cfg.embedding_model,
        reranker=cfg.reranker_model,
        transfer="endless",
        repair="lama",
        transfer_runtime=cfg.transfer_runtime,
        max_tools=5,
    )
