"""Managed, resumable downloads and isolated inference runtimes.

Jobs run outside the web process; progress describes bytes/stages, never a timer.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx
from filelock import FileLock

from .config import ROOT, settings
from .model_registry import atomic_json, get_entry, update_entry

OLLAMA_URL = "http://127.0.0.1:11435"
# Pinned official Windows x64 release; verified against its published SHA256.
OLLAMA_RELEASE = "v0.40.10"
OLLAMA_SHA256 = "d52aea2d323bdd324bf66a4b1226844aa5fb665cc5974fbcceceda1ce3d934eb"
CATALOG = [
    {
        "model": "qwen3:8b",
        "name": "Qwen3 8B · 通用主控",
        "size": "约 5.2 GB · 较高内存占用，图片使用已有本地 VLM",
    },
    {
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "name": "Qwen3 4B Instruct · 通用主控",
        "size": "约 2.5 GB · 图片交给已有本地 VLM",
    },
    {"model": "qwen3-vl:2b-instruct", "name": "Qwen3-VL 2B Instruct", "size": "约 1.9 GB · 入门"},
    {"model": "qwen3-vl:4b-instruct", "name": "Qwen3-VL 4B Instruct", "size": "约 3.3 GB · 推荐"},
    {"model": "qwen3-vl:8b-instruct", "name": "Qwen3-VL 8B Instruct", "size": "约 6.2 GB · 需更多内存"},
]
# A fixed manifest; there is intentionally no component picker in the product.
BUNDLE = [
    ("Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "models/Qwen3-VL-4B-Instruct", None),
    (
        "Illustrious XL",
        "OnomaAIResearch/Illustrious-xl-early-release-v0",
        "models/Illustrious-xl-early-release-v0",
        ["Illustrious-XL-v0.1.safetensors"],
    ),
    ("SDXL VAE", "madebyollin/sdxl-vae-fp16-fix", "models/sdxl-vae-fp16-fix", ["sdxl.vae.safetensors"]),
    (
        "ControlNet Canny",
        "diffusers/controlnet-canny-sdxl-1.0",
        "models/controlnet-canny-sdxl-1.0",
        ["config.json", "diffusion_pytorch_model.fp16.safetensors"],
    ),
    (
        "SAM2",
        "facebook/sam2-hiera-large",
        "models/sam2",
        ["config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json"],
    ),
    ("BGE-M3", "BAAI/bge-m3", "retrieval/bge-m3", None),
    ("BGE Reranker", "BAAI/bge-reranker-v2-m3", "retrieval/bge-reranker-v2-m3", None),
    ("Anime LaMa", "df1412/anime-big-lama", "repair", ["anime-manga-big-lama.pt"]),
    ("SDXL 配置与分词器", "stabilityai/stable-diffusion-xl-base-1.0", "models/sdxl-config", ["@sdxl-config"]),
]


class Cancelled(Exception):
    pass


def jobs_dir():
    folder = settings().data_dir / "deployments"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def job_path(model_id):
    # Never accept a client-controlled filesystem path.
    if not __import__("re").fullmatch(r"m_[a-z0-9]{1,16}", model_id):
        raise ValueError("无效的模型 ID")
    return jobs_dir() / (model_id + ".json")


def read_job(model_id):
    path = job_path(model_id)
    if not path.exists():
        return None
    job = json.loads(path.read_text(encoding="utf-8"))
    # The process releases this lock on exit, including unexpected termination.
    if job["status"] in {"queued", "running"} and time.time() - job["updated_at"] > 20:
        from filelock import Timeout

        try:
            with FileLock(str(path) + ".process", timeout=0):
                job.update(status="interrupted", phase="安装进程已停止，可重试并复用已下载文件")
        except Timeout:
            pass
    return job


def launch(model_id):
    entry = get_entry(model_id)
    from .environment import assert_deployment_hardware

    assert_deployment_hardware(entry["kind"], entry["model"])
    prior = read_job(model_id)
    if prior and prior["status"] in {"queued", "running"}:
        raise ValueError("此模型正在部署")
    names = (
        ["Ollama 运行时", entry["model"], "模型能力验证"]
        if entry["kind"] == "local"
        else ["隔离推理环境", *[row[0] for row in BUNDLE], "部署与验证"]
    )
    job = dict(
        model_id=model_id,
        status="queued",
        phase="等待部署",
        current=0,
        items=[dict(name=name, status="queued", completed=0, total=None) for name in names],
        updated_at=time.time(),
        error=None,
    )
    path = job_path(model_id)
    path.with_suffix(".cancel").unlink(missing_ok=True)
    atomic_json(path, job)
    update_entry(model_id, status="installing")
    env = dict(
        os.environ,
        ENDLEXX_DATA_DIR=str(settings().data_dir),
        PYTHONIOENCODING="utf-8",
        USE_TF="0",
        USE_FLAX="0",
    )
    logs = settings().data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "deployments.log").open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/run_deployment.py"), model_id],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    return job | {"pid": process.pid}


def cancel(model_id):
    job = read_job(model_id)
    if job and job["status"] in {"queued", "running"}:
        job_path(model_id).with_suffix(".cancel").touch()
    return job


def record_process(name, pid):
    path = ROOT / "data/processes.json"
    with FileLock(str(ROOT / "data/launcher.lock"), timeout=45):
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        value[name] = pid
        atomic_json(path, value)


class Installer:
    def __init__(self, model_id):
        self.path = job_path(model_id)
        self.entry = get_entry(model_id)
        self.job = json.loads(self.path.read_text(encoding="utf-8"))
        self.last_save = 0

    def check(self):
        if self.path.with_suffix(".cancel").exists():
            raise Cancelled()

    def save(self, **changes):
        self.check()
        self.job.update(changes, updated_at=time.time())
        atomic_json(self.path, self.job)

    def step(self, index, phase=None):
        self.save(current=index, phase=phase or self.job["items"][index]["name"])
        self.job["items"][index]["status"] = "running"
        self.save()

    def progress(self, completed=0, total=None, detail=""):
        self.check()
        self.job["items"][self.job["current"]].update(completed=completed, total=total, detail=detail)
        if time.monotonic() - self.last_save > 0.25:
            self.save()
            self.last_save = time.monotonic()

    def done(self, detail=""):
        item = self.job["items"][self.job["current"]]
        item.update(status="completed", detail=detail)
        if item.get("total"):
            item["completed"] = item["total"]
        self.save()

    def download(
        self, url, target, expected_hash=None, expected_size=None, completed_before=0, component_total=None
    ):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and (not expected_size or target.stat().st_size == expected_size):
            if not expected_hash or sha256(target) == expected_hash:
                self.progress(
                    completed_before + target.stat().st_size,
                    component_total or target.stat().st_size,
                    target.name + " · 已存在",
                )
                return
        partial = target.with_name(target.name + ".part")
        partial_meta = target.with_name(target.name + ".part.json")
        identity = dict(url=url, sha256=expected_hash, size=expected_size)
        prior = json.loads(partial_meta.read_text(encoding="utf-8")) if partial_meta.exists() else None
        if partial.exists() and prior != identity:
            partial.unlink()
        atomic_json(partial_meta, identity)
        offset = partial.stat().st_size if partial.exists() else 0
        # URLs are revision-pinned, so a resumed part cannot silently switch revisions.
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            with client.stream(
                "GET", url, headers={"Range": f"bytes={offset}-"} if offset else {}
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                elif offset:
                    if not response.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("下载服务器返回了不匹配的续传位置，请重试")
                total = expected_size or (int(response.headers.get("content-length", 0)) + offset) or None
                if total and shutil.disk_usage(target.parent).free < total - offset + 256 * 1024**2:
                    raise ValueError("磁盘空间不足，请清理模型所在磁盘后重试")
                with partial.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        self.check()
                        stream.write(chunk)
                        offset += len(chunk)
                        self.progress(completed_before + offset, component_total or total, target.name)
        if expected_size and offset != expected_size:
            raise ValueError(f"下载文件大小不匹配：{target.name}")
        if expected_hash and sha256(partial) != expected_hash:
            partial.unlink(missing_ok=True)
            raise ValueError(f"下载校验失败：{target.name}，可重试")
        os.replace(partial, target)
        partial_meta.unlink(missing_ok=True)

    def command(self, arguments, phase):
        self.save(phase=phase)
        with (settings().data_dir / "logs/deployments.log").open("a", encoding="utf-8") as stream:
            child = subprocess.Popen(
                arguments,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, USE_TF="0", USE_FLAX="0"),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            started = time.monotonic()
            try:
                while child.poll() is None:
                    self.check()
                    if time.monotonic() - started > 3600:
                        raise ValueError("环境安装超时，可在网络恢复后重试")
                    self.save()
                    time.sleep(0.5)
            except BaseException:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    child.terminate()
                child.wait(timeout=15)
                raise
            if child.returncode:
                raise ValueError(f"{phase}失败（退出码 {child.returncode}）；详见 data/logs/deployments.log")

    def ollama(self):
        self.step(0)
        ensure_ollama(self)
        self.done("独立运行在本机 11435 端口")
        self.step(1, "下载主控模型")
        with httpx.Client(timeout=httpx.Timeout(600, connect=10), trust_env=False) as client:
            layers = {}
            with client.stream(
                "POST", OLLAMA_URL + "/api/pull", json={"model": self.entry["model"], "stream": True}
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    self.check()
                    if not line:
                        continue
                    update = json.loads(line)
                    if update.get("error"):
                        raise ValueError("模型下载失败：请确认该模型标签存在，或检查 Ollama 下载日志")
                    if update.get("digest") and update.get("total"):
                        layers[update["digest"]] = (update.get("completed", 0), update["total"])
                    self.progress(
                        sum(row[0] for row in layers.values()),
                        sum(row[1] for row in layers.values()) or None,
                        update.get("status", ""),
                    )
            self.done("模型下载及校验完成")
            self.step(2)
            info = client.post(OLLAMA_URL + "/api/show", json={"model": self.entry["model"]})
            info.raise_for_status()
            capabilities = info.json().get("capabilities", [])
            if "completion" not in capabilities:
                raise ValueError("此模型不能生成对话，请选择通用指令或视觉指令模型")
            has_vision = "vision" in capabilities
            if (
                not has_vision
                and not (settings().endless_root / "models/Qwen3-VL-4B-Instruct/config.json").is_file()
            ):
                raise ValueError("通用主控已下载；请先部署多模型组合，为图片任务提供本地视觉模型后重试")
            from .providers import ollama_generate

            text = ollama_generate(
                self.entry["model"], [{"role": "user", "content": "Reply OK"}], [], 64, False
            )
            if not text:
                raise ValueError("模型加载后未返回内容，请检查可用内存")
            if not has_vision:
                update_entry(self.entry["id"], capabilities=capabilities, vision_fallback="native")
                self.done("通用模型负责规划与摘要；图片输入使用已有本地 Qwen3-VL")
                return
            from PIL import Image

            probe = self.path.with_suffix(".vision.png")
            Image.new("RGB", (128, 128), "red").save(probe)
            try:
                observation = ollama_generate(
                    self.entry["model"],
                    [
                        {
                            "role": "user",
                            "content": "What is the dominant color of this image? Reply with one English color word.",
                        }
                    ],
                    [probe],
                    32,
                    False,
                )
                if "red" not in observation.lower():
                    raise ValueError("模型未通过基础图片输入验证，请检查视觉模型是否完整")
            finally:
                probe.unlink(missing_ok=True)
            update_entry(self.entry["id"], capabilities=capabilities, vision_fallback=None)
        self.done("已验证模型可加载并支持视觉输入")

    def hf_component(self, repo, directory, files):
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.get(f"https://huggingface.co/api/models/{repo}?blobs=true")
            response.raise_for_status()
            manifest = response.json()
        revision = manifest["sha"]
        candidates = manifest["siblings"]
        if files == ["@sdxl-config"]:
            files = [row["rfilename"] for row in candidates if row["rfilename"].endswith((".json", ".txt"))]
        if files is None:
            # One PyTorch implementation only; omit ONNX, alternate checkpoints and code.
            has_safe = any(
                row["rfilename"].endswith(".safetensors") and "/" not in row["rfilename"]
                for row in candidates
            )
            files = [
                row["rfilename"]
                for row in candidates
                if "/" not in row["rfilename"]
                and (
                    row["rfilename"].endswith((".json", ".txt", ".model", ".safetensors"))
                    or (row["rfilename"] == "pytorch_model.bin" and not has_safe)
                )
                and not row["rfilename"].startswith(("onnx", "openvino"))
            ]
        lookup = {row["rfilename"]: row for row in candidates}
        component_total = sum(
            (lookup.get(name, {}).get("size") or (lookup.get(name, {}).get("lfs") or {}).get("size") or 0)
            for name in files
        )
        completed_before = 0
        for filename in files:
            if filename not in lookup or Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError("模型仓库缺少清单文件：" + filename)
            metadata = lookup[filename]
            lfs = metadata.get("lfs") or {}
            self.download(
                f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
                directory / filename,
                lfs.get("sha256"),
                metadata.get("size") or lfs.get("size"),
                completed_before,
                component_total or None,
            )
            completed_before += (directory / filename).stat().st_size
        atomic_json(directory / "endlexx-source.json", dict(repo=repo, revision=revision, files=files))

    def ensure_torch(self):
        cfg = settings()
        python = cfg.model_python
        if not torch_environment_valid(python):
            from .environment import assert_deployment_hardware

            assert_deployment_hardware("hybrid")
            if shutil.disk_usage(cfg.data_dir).free < 12 * 1024**3:
                raise ValueError(
                    "首次安装组合推理依赖建议至少预留 12 GB 可用空间（含下载缓存），请清理项目磁盘后重试"
                )
            # The ML environment is separate from the application environment.
            python = cfg.data_dir / "runtimes/torch/Scripts/python.exe"
            if os.name != "nt":
                python = cfg.data_dir / "runtimes/torch/bin/python"
            self.command([sys.executable, "-m", "venv", str(python.parents[1])], "创建隔离推理环境")
            self.command(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "torch==2.6.0",
                    "torchvision==0.21.0",
                    "torchaudio==2.6.0",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                ],
                "安装 PyTorch CUDA 运行时",
            )
            self.command(
                [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements-bundle.txt")],
                "安装推理依赖",
            )
            self.command([str(python), "-c", TORCH_PROBE], "验证 CUDA 环境（研究组合需要 NVIDIA GPU）")
        from dotenv import set_key

        set_key(str(ROOT / ".env"), "ENDLEXX_MODEL_PYTHON", str(python))
        cfg.model_python = python
        return python

    def bundle(self):
        cfg = settings()
        self.step(0)
        python = self.ensure_torch()
        self.done("隔离推理环境已验证")
        root = cfg.endless_root
        # Existing research directories are read-only inputs to the application installer.
        core_complete = all(
            component_present(root / relative, files)
            for _, _, relative, files in BUNDLE
            if relative.startswith("models/") and relative != "models/sdxl-config"
        )
        if not (root / "model_utils.py").exists() or not core_complete:
            root = cfg.data_dir / "bundle/endless"
            shutil.copytree(ROOT / "vendor/endless", root, dirs_exist_ok=True)
        paths = []
        for index, (_name, repo, relative, files) in enumerate(BUNDLE, 1):
            self.step(index)
            if relative.startswith("models/"):
                directory = root / relative
                if relative == "models/sdxl-config":
                    directory = cfg.data_dir / "bundle/sdxl-config"
            elif relative.startswith("retrieval/"):
                directory = cfg.data_dir / "models" / relative.split("/")[1]
            else:
                directory = cfg.data_dir / "bundle/repair"
                if cfg.lama_path.is_file():
                    directory = cfg.lama_path.parent
            if component_present(directory, files):
                self.progress(detail="复用本机已有文件")
            else:
                # Never write into sibling research repositories.
                if not directory.resolve().is_relative_to(cfg.data_dir.resolve()):
                    raise ValueError("现有研究目录文件不完整，请使用 Endlexx 管理的数据目录重新部署")
                self.hf_component(repo, directory, files)
            paths.append(directory)
            self.done("已下载或复用本机权重")
        self.step(len(BUNDLE) + 1)
        from dotenv import set_key

        for key, value in {
            "MODEL_PYTHON": python,
            "ENDLESS_ROOT": root,
            "LAMA_PATH": paths[7] / "anime-manga-big-lama.pt",
        }.items():
            set_key(str(ROOT / ".env"), "ENDLEXX_" + key, str(value))
        settings.cache_clear()
        start_native(python, root, paths[5], paths[6])
        from .providers import runtime_call

        if not runtime_call(
            "/generate", {"messages": [{"role": "user", "content": "Reply OK"}], "max_tokens": 8}
        ).get("text"):
            raise ValueError("主控模型未能完成加载验证")
        self.done("模型服务已启动；角色迁移另需导入对应角色权重包")

    def run(self):
        with FileLock(str(self.path) + ".process", timeout=0):
            try:
                # Sequential even across two independently requested deployments.
                lock = FileLock(str(settings().data_dir / "deployment.lock"))
                while True:
                    self.check()
                    try:
                        lock.acquire(timeout=0)
                        break
                    except __import__("filelock").Timeout:
                        self.save(phase="等待其他模型部署完成")
                        time.sleep(0.5)
                try:
                    self.save(status="running")
                    if self.entry["kind"] == "local":
                        self.ollama()
                    else:
                        self.bundle()
                    self.save(status="completed", phase="部署完成")
                    update_entry(self.entry["id"], status="ready")
                finally:
                    lock.release()
            except Exception as exc:
                import traceback

                frames = [
                    (Path(frame.filename).name, frame.name, frame.lineno)
                    for frame in traceback.extract_tb(exc.__traceback__)
                ]
                print(
                    json.dumps(
                        dict(deployment=self.entry["id"], error_type=type(exc).__name__, frames=frames)
                    ),
                    flush=True,
                )
                cancelled = isinstance(exc, Cancelled)
                # Do not persist HTTP bodies, URL query strings, tokens or exception reprs.
                error = (
                    "已取消，可重试并复用下载缓存"
                    if cancelled
                    else (
                        str(exc)
                        if isinstance(exc, ValueError)
                        else f"部署失败（{type(exc).__name__}），请检查网络与磁盘后重试"
                    )
                )
                self.job.update(
                    status="cancelled" if cancelled else "failed",
                    error=error,
                    phase=error,
                    updated_at=time.time(),
                )
                self.job["items"][self.job["current"]]["status"] = self.job["status"]
                atomic_json(self.path, self.job)
                update_entry(self.entry["id"], status="failed")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


TORCH_PROBE = "import torch,transformers,diffusers,peft,bitsandbytes,cv2; assert torch.cuda.is_available()"


def torch_environment_valid(python):
    if not Path(python).is_file():
        return False
    try:
        return (
            subprocess.run(
                [str(python), "-c", TORCH_PROBE],
                capture_output=True,
                env=dict(os.environ, USE_TF="0", USE_FLAX="0"),
                timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def component_present(directory, files):
    directory = Path(directory)
    if files == ["@sdxl-config"]:
        return (directory / "endlexx-source.json").is_file()
    if files:
        return all((directory / name).is_file() and (directory / name).stat().st_size > 0 for name in files)
    if not all((directory / name).is_file() for name in ["config.json", "tokenizer_config.json"]):
        return False
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        return bool(shards) and all(
            (directory / name).is_file() and (directory / name).stat().st_size > 0 for name in shards
        )
    return any(
        (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in ["model.safetensors", "pytorch_model.bin"]
    )


def extract_zip(source, directory):
    directory = directory.resolve()
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            target = (directory / item.filename).resolve()
            if not target.is_relative_to(directory) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("运行时压缩包包含不安全路径")
        directory.mkdir(parents=True, exist_ok=True)
        if (
            shutil.disk_usage(directory).free
            < sum(item.file_size for item in archive.infolist()) + 256 * 1024**2
        ):
            raise ValueError("磁盘空间不足以解压运行时，请清理项目磁盘后重试；已下载的压缩包会保留")
        archive.extractall(directory)


def ensure_ollama(installer=None):
    folder = settings().data_dir / "runtimes/ollama"
    folder.mkdir(parents=True, exist_ok=True)
    executable = folder / "ollama.exe"
    installed = shutil.which("ollama")
    if not installed:
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe"
        installed = str(candidate) if candidate.is_file() else None
    if installed:
        executable = Path(installed)
    if not executable.is_file():
        if installer is None:
            raise ValueError("本地运行时尚未安装，请在模型窗口重试部署")
        if os.name != "nt":
            raise ValueError("当前自动运行时安装支持 Windows x64；其他系统请先安装 Ollama")
        archive = folder / (OLLAMA_RELEASE + ".zip")
        installer.download(
            f"https://github.com/ollama/ollama/releases/download/{OLLAMA_RELEASE}/ollama-windows-amd64.zip",
            archive,
            OLLAMA_SHA256,
        )
        installer.save(phase="解压 Ollama 运行时")
        extract_zip(archive, folder)
    with FileLock(str(folder / "service.lock"), timeout=15):
        marker = folder / "service.json"
        state = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
        with httpx.Client(timeout=2, trust_env=False) as client:
            try:
                if client.get(OLLAMA_URL + "/api/version").is_success:
                    if state.get("owner") != str(ROOT):
                        raise ValueError("11435 端口被其他服务占用，请先释放此端口")
                    return
            except httpx.HTTPError:
                pass
        env = dict(
            os.environ,
            OLLAMA_HOST="127.0.0.1:11435",
            OLLAMA_MODELS=str(settings().data_dir / "models/ollama"),
            OLLAMA_NO_CLOUD="1",
            OLLAMA_CONTEXT_LENGTH="8192",
            OLLAMA_MAX_LOADED_MODELS="1",
        )
        logs = settings().data_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / "ollama.log").open("a", encoding="utf-8") as stream:
            child = subprocess.Popen(
                [str(executable), "serve"],
                cwd=folder,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        atomic_json(marker, dict(owner=str(ROOT), pid=child.pid, executable=str(executable)))
        if executable.resolve().is_relative_to(ROOT):
            record_process("ollama", child.pid)
        for _ in range(60):
            if installer:
                installer.check()
            try:
                if httpx.get(OLLAMA_URL + "/api/version", timeout=1, trust_env=False).is_success:
                    return
            except httpx.HTTPError:
                pass
            if child.poll() is not None:
                break
            time.sleep(0.5)
        raise ValueError("本地运行时未能启动，详见 data/logs/ollama.log")


def start_native(python, root, embedding, reranker):
    cfg = settings()
    with httpx.Client(timeout=2, trust_env=False) as client:
        try:
            health = client.get(cfg.model_runtime_url + "/health").json()
            if health.get("ready") and health.get("embedding_ready") and health.get("reranker_ready"):
                return
            raise ValueError("旧模型服务仍在运行，请重启工作台后重试部署验证")
        except httpx.HTTPError:
            pass
    with (cfg.data_dir / "logs/model.log").open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(
            [
                str(python),
                str(ROOT / "scripts/model_runtime.py"),
                "--model",
                str(root / "models/Qwen3-VL-4B-Instruct"),
                "--embedding",
                str(embedding),
                "--reranker",
                str(reranker),
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, USE_TF="0", USE_FLAX="0"),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    record_process("model", child.pid)
    for _ in range(120):
        try:
            health = httpx.get(cfg.model_runtime_url + "/health", timeout=2, trust_env=False).json()
            if health.get("ready") and health.get("embedding_ready") and health.get("reranker_ready"):
                return
        except httpx.HTTPError:
            pass
        if child.poll() is not None:
            break
        time.sleep(0.5)
    raise ValueError("推理服务验证未通过，请查看 data/logs/model.log 后重试")
