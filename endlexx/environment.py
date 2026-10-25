"""Local capability checks and repair jobs. No user-provided commands are executed."""

import csv
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock, Timeout

from .config import ROOT, refresh_model_paths, settings
from .model_registry import atomic_json


def system_memory():
    if os.name == "nt":
        import ctypes

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page",
                    "available_page",
                    "virtual",
                    "available_virtual",
                    "extended",
                )
            ]

        value = Memory()
        value.length = ctypes.sizeof(value)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            return value.total
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def gpu_info():
    executable = shutil.which("nvidia-smi")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/nvidia-smi.exe"
        executable = str(candidate) if candidate.is_file() else None
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode:
            return []
        return [
            dict(name=row[0].strip(), driver=row[1].strip(), memory_mb=int(row[2].strip()))
            for row in csv.reader(io.StringIO(result.stdout))
            if len(row) == 3
        ]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def hardware_checks(target, model="", *, gpus=None, memory=None, free=None):
    gpus = gpu_info() if gpus is None else gpus
    memory = system_memory() if memory is None else memory
    folder = settings().data_dir
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free if free is None else free
    checks = []

    def add(key, name, status, detail, fixable=False):
        checks.append(dict(id=key, name=name, status=status, detail=detail, fixable=fixable))

    arch_ok = platform.machine().lower() in {"amd64", "x86_64"} and sys.maxsize > 2**32
    add(
        "platform",
        "操作系统与架构",
        "ready" if arch_ok else "blocked",
        platform.system() + " " + platform.machine() + ("" if arch_ok else "；自动部署当前仅支持 x64"),
    )
    minimum = 8 * 1024**3
    add(
        "memory",
        "系统内存",
        "blocked"
        if memory and memory < minimum and target == "hybrid"
        else "ready"
        if memory and memory >= minimum
        else "warning",
        f"{memory / 1024**3:.1f} GB；本地模型的实际加载需求取决于型号"
        if memory
        else "暂无法读取；部署时仍会验证实际加载",
    )
    add(
        "disk",
        "项目磁盘空间",
        "ready" if free >= 1024**3 else "blocked",
        f"剩余 {free / 1024**3:.1f} GB；下载逐文件检查剩余空间，组合首次安装通常需要数十 GB",
    )
    if target == "hybrid":
        compatible = []
        for row in gpus:
            try:
                version = tuple(int(part) for part in row["driver"].split("."))
                if version >= ((528, 33) if os.name == "nt" else (525, 60, 13)):
                    compatible.append(row)
            except ValueError:
                pass
        detail = "; ".join(f"{row['name']} / {row['memory_mb']} MB / 驱动 {row['driver']}" for row in gpus)
        add(
            "gpu",
            "研究组合 NVIDIA 驱动",
            "ready" if compatible else "blocked",
            detail
            if compatible
            else "未检测到可用 NVIDIA 驱动或版本过旧。请安装/更新官方驱动；需要管理员授权或重启时由系统处理。可先使用 API 或本地 CPU 模型。",
        )
        if compatible and max(row["memory_mb"] for row in compatible) < 8 * 1024:
            add(
                "vram",
                "研究组合显存",
                "warning",
                "显存不足 8 GB；可尝试卸载/CPU 卸载，仍须以实际模型加载验证为准",
            )
    else:
        add(
            "gpu",
            "本地模型计算设备",
            "ready" if gpus else "warning",
            " / ".join(row["name"] for row in gpus)
            if gpus
            else "未检测到 NVIDIA GPU；Ollama 可尝试 CPU，速度与可用内存取决于机器",
        )
    return checks


def assert_deployment_hardware(target, model=""):
    blocked = [row for row in hardware_checks(target, model) if row["status"] == "blocked"]
    if blocked:
        raise ValueError("；".join(row["detail"] for row in blocked))


def job_path():
    return settings().data_dir / "environment/repair.json"


def repair_status():
    path = job_path()
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["status"] in {"queued", "running"} and time.time() - value["updated_at"] > 20:
        try:
            with FileLock(str(path) + ".process", timeout=0):
                value.update(status="interrupted", phase="环境配置进程已停止，可重试并复用缓存")
        except Timeout:
            pass
    return value


def launch_repair(target):
    if target not in {"local", "hybrid"}:
        raise ValueError("无效的环境类型")
    assert_deployment_hardware(target)
    path = job_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".launch", timeout=5):
        prior = repair_status()
        if prior and prior["status"] in {"queued", "running"}:
            raise ValueError("环境配置已在进行中")
        names = [
            "硬件与空间检查",
            "Ollama 运行时" if target == "local" else "PyTorch CUDA 与推理依赖",
            "验证环境",
        ]
        value = dict(
            status="queued",
            target=target,
            current=0,
            phase="等待环境配置",
            updated_at=time.time(),
            error=None,
            items=[dict(name=name, status="queued", completed=0, total=None) for name in names],
        )
        path.with_suffix(".cancel").unlink(missing_ok=True)
        atomic_json(path, value)
        logs = settings().data_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        try:
            with (logs / "deployments.log").open("a", encoding="utf-8") as stream:
                subprocess.Popen(
                    [sys.executable, str(ROOT / "scripts/repair_environment.py")],
                    cwd=ROOT,
                    env=dict(os.environ, ENDLEXX_DATA_DIR=str(settings().data_dir), PYTHONIOENCODING="utf-8"),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
        except OSError:
            atomic_json(path, value | dict(status="failed", phase="无法启动环境配置，请检查目录权限后重试"))
            raise ValueError("无法启动环境配置，请检查目录权限后重试") from None
    return value


def cancel_repair():
    value = repair_status()
    if value and value["status"] in {"queued", "running"}:
        job_path().with_suffix(".cancel").touch()
    return value


def report():
    from .deployments import OLLAMA_URL, torch_environment_valid

    refresh_model_paths()
    cfg = settings()
    gpus = gpu_info()
    memory = system_memory()
    free = shutil.disk_usage(cfg.data_dir).free
    checks = hardware_checks("local", gpus=gpus, memory=memory, free=free)
    checks.insert(
        0,
        dict(
            id="app",
            name="应用 Python 与依赖",
            status="ready",
            detail=f"Python {platform.python_version()}；启动器每次检查锁定依赖和应用导入",
            fixable=False,
        ),
    )
    checks.append(
        dict(
            id="web",
            name="工作台页面",
            status="ready" if (ROOT / "frontend/dist/index.html").is_file() else "repairable",
            detail="页面缺失或源码变化时，启动器会自动准备 Node 并构建；可重新运行 start.cmd --repair",
            fixable=False,
        )
    )
    from sqlalchemy import text

    from .db import Session

    service_ok = True
    try:
        with Session() as db:
            db.execute(text("SELECT 1"))
        if cfg.task_backend == "celery":
            import redis

            client = redis.Redis.from_url(cfg.redis_url, socket_connect_timeout=2, socket_timeout=2)
            try:
                client.ping()
            finally:
                client.close()
    except Exception:
        service_ok = False
    detail = (
        (
            "本机 SQLite 与本地队列，不依赖 WSL/Docker"
            if cfg.database_url.startswith("sqlite") and cfg.task_backend == "local"
            else "已验证配置的数据库与队列连通；不会自动切换数据库或迁移聊天记录"
        )
        if service_ok
        else "已配置的数据库或队列不可用；重新启动时会尝试恢复本项目已登记服务，不改写外部数据库配置"
    )
    checks.append(
        dict(
            id="services",
            name="数据库与任务队列",
            status="ready" if service_ok else "blocked",
            detail=detail,
            fixable=False,
        )
    )
    ollama = cfg.data_dir / "runtimes/ollama/ollama.exe"
    found = ollama.is_file() or shutil.which("ollama") is not None
    checks.append(
        dict(
            id="ollama",
            name="本地模型运行时",
            status="ready" if found else "repairable",
            detail="运行时已安装，选择模型时自动启动并验证"
            if found
            else "可自动下载隔离的 Ollama 运行时，无需系统安装",
            fixable=True,
        )
    )
    torch_ready = torch_environment_valid(cfg.model_python)
    checks.append(
        dict(
            id="torch",
            name="组合推理环境",
            status="ready" if torch_ready else "repairable",
            detail="已验证 PyTorch CUDA 和推理依赖"
            if torch_ready
            else "可自动创建独立环境并安装 PyTorch CUDA、Diffusers 等依赖",
            fixable=True,
        )
    )
    combination = hardware_checks("hybrid", gpus=gpus, memory=memory, free=free)
    path = ROOT / "data/environment/bootstrap.json"
    return dict(
        checks=checks,
        combination=combination,
        repair=repair_status(),
        checked_at=time.time(),
        bootstrap=json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None,
        local_runtime_url=OLLAMA_URL,
    )


def run_repair():
    from .deployments import Cancelled, Installer, ensure_ollama

    class Repair(Installer):
        def __init__(self):
            self.path = job_path()
            self.job = json.loads(self.path.read_text(encoding="utf-8"))
            self.last_save = 0

    installer = Repair()
    with FileLock(str(installer.path) + ".process", timeout=0):
        lock = FileLock(str(settings().data_dir / "deployment.lock"))
        try:
            while True:
                installer.check()
                try:
                    lock.acquire(timeout=0)
                    break
                except Timeout:
                    installer.save(phase="等待其他环境或模型安装完成")
                    time.sleep(0.5)
            installer.save(status="running")
            installer.step(0)
            assert_deployment_hardware(installer.job["target"])
            installer.done("硬件与空间检查通过")
            installer.step(1)
            if installer.job["target"] == "local":
                ensure_ollama(installer)
            else:
                installer.ensure_torch()
            installer.done("环境配置完成")
            installer.step(2)
            if installer.job["target"] == "hybrid":
                from .deployments import torch_environment_valid

                if not torch_environment_valid(settings().model_python):
                    raise ValueError("推理依赖验证失败，可重试修复")
            installer.done("环境已验证；模型权重由模型部署任务按需下载")
            installer.save(status="completed", phase="环境配置完成")
        except Exception as exc:
            message = (
                "已取消，可重试并复用缓存"
                if isinstance(exc, Cancelled)
                else str(exc)
                if isinstance(exc, ValueError)
                else f"环境配置失败（{type(exc).__name__}），请检查网络、磁盘和日志后重试"
            )
            status = "cancelled" if isinstance(exc, Cancelled) else "failed"
            installer.job.update(status=status, phase=message, error=message, updated_at=time.time())
            installer.job["items"][installer.job["current"]]["status"] = status
            atomic_json(installer.path, installer.job)
        finally:
            if lock.is_locked:
                lock.release()
