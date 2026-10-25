"""First-run setup. Standard library only: usable before application dependencies exist."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE_VERSION = "22.23.4"
NODE_SHA256 = "cbc74b5e0982c9dbb41525fd8fb5dcd32fbee237863f7829072f4639a1ce5f20"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(30):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.02)


def dependencies_valid(root=ROOT):
    """Validate all pinned versions, then check imports/transitive requirements separately."""
    try:
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        return False
    for line in (root / "requirements.lock").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            if importlib.metadata.version(requirement.name) not in requirement.specifier:
                return False
        except importlib.metadata.PackageNotFoundError:
            return False
    return True


def frontend_fingerprint(root=ROOT):
    folder = root / "frontend"
    sources = sorted((folder / "src").rglob("*"))
    sources.extend(
        folder / name
        for name in ("package.json", "package-lock.json", "index.html", "tsconfig.json", "vite.config.ts")
    )
    digest = hashlib.sha256()
    for path in sources:
        if path.is_file():
            digest.update(str(path.relative_to(folder)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def frontend_present(root=ROOT):
    dist = root / "frontend/dist"
    index = dist / "index.html"
    if not index.is_file():
        return False
    assets = re.findall(r'(?:src|href)="/((?:assets|static)/[^"]+)"', index.read_text(encoding="utf-8"))
    return bool(assets) and all((dist / name).is_file() for name in assets)


def initialize_config(root=ROOT):
    target = root / ".env"
    if not target.exists():
        text = (root / ".env.example").read_text(encoding="utf-8")
        text = text.replace("replace-with-random-local-token", secrets.token_hex(24))
        # Exclusive creation protects an existing configuration even across launchers.
        try:
            with target.open("x", encoding="utf-8") as stream:
                stream.write(text)
        except FileExistsError:
            pass


class Bootstrap:
    def __init__(self, root=ROOT):
        self.root = root
        self.path = root / "data/environment/bootstrap.json"
        self.job = dict(status="running", phase="检查应用环境", items=[], updated_at=time.time())
        self.log = root / "data/logs/bootstrap.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.env = dict(
            os.environ, PIP_CACHE_DIR=str(root / "data/cache/pip"), PIP_DISABLE_PIP_VERSION_CHECK="1"
        )

    def status(self, phase, **extra):
        self.job.update(phase=phase, updated_at=time.time(), **extra)
        write_json(self.path, self.job)
        print(phase, flush=True)

    def command(self, arguments, phase, env=None):
        self.status(phase)
        with self.log.open("a", encoding="utf-8") as stream:
            result = subprocess.run(
                arguments,
                cwd=self.root,
                env=env or self.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=3600,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        if result.returncode:
            raise ValueError(f"{phase}失败，请检查网络并重新启动；详情见 data/logs/bootstrap.log")

    def ensure_node(self):
        executable = shutil.which("node")
        if executable:
            try:
                if self.find_npm(Path(executable)) and (
                    int(
                        subprocess.check_output(
                            [executable, "-p", "process.versions.node.split('.')[0]"], timeout=10
                        )
                    )
                    >= 22
                ):
                    return Path(executable)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        if os.name != "nt":
            raise ValueError("此系统的 Node 自动安装尚未支持；请安装 Node 22+ 后重试")
        folder = self.root / f"data/runtimes/node-v{NODE_VERSION}-win-x64"
        executable = folder / "node.exe"
        if executable.exists():
            return executable
        archive = self.root / f"data/runtimes/node-v{NODE_VERSION}-win-x64.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != NODE_SHA256:
            self.status("下载项目专用 Node.js", completed=0, total=None)
            partial = archive.with_suffix(".part")
            with (
                urllib.request.urlopen(
                    f"https://nodejs.org/dist/v{NODE_VERSION}/{archive.name}", timeout=60
                ) as response,
                partial.open("wb") as stream,
            ):
                total = int(response.headers.get("Content-Length", 0)) or None
                received, saved = 0, 0
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    received += len(chunk)
                    if time.monotonic() - saved > 0.5:
                        self.status("下载项目专用 Node.js", completed=received, total=total)
                        saved = time.monotonic()
            if hashlib.sha256(partial.read_bytes()).hexdigest() != NODE_SHA256:
                raise ValueError("Node.js 下载校验失败，重新启动可重试")
            os.replace(partial, archive)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (archive.parent / member.filename).resolve()
                if (
                    not target.is_relative_to(archive.parent.resolve())
                    or ((member.external_attr >> 16) & 0o170000) == 0o120000
                ):
                    raise ValueError("Node.js 压缩包包含不安全路径")
            bundle.extractall(archive.parent)
        return executable

    @staticmethod
    def find_npm(node):
        bundled = node.parent / "node_modules/npm/bin/npm-cli.js"
        if bundled.is_file():
            return bundled
        candidate = shutil.which("npm.cmd" if os.name == "nt" else "npm")
        if candidate:
            resolved = Path(candidate).resolve()
            if resolved.name == "npm-cli.js":
                return resolved
            bundled = resolved.parent / "node_modules/npm/bin/npm-cli.js"
            if bundled.is_file():
                return bundled
        return None

    def run(self, repair=False, build=False):
        try:
            self.status("[2/4] 检查并配置应用依赖")
            valid = dependencies_valid(self.root)
            if valid:
                valid = (
                    subprocess.run(
                        [sys.executable, "-m", "pip", "check"], capture_output=True, timeout=60
                    ).returncode
                    == 0
                )
            if not valid or repair:
                self.command([sys.executable, "-m", "ensurepip", "--upgrade"], "准备应用包管理器")
                self.command(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "-r",
                        str(self.root / "requirements.lock"),
                    ],
                    "安装锁定的应用依赖",
                )
            self.command([sys.executable, "-m", "pip", "check"], "验证应用依赖")
            self.command(
                [
                    sys.executable,
                    "-c",
                    "import fastapi,uvicorn,sqlalchemy,langgraph,celery,cv2; import endlexx.api",
                ],
                "验证应用模块",
            )
            self.status("[3/4] 检查并准备工作台页面")
            marker = self.root / "data/environment/frontend.json"
            fingerprint = frontend_fingerprint(self.root)
            prior = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
            if build or not frontend_present(self.root) or prior.get("fingerprint") != fingerprint:
                node = self.ensure_node()
                env = dict(
                    os.environ,
                    PATH=str(node.parent) + os.pathsep + os.environ.get("PATH", ""),
                    npm_config_cache=str(self.root / "data/cache/npm"),
                )
                npm = self.find_npm(node)
                if npm is None:
                    raise ValueError(
                        "检测到 Node 但 npm 不完整；请移除损坏的 Node 安装或使用项目专用 Node 后重试"
                    )
                self.command(
                    [
                        str(node),
                        str(npm),
                        "--prefix",
                        str(self.root / "frontend"),
                        "ci",
                        "--no-audit",
                        "--no-fund",
                    ],
                    "安装工作台依赖",
                    env,
                )
                self.command(
                    [str(node), str(npm), "--prefix", str(self.root / "frontend"), "run", "build"],
                    "构建工作台页面",
                    env,
                )
                if not frontend_present(self.root):
                    raise ValueError("工作台构建产物不完整")
                write_json(marker, dict(fingerprint=fingerprint))
            self.status("[4/4] 配置本机工作空间")
            initialize_config(self.root)
            for name in ("assets", "tasks", "logs", "models", "deployments"):
                (self.root / "data" / name).mkdir(parents=True, exist_ok=True)
            self.status(
                "应用环境已就绪；模型环境按所选方式自动准备", status="completed", completed=0, total=None
            )
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, ValueError)
                else f"环境配置失败（{type(exc).__name__}），请检查网络、磁盘与项目目录写入权限后重试"
            )
            self.status(message, status="failed", error=message)
            raise SystemExit(1) from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--build", action="store_true")
    options = parser.parse_args()
    Bootstrap().run(options.repair, options.build)
