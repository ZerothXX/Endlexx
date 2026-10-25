"""Build and verify a reproducible, allowlisted GitHub source archive (stdlib only)."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"
PREFIX = f"Endlexx-{VERSION}"
ROOT_FILES = (
    ".env.example", ".gitignore", ".gitattributes", "README.md", "README.en.md", "LICENSE.md",
    "THIRD_PARTY_NOTICES.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md",
    "pyproject.toml", "requirements.lock", "requirements-model.txt", "requirements-bundle.txt",
    "start.cmd", "start.ps1", "alembic.ini", "compose.yaml",
    "frontend/package.json", "frontend/package-lock.json", "frontend/index.html",
    "frontend/tsconfig.json", "frontend/vite.config.ts", "frontend/playwright.config.ts",
    "vendor/endless/README.md", "vendor/endless/source-manifest.json",
)
TREES = {
    "endlexx": {".py"}, "scripts": {".py", ".ps1"}, "tests": {".py"},
    "migrations": {".py"}, "evals": {".json"}, "frontend/src": {".ts", ".tsx", ".css"},
    "frontend/tests": {".ts"}, ".github/workflows": {".yml", ".yaml"},
    "vendor/endless": {".py"},
}
DOCS = (
    "环境自动配置.md", "模型管理.md", "实现状态.md", "验收报告.md",
    "Agent质量与改进验收.md", "RELEASE.md",
)
FORBIDDEN_PARTS = {
    "data", ".git", ".venv", ".modelenv", "node_modules", "__pycache__",
    ".archside", ".pytest_cache", ".ruff_cache", "PPO", "dist", "build",
}
FORBIDDEN_SUFFIXES = {
    ".pyc", ".db", ".sqlite", ".sqlite3", ".pt", ".pth", ".safetensors",
    ".ckpt", ".onnx", ".bin", ".pem", ".key", ".log", ".tsbuildinfo",
}
MAX_FILE_SIZE = 5 * 1024 * 1024
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:ghp_|gho_|ghu_|ghs_)[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
)


def validate_source(name, content):
    path = PurePosixPath(name)
    if (
        path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
        or any(part in FORBIDDEN_PARTS for part in path.parts)
        or (path.name.startswith(".env") and path.name != ".env.example")
        or path.suffix.lower() in FORBIDDEN_SUFFIXES
    ):
        raise ValueError(f"Forbidden archive path: {name}")
    if len(content) > MAX_FILE_SIZE:
        raise ValueError(f"Unexpected large source file: {name}")
    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
        # Never include the matched value in an exception/log.
        raise ValueError(f"Potential credential in source file: {name}")


def collect_sources(root):
    sources = {}
    names = list(ROOT_FILES) + ["docs/" + name for name in DOCS]
    for directory, suffixes in TREES.items():
        folder = root / directory
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"Missing or unsafe source directory: {directory}")
        names.extend(
            path.relative_to(root).as_posix()
            for path in folder.rglob("*")
            if path.suffix in suffixes and "__pycache__" not in path.parts
        )
    for name in sorted(set(names)):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe source link: {name}")
        if not path.is_file():
            raise ValueError(f"Missing required source file: {name}")
        content = path.read_bytes()
        validate_source(name, content)
        sources[name] = content
    inventory = "vendor/endless/source-manifest.json"
    if inventory in sources:
        for name, digest in json.loads(sources[inventory]).items():
            content = sources.get("vendor/endless/" + name)
            if content is None or hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Vendor snapshot hash mismatch: {name}")
    return sources


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip():
            raise ValueError("Duplicate paths or damaged ZIP contents")
        manifest = json.loads(archive.read(f"{PREFIX}/SOURCE_MANIFEST.json"))
        expected = {f"{PREFIX}/{name}" for name in manifest["files"]}
        if set(names) != expected | {f"{PREFIX}/SOURCE_MANIFEST.json"}:
            raise ValueError("Archive contents do not match the source manifest")
        for name, digest in manifest["files"].items():
            info = archive.getinfo(f"{PREFIX}/{name}")
            if info.file_size > MAX_FILE_SIZE or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive member: {name}")
            content = archive.read(info)
            validate_source(name, content)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Source hash mismatch: {name}")
    return {"version": VERSION, "files": len(expected), "bytes": path.stat().st_size}


def build_archive(root, output):
    sources = collect_sources(root)
    manifest = {
        "version": VERSION,
        "kind": "single-user research preview source",
        "excludes": ["credentials", "databases", "assets", "weights", "runtimes", "legacy PPO research"],
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
    }
    sources["SOURCE_MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(sources.items()):
                info = zipfile.ZipInfo(f"{PREFIX}/{name}", date_time=(2026, 10, 26, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
        result = verify_archive(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return result | {"archive": output.name, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / f"dist/{PREFIX}-source.zip")
    parser.add_argument("--verify", type=Path, help="Verify an existing archive without extracting it")
    args = parser.parse_args()
    result = verify_archive(args.verify) if args.verify else build_archive(ROOT, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
