from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ENDLEXX_", env_file=ROOT / ".env", extra="ignore")
    database_url: str = "sqlite:///./data/endlexx.db"
    data_dir: Path = ROOT / "data"
    endless_root: Path = ROOT.parent / "Endless_xxx"
    model_python: Path = ROOT / ".modelenv/Scripts/python.exe"
    lama_path: Path = ROOT.parent / "image_inpainting/anime-manga-big-lama.pt"
    model_runtime_url: str = "http://127.0.0.1:8766"
    model_runtime_token: str = ""
    task_backend: str = "local"
    redis_url: str = "redis://127.0.0.1:6379/0"
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    api_model: str = "qwen3-vl-plus"
    vision_api_model: str = "qwen3-vl-plus"
    hybrid_controller_source: Literal["local", "api"] = "local"
    hybrid_vision_source: Literal["local", "api"] = "local"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    require_access_token: bool = False
    access_token: str = ""
    model_timeout: int = 240
    task_timeout: int = 1800
    context_history_bytes: int = Field(default=3000, ge=512, le=100000)
    transfer_runtime: Literal["balanced", "research"] = "balanced"

    @field_validator("data_dir", "endless_root", "model_python", "lama_path")
    @classmethod
    def resolve_workspace_path(cls, value: Path) -> Path:
        return (value if value.is_absolute() else ROOT / value).resolve()

    @property
    def gpu_lock(self) -> Path:
        return self.data_dir.resolve() / "gpu.lock"


@lru_cache
def settings() -> Settings:
    return Settings()


def refresh_model_paths():
    """Reload only installer-owned paths; keep the running app/DB configuration stable."""
    from dotenv import dotenv_values

    values = dotenv_values(ROOT / ".env")
    cfg = settings()
    for name in ("endless_root", "model_python", "lama_path"):
        value = values.get("ENDLEXX_" + name.upper())
        if value:
            path = Path(value)
            setattr(cfg, name, (path if path.is_absolute() else ROOT / path).resolve())
