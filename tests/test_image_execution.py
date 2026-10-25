import json
import sys
from types import SimpleNamespace

import pytest

from endlexx.config import settings
from endlexx.images import NeedsClarification, transfer_image
from scripts.endless_inference import configure_offline_loaders


def test_offline_loader_preserves_diffusers_nested_vae_arguments(monkeypatch, tmp_path):
    calls = []
    vae = SimpleNamespace(from_single_file=lambda *a, **kw: calls.append((a, kw)) or "vae")

    def pipeline_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return vae.from_single_file(
            "embedded-vae", config={"sample_size": 1024}, subfolder="embedded", local_files_only=True
        )

    pipeline = SimpleNamespace(from_single_file=pipeline_loader)
    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(AutoencoderKL=vae, StableDiffusionXLPipeline=pipeline)
    )
    configure_offline_loaders(tmp_path)
    assert pipeline.from_single_file("checkpoint", torch_dtype="fp16") == "vae"
    assert calls[0][1]["config"] == str(tmp_path)
    assert calls[1][1]["config"] == {"sample_size": 1024}
    assert calls[1][1]["subfolder"] == "embedded"
    vae.from_single_file("replacement-vae")
    assert calls[2][1] == {"config": str(tmp_path), "subfolder": "vae", "local_files_only": True}


def test_transfer_never_delivers_stale_output_from_an_earlier_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(settings(), "endless_root", tmp_path)
    manifest = tmp_path / "output/characters/fixture/manifest.json"
    manifest.parent.mkdir(parents=True)
    weight = manifest.parent / "adapter.bin"
    weight.write_bytes(b"fixture")
    manifest.write_text(json.dumps({"weights": [{"file": "fixture/adapter.bin"}]}))
    source = tmp_path / "source.png"
    output = tmp_path / "output/result/fixture_result/fixture_source_result.png"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"stale")
    monkeypatch.setattr("endlexx.images.asset_path", lambda _id: source)
    monkeypatch.setattr("endlexx.images.runtime_call", lambda *_: None)
    monkeypatch.setattr("endlexx.images.run_process", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="新结果"):
        transfer_image("fixture-task", "source", "fixture", lambda *_: None)
    assert output.read_bytes() == b"stale"
    manifest.write_text('{"weights":[]}')
    with pytest.raises(NeedsClarification, match="权重清单"):
        transfer_image("fixture-task", "source", "fixture", lambda *_: None)
