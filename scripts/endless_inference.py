"""Invoke Endless's formal entry with an application-owned memory policy.

Research sources, weights, resolution, sampling steps and algorithms stay untouched.
Only Diffusers component placement and attention implementation are adapted in memory.
"""

import argparse
import importlib
import os
import runpy
import sys
from pathlib import Path


def configure_offline_loaders(managed_config):
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    pipeline_loader = StableDiffusionXLPipeline.from_single_file
    vae_loader = AutoencoderKL.from_single_file

    def load_pipeline(*args, **kwargs):
        kwargs.setdefault("config", str(managed_config))
        kwargs.setdefault("local_files_only", True)
        return pipeline_loader(*args, **kwargs)

    def load_vae(*args, **kwargs):
        # Diffusers also calls this loader while assembling the pipeline,
        # already supplying config/subfolder. Preserve those arguments.
        kwargs.setdefault("config", str(managed_config))
        kwargs.setdefault("subfolder", "vae")
        kwargs.setdefault("local_files_only", True)
        return vae_loader(*args, **kwargs)

    StableDiffusionXLPipeline.from_single_file = load_pipeline
    AutoencoderKL.from_single_file = load_vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runtime", choices=["balanced", "research"], default="balanced")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    # Managed installations include the SDXL configs/tokenizers, so first inference
    # does not need an implicit download into a global Hugging Face cache.
    managed_config = (
        Path(os.environ.get("ENDLEXX_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data")))
        / "bundle/sdxl-config"
    )
    if managed_config.is_dir():
        configure_offline_loaders(managed_config)
    if args.runtime == "balanced":
        config = importlib.import_module("config")
        models = importlib.import_module("model_utils")
        config.ATTENTION_SLICING = False

        def place(pipe, components):
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            pipe.disable_attention_slicing()  # Restores PyTorch SDPA on this pinned Diffusers build.
            pipe.enable_model_cpu_offload(gpu_id=0, device=components["device"])
            pipe.set_progress_bar_config(disable=True)

        models._place = place
        print(
            "[Endlexx] runtime=balanced: SDPA + model CPU offload; research generation parameters preserved",
            flush=True,
        )
    sys.argv = [str(root / "test.py"), "--character", args.character, "--input", str(args.input.resolve())]
    runpy.run_path(str(root / "test.py"), run_name="__main__")


if __name__ == "__main__":
    main()
