"""Image execution boundary. The application supplies an automatically generated mask."""

import argparse
import json
import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"


def repair(args):
    import numpy as np
    import torch
    from PIL import Image, ImageFilter, ImageOps

    source = Image.open(args.input).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    if mask.size != source.size:
        raise ValueError("Mask and source coordinate systems differ")
    if not np.any(np.asarray(mask) > 127):
        raise ValueError("Empty mask")
    size = source.size
    canvas = source.copy()
    canvas.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    small_mask = mask.resize(canvas.size, Image.Resampling.NEAREST)
    pad_w, pad_h = (-canvas.width) % 8, (-canvas.height) % 8
    image_array = np.asarray(canvas).astype(np.float32) / 255
    mask_array = (np.asarray(small_mask) > 127).astype(np.float32)
    image_array = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask_array = np.pad(mask_array, ((0, pad_h), (0, pad_w)), mode="constant")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_t = torch.from_numpy(image_array.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_array.copy()).unsqueeze(0).unsqueeze(0).to(device)
    model = torch.jit.load(args.weights, map_location=device).eval()
    with torch.inference_mode():
        output = model(image_t * (1 - mask_t), mask_t)
    if isinstance(output, dict):
        output = output.get("inpainted", output.get("image"))
    image = (output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    predicted = (
        Image.fromarray(image)
        .crop((0, 0, canvas.width, canvas.height))
        .resize(size, Image.Resampling.LANCZOS)
    )
    # Only an explicitly bounded two-pixel transition may change outside the raw mask.
    support = mask.filter(ImageFilter.MaxFilter(5))
    alpha = ImageOps.autocontrast(mask.filter(ImageFilter.GaussianBlur(1)))
    alpha_array = np.asarray(alpha).copy()
    alpha_array[np.asarray(support) == 0] = 0
    alpha_array[np.asarray(mask) > 127] = 255
    result = Image.composite(predicted, source, Image.fromarray(alpha_array))
    result.save(args.output)
    outside = np.asarray(support) == 0
    delta = np.abs(np.asarray(result).astype(int) - np.asarray(source).astype(int))
    metrics = {
        "backend": "LaMa",
        "device": device,
        "mask_fraction": float((np.asarray(mask) > 127).mean()),
        "protected_pixels_changed": int(np.any(delta > 0, axis=2)[outside].sum()),
        "resolution": list(size),
        "ppo": False,
    }
    Path(args.report).write_text(json.dumps(metrics), encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    repair(parser.parse_args())
