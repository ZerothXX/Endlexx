# -*- coding: utf-8 -*-
"""image_utils.py —— 图像工具（唯一实现）

图片读取（含 EXIF 方向修正）、等比缩放、保存、Canny 边缘提取、
输出目录初始化。只依赖 PIL / opencv / numpy，不 import torch。
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

import config


def load_image(path: str) -> Image.Image:
    """读取图片并返回 RGB 模式的 PIL.Image；读取失败抛出带路径的异常。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[image_utils] 图片文件不存在: {path}")
    try:
        image = ImageOps.exif_transpose(Image.open(path))
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")
    except Exception as exc:
        raise IOError(f"[image_utils] 图片读取失败: {path}，原因: {exc}") from exc


def resize_fit(img: Image.Image, max_side: int, max_long: int = None,
               min_dim: int = 256) -> Image.Image:
    """等比缩放（不裁剪、不拉伸）：短边对齐 max_side，长边不超过 max_long。

    保留原图完整构图，同时让主体占据足够像素（早期"长边=640"会让全身图里的
    面孔只剩几十像素，五官被拖垮）。尺寸取整到 8 的倍数。
    """
    max_long = int(config.INFERENCE_MAX_LONG if max_long is None else max_long)
    if max_side < 8 or max_long < 8:
        raise ValueError(f"[image_utils] max_side/max_long 必须 >= 8，收到 {max_side}/{max_long}")
    width, height = img.size
    scale = max_side / min(width, height)
    if max(width * scale, height * scale) > max_long:
        scale = max_long / max(width, height)
    new_w = max(8, int(width * scale / 8) * 8)
    new_h = max(8, int(height * scale / 8) * 8)
    if min(new_w, new_h) < min_dim:
        raise ValueError(
            "输入长宽比在给定尺寸限制下无法在不拉伸的前提下适配；"
            "请提高 INFERENCE_MAX_LONG 或改用长宽比更温和的输入。")
    if (new_w, new_h) != (width, height):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def save_image(img: Image.Image, path: str) -> str:
    """保存为 PNG（目录自动创建）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        img.save(path, format="PNG")
    except Exception as exc:
        raise IOError(f"[image_utils] 图片保存失败: {path}，原因: {exc}") from exc
    return path


def extract_canny(img: Image.Image, low: int = None, high: int = None) -> Image.Image:
    """Canny 边缘图，返回黑底白边的三通道 RGB（ControlNet 结构条件）。"""
    low = config.CANNY_LOW if low is None else low
    high = config.CANNY_HIGH if high is None else high
    edges = cv2.Canny(np.array(img.convert("L")), low, high)
    return Image.fromarray(edges).convert("RGB")


def create_dirs() -> None:
    """创建运行所需目录（幂等）。"""
    for path in (config.INPUT_DIR, config.TRAIN_MODELS_DIR, config.CHARACTERS_DIR,
                 config.SEMANTIC_DIR, config.RESULT_DIR, config.LOGS_DIR,
                 config.CURVES_DIR, config.MANIFEST_DIR, config.PROFILE_DIR,
                 os.path.dirname(config.REGISTRY_DB_PATH)):
        os.makedirs(path, exist_ok=True)
