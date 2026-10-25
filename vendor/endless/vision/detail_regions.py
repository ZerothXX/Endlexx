# -*- coding: utf-8 -*-
"""vision/detail_regions.py —— 概念样本的区域定位（自动、通用）

职责：为"概念分支"找出角色图里值得裁剪的细节区域（配饰、图案、花纹、服饰局部），
产出归一化框，供 dataset.build_concept_entries 生成真实裁剪样本。

两条来源，都不写死角色：
  1. 角色档案里的 visual_features（VLM 观察到的特征 + 所属区域 + 来源图片）；
  2. 区域定位（VLM 框 + SAM 掩版）给出的像素级范围——掩版能拿到时用它的外接框，
     比纯文字框准确。

产出 {rel_path: [{"box": (l,t,r,b), "kind": str, "name": str}]}。
没有任何可信来源时返回空，绝不臆造裁剪位置。
"""
from pathlib import Path

import config
from character import profile as profile_module

# 档案区域名 -> 裁剪用途标签（仅用于标注，不参与定位）
_REGION_KIND = {
    "head": "accessory", "chest": "pattern", "waist": "pattern", "torso": "garment",
    "arms": "garment", "legs": "garment", "feet": "garment", "surface": "pattern",
    "whole": "detail",
}


def locate_detail_regions(folder, profile=None, image_size=None, masks_by_region=None):
    """按档案特征给出细节区域；同一图上去重。

    若给了 masks_by_region（该图已由 SAM 得到的像素级区域掩版），优先用掩版的外接框，
    因为它比"按区域名假设的构图位置"准确得多。
    """
    profile = profile or profile_module.load_profile(folder)
    if not profile:
        return {}, ["没有角色档案，无法定位细节区域"]
    if masks_by_region and image_size:
        from_masks = boxes_from_masks(masks_by_region, image_size)
        if from_masks:
            return {"__per_image__": _dedupe(from_masks)}, []
    features = profile.get("visual_features") or []
    if not features:
        return {}, ["档案里没有 visual_features，无法定位细节区域"]
    boxes = _boxes_from_profile(features)
    if not boxes:
        return {}, ["特征里没有可用的区域信息"]
    detections = {rel_path: _dedupe(items) for rel_path, items in boxes.items()}
    return detections, []


def _boxes_from_profile(features):
    """由特征区域推出框：按区域名取画面中的合理位置，再按需要收缩。

    这些是**构图上的保守假设**（例如腰带大致在画面中部），不是身份臆测；
    只在档案没有像素级证据时使用，并且尺寸取得克制，避免裁进大片背景。
    """
    region_box = {
        "head": (0.18, 0.02, 0.82, 0.34),
        "chest": (0.22, 0.22, 0.78, 0.52),
        "waist": (0.22, 0.34, 0.78, 0.64),
        "torso": (0.18, 0.18, 0.82, 0.68),
        "arms": (0.05, 0.22, 0.95, 0.66),
        "legs": (0.22, 0.52, 0.78, 0.92),
        "feet": (0.22, 0.78, 0.78, 0.99),
        "surface": (0.20, 0.20, 0.80, 0.80),
        "whole": (0.10, 0.05, 0.90, 0.95),
    }
    boxes = {}
    for feature in features:
        region = str(feature.get("region") or "whole").lower()
        if region not in region_box:
            continue
        kind = str(feature.get("kind") or _REGION_KIND.get(region, "detail"))
        name = str(feature.get("name") or region)
        for image in feature.get("source_images") or []:
            rel_path = Path(str(image)).name
            boxes.setdefault(rel_path, []).append(
                {"box": region_box[region], "kind": kind, "name": name})
    return boxes


def _dedupe(items, tolerance=0.05):
    """同一图上去掉几乎相同的框（同一区域被多个特征提到时只留一个）。"""
    kept = []
    for item in items:
        box = item["box"]
        duplicate = False
        for existing in kept:
            other = existing["box"]
            if all(abs(a - b) <= tolerance for a, b in zip(box, other)):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def refine_object_box(image, box, min_gain=1.25):
    """用"边缘密度"把物体框挪到真正有物体的位置。

    实测问题：视觉模型给的"物体表面"框会漂移——同一张杯子图，有时框住杯子、
    有时框到右下角的桌面。物体表面与桌面的区别是**边缘密度**（杯口、把手、高光
    都带着边缘；桌面是平滑的）。因此这里在粗网格上滑窗，找边缘密度最高的位置，
    只有明显更优（超过 min_gain 倍）才采用，否则保留原框。

    判据只用图像自身的统计量，不含任何物体类别假设，换物体同样成立。
    """
    import numpy as _np
    if image is None or not box:
        return box, "无框可校正"
    grey = _np.asarray(image.convert("L")).astype(_np.float32)
    if grey.size < 256:
        return box, "图像过小"
    gx = _np.zeros_like(grey)
    gy = _np.zeros_like(grey)
    gx[:, 1:-1] = grey[:, 2:] - grey[:, :-2]
    gy[1:-1, :] = grey[2:, :] - grey[:-2, :]
    edges = _np.hypot(gx, gy)
    height, width = grey.shape

    def density(candidate):
        left, top, right, bottom = (int(candidate[0] * width), int(candidate[1] * height),
                                    int(candidate[2] * width), int(candidate[3] * height))
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        if right - left < 8 or bottom - top < 8:
            return 0.0
        return float(edges[top:bottom, left:left + (right - left)].mean())

    current = density(box)
    if current <= 0:
        return box, "原框无内容"
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    best, best_score = box, current
    for left in _np.arange(0.0, max(0.01, 1.0 - box_w), 0.05):
        for top in _np.arange(0.0, max(0.01, 1.0 - box_h), 0.05):
            candidate = (float(left), float(top), float(left + box_w), float(top + box_h))
            score = density(candidate)
            if score > best_score:
                best, best_score = candidate, score
    if best is box or best_score < current * min_gain:
        return box, f"原框已足够（边缘密度 {current:.1f}）"
    print(f"[detail_regions] 物体框已按边缘密度校正: 密度 {current:.1f} -> {best_score:.1f}")
    return best, f"按边缘密度校正（{current:.1f} -> {best_score:.1f}）"


def boxes_from_masks(masks_by_region, size):
    """把像素级掩版转成归一化外接框（掩版比文字框准，能拿到就用它）。"""
    detections = []
    width, height = size
    for region, mask in (masks_by_region or {}).items():
        if mask is None:
            continue
        bbox = mask.convert("L").point(lambda v: 255 if v >= 128 else 0).getbbox()
        if not bbox:
            continue
        left, top, right, bottom = bbox
        detections.append({
            "box": (left / width, top / height, right / width, bottom / height),
            "kind": _REGION_KIND.get(region, "detail"),
            "name": region,
        })
    return detections
