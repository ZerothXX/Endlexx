# -*- coding: utf-8 -*-
"""vision/mark_shape.py —— 从标志裁剪里取出**标志本身的形状**（只保留轮廓内的像素）。

## 解决什么问题（实测依据）

第七轮把标志贴上物体后，观感是"贴了一排小照片"：因为直接用了整个裁剪矩形，
每个重复单元都是一块**带背景的方形照片**（黑底 + 衣服图案）。
用户要的是"看得出的角色独有标志"，而不是一叠方形贴纸。

## 做法（两条路，按可靠性排序）

1. **SAM 分割**（首选）：裁剪里标志通常是中心最显眼的那一块，
   用"中心点 + 覆盖大部分画面的框"提示 SAM，取最大候选掩版，
   再与"裁剪矩形"取交集并做形态学清理。SAM 对"这是一个独立物件"分得很好，
   实测能把胸甲/樱花/扇子这类徽记从裁片里分出来。
2. **回退：颜色离群 + 最大连通块**（SAM 不可用时）：
   以裁剪四边颜色为背景估计，取明显偏离背景的最大连通块。
   这条在"标志占满裁剪"时会退化成整块（那时也就没有可分离的背景）。

两条路都只输出 **0/255 的 alpha 掩版**，并与素材像素一起构成新的 rg 素材。
判据只用图像自身统计量，不含角色/题材假设。
"""
import os

import numpy as np
import cv2
from PIL import Image

import config


def _largest_component(binary, keep_ratio=0.12):
    """保留面积达到最大块 keep_ratio 的所有连通块（不要只留一块：
    徽记常有分离的部件，例如花心与花瓣、扇骨与扇面）。"""
    if binary is None or not binary.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    keep = np.zeros_like(binary, dtype=bool)
    for index, area in enumerate(areas, start=1):
        if area >= largest * keep_ratio:
            keep |= (labels == index)
    return keep


def _clean(mask):
    """形态学清理：闭运算补内部小洞 -> 开运算去毛刺 -> 补内部孔洞。"""
    array = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    array = cv2.morphologyEx(array, cv2.MORPH_CLOSE, kernel, iterations=2)
    array = cv2.morphologyEx(array, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = array > 128
    # 补孔：先做一次填充，再与"只在内部"的差集相加
    flood = binary.astype(np.uint8).copy()
    height, width = flood.shape
    canvas = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, canvas, (0, 0), 1)
    holes = (flood == 0)
    return binary | holes


def _colour_fallback(bgr):
    """回退判据：以裁剪四边颜色为背景估计，取明显偏离背景的最大连通块。"""
    height, width = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    band = max(1, min(height, width) // 12)
    border = np.concatenate([
        lab[:band, :, :].reshape(-1, 3), lab[-band:, :, :].reshape(-1, 3),
        lab[:, :band, :].reshape(-1, 3), lab[:, -band:, :].reshape(-1, 3)])
    background = np.median(border, axis=0)
    distance = np.sqrt(((lab - background) ** 2).sum(axis=2))
    spread = float(np.percentile(distance, 95)) or 1.0
    mask = distance > spread * float(getattr(config, "MARK_SHAPE_COLOUR_RATIO", 0.45))
    mask = _largest_component(mask)
    return mask


def _background_alpha(bgr):
    """按"与主导背景色的距离"生成 alpha：**背景透掉，标志留下**。

    为什么需要这一条（实测）：`m19`（白色樱花图案）的裁剪是**白底 + 白色花瓣 +
    棕色描边**——花瓣内部与背景同为白色，光看颜色没有边界；而"把白底也贴上去"
    就会在深色器物上糊出一块白方块。反过来，只要把**主导背景色**的像素透掉，
    留下的就是花瓣描边与花心，贴到器物上正好是"一个樱花轮廓"，这才是用户要的。

    判据只用裁剪自身统计量：
      1. 取四边颜色的中位数作为背景色估计；
      2. 距离小于容差的像素判为背景（alpha=0）；
      3. 中间地带做平滑过渡，避免锯齿；
      4. 只保留最大连通块（避免把零散的同色小块一起贴上去）。
    """
    height, width = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    band = max(1, min(height, width) // 10)
    border = np.concatenate([
        lab[:band, :, :].reshape(-1, 3), lab[-band:, :, :].reshape(-1, 3),
        lab[:, :band, :].reshape(-1, 3), lab[:, -band:, :].reshape(-1, 3)])
    background = np.median(border, axis=0)
    distance = np.sqrt(((lab - background) ** 2).sum(axis=2))
    near = float(getattr(config, "MARK_SHAPE_BG_TOLERANCE", 12.0))
    far = near * float(getattr(config, "MARK_SHAPE_BG_RAMP", 2.2))
    alpha = np.clip((distance - near) / max(1.0, far - near), 0.0, 1.0)
    foreground = alpha > 0.5
    if foreground.sum() < 16:
        return None
    kept = _largest_component(foreground, keep_ratio=0.25)
    if kept is None or kept.sum() < 16:
        return None
    # 背景判定为"主导色"，因此只保留被判为前景的连通块
    return (alpha * kept).astype(np.float32)


def extract_shape(crop_path, sam=None, use_sam=None):
    """返回 (rgba, 审计 dict)。rgba 的第 4 通道是"标志轮廓"alpha。

    `sam=None` 时自动走颜色回退；传 (model, processor) 则先用 SAM。
    审计里写清用了哪条路、覆盖率多少，便于复核与回归测试。
    """
    audit = {"crop": os.path.relpath(crop_path, config.PROJECT_ROOT)
             if os.path.isabs(crop_path) else crop_path}
    bgr = cv2.imread(crop_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, {**audit, "ok": False, "reason": "素材无法解码"}
    height, width = bgr.shape[:2]
    if min(height, width) < 8:
        return None, {**audit, "ok": False, "reason": f"素材过小 {width}x{height}"}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if use_sam is None:
        use_sam = bool(getattr(config, "MARK_SHAPE_USE_SAM", True))
    mask = None
    method = ""
    if use_sam and sam is not None:
        try:
            from vision.segmentation import sam_point_maskbox
            # 正例点取裁剪中心（标志通常居中），框取整张裁剪并略向内收，
            # 让 SAM 在"这一块东西"的范围里选，而不是把整幅背景也当对象。
            points = [(width * 0.5, height * 0.5)]
            box = (width * 0.02, height * 0.02, width * 0.98, height * 0.98)
            # **取全部候选自己挑**：单点提示下 SAM 的三个候选常分别是
            # "一小块碎片 / 合适的一块 / 整幅画面"（实测覆盖 0.2% 与 100% 都出现过），
            # 而最高 IoU 分并不总落在"合适"那个上。标志应当占裁剪的一个合理比例，
            # 因此按"覆盖率离期望区间最近"来选，而不是按分数选。
            candidates = sam_point_maskbox(
                Image.fromarray(rgb), sam, points, labels=[1], box=box,
                return_all=True)
            chosen, best_cost = None, None
            low = float(getattr(config, "MARK_SHAPE_MIN_COVERAGE", 0.12))
            high = float(getattr(config, "MARK_SHAPE_MAX_COVERAGE", 0.92))
            for array, score in (candidates or []):
                coverage = float(array.mean())
                if coverage < 0.004:
                    continue                      # 碎片，直接排除
                if coverage < low:
                    cost = (low - coverage) + 1.0
                elif coverage > high:
                    cost = (coverage - high) + 1.0
                else:
                    cost = 0.0
                cost -= 0.15 * float(score)        # 同等合理时偏好分数高的
                if best_cost is None or cost < best_cost:
                    chosen, best_cost = array, cost
            if chosen is not None:
                coverage = float(chosen.mean())
                if low <= coverage <= high:
                    mask, method = chosen, "sam"
                else:
                    audit["sam_note"] = (
                        f"SAM 候选覆盖率 {coverage:.1%} 不在 "
                        f"[{low:.0%},{high:.0%}] 内，退回颜色判据")
            else:
                audit["sam_note"] = "SAM 没有给出可用候选，退回颜色判据"
        except Exception as exc:            # noqa: BLE001
            audit["sam_note"] = f"SAM 不可用: {type(exc).__name__}: {exc}"
    if mask is None:
        # SAM 无可用候选时，先试**背景透传**（对"扁平色 + 单一背景"的标志最有效），
        # 再退回颜色离群判据。
        bg_alpha = _background_alpha(bgr)
        if bg_alpha is not None and bg_alpha.mean() < float(
                getattr(config, "MARK_SHAPE_MAX_COVERAGE", 0.92)):
            coverage = float((bg_alpha > 0.5).mean())
            alpha_only = (bg_alpha * 255).astype(np.uint8)
            feather_now = float(getattr(config, "MARK_SHAPE_FEATHER", 1.6))
            if feather_now > 0:
                alpha_only = cv2.GaussianBlur(alpha_only, (0, 0), feather_now)
            audit.update({"ok": True, "method": "background",
                          "coverage": round(coverage, 4),
                          "size": [width, height]})
            return np.dstack([rgb, np.clip(alpha_only, 0, 255).astype(np.uint8)]), audit
        mask = _colour_fallback(bgr)
        method = "colour"
    if mask is None or mask.sum() < 16:
        return None, {**audit, "ok": False, "reason": "没有可分离的标志形状"}

    mask = _clean(mask)
    coverage = float(mask.mean())
    if coverage > float(getattr(config, "MARK_SHAPE_MAX_COVERAGE", 0.92)):
        # 整块都是标志（裁剪本来就很紧）：保留矩形，只做羽化——这不算失败，
        # 但要如实标注，好让"贴纸感"的来源可追溯。
        method = f"{method}+full"
    alpha = (mask.astype(np.float32) * 255.0)

    # 羽化轮廓边缘：硬切边同样是"贴纸感"的来源
    feather = float(getattr(config, "MARK_SHAPE_FEATHER", 1.6))
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    audit.update({"ok": True, "method": method, "coverage": round(coverage, 4),
                  "size": [width, height]})
    return rgba, audit


def extract_library(folder, observations=None, sam=None, limit=None):
    """为库里每个标志挑一张素材并抽出形状，返回 {mark_id: (rgba, audit)}。

    素材选择与承载侧保持一致：优先该标志**最大的逐图真实裁剪**。
    """
    import image_utils
    from character import marks as marks_module

    observations = observations if observations is not None else \
        marks_module.load_observations(folder)
    library = marks_module.load_library(folder)
    if not library:
        return {}
    best = {}
    for row in observations:
        crop = row.get("crop_path")
        if not crop:
            continue
        path = os.path.join(config.PROJECT_ROOT, crop)
        if not os.path.isfile(path):
            continue
        try:
            size = image_utils.load_image(path).size
        except Exception:
            continue
        area = size[0] * size[1]
        if area > best.get(row["mark_id"], (0, None))[0]:
            best[row["mark_id"]] = (area, path)
    shapes = {}
    for index, (mark_id, (_area, path)) in enumerate(sorted(best.items())):
        if limit and index >= limit:
            break
        rgba, audit = extract_shape(path, sam=sam)
        if rgba is not None:
            shapes[mark_id] = (rgba, audit)
    return shapes
