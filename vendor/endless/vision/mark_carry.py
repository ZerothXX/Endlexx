# -*- coding: utf-8 -*-
"""vision/mark_carry.py —— 非人物"加法"的**确定性承载**（把标志真实像素贴到目标表面）。

## 为什么要走确定性这条路（实测依据）

规格 §6 原本假设"概念权重能把标志画到目标主体的表面上"。实测（2026-10-05）：
把承载条件调到与训练同形、概念强度 1.45/2.5、重绘强度 0.62/0.80，
杯子图**改写比例 81~85%、平均差异 19.95~29.19**（支路确实在重绘），
但人工核对**依旧是素白杯子，没有任何标志图案**。
也就是说：概念权重学的是"角色裁片长什么样"，缺少"把图案画到物体表面"的承载语义，
**推理侧再怎么调都换不出可辨的标志**。

而用户的判据是"非人物加法仍一眼可辨为该角色独有标志"。
"一眼可辨"最可靠的实现不是让扩散去"想象"标志，而是把**用户参考图里标志的真实像素**
按目标表面的透视与光照约束贴上去：

    * 形状、颜色、比例 100% 来自真实素材 -> 必然"看得出是这个角色"；
    * 透视/明暗按目标表面适配 -> 满足 §6"adapt the real shape to the surface perspective"；
    * 全程确定性、可复核 -> 不引入新的"随机画错"风险。

扩散仍然负责**配色氛围**（这一条实测有效：杯身确实被染上角色配色）；
本模块只负责让**标志本身**清晰可辨。两者分工明确。

## 设计约束（都是为了不骗自己）

* 只使用**有真实像素证据**的素材（逐图裁剪或 mark_refs 参考图），
  没有素材的标志**不做**，并报告原因——不拿"通用花纹"顶替。
* 贴的位置来自**目标主体自身的掩版/框**，不复制原图的身体部位坐标（§6 明确禁止）。
* 明暗必须重新匹配：直接贴会像"贴纸"，所以按目标区域的平均明度与对比度做线性调整。
"""
import os

import numpy as np
import cv2
from PIL import Image

import config


# ---------------------------------------------------------------------------
# 素材：从裁剪里取出"标志本身"的像素 + 透明度
# ---------------------------------------------------------------------------
def load_asset(path, feather_ratio=0.10):
    """读一张标志裁剪，返回 (rgba ndarray, 说明)。

    **直接用裁剪矩形本身，不做背景抠图**（这一条是实测改过来的）：
    逐图裁剪是按标志的定位框裁出来的，框里**就是标志本身**，没有可分离的背景。
    早期版本按"四边中位色 = 背景色"去抠主体，结果把整块裁片当成主体、
    并配合连通块筛选把贴图弄成了**倒置/错位的方形块**——
    实测出图在杯口左上角贴出一块方形的角色图案，位置与形状都不对。
    既然素材本身就是标志的像素，就直接用它，只在边缘羽化以避免"硬方形贴纸"。
    """
    if not path or not os.path.isfile(path):
        return None, "素材文件不存在"
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, "素材无法解码"
    height, width = bgr.shape[:2]
    if min(height, width) < 8:
        return None, f"素材过小 {width}x{height}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # 羽化：边界 alpha 由 0 平滑升到 1，内部为 1。只羽化、不抠图，
    # 既保留标志的完整像素，又不留硬边框。
    fx = max(2, int(width * feather_ratio))
    fy = max(2, int(height * feather_ratio))
    ramp_x = np.clip(np.arange(width, dtype=np.float32) / fx, 0.0, 1.0)
    ramp_x = np.minimum(ramp_x, ramp_x[::-1])
    ramp_y = np.clip(np.arange(height, dtype=np.float32) / fy, 0.0, 1.0)
    ramp_y = np.minimum(ramp_y, ramp_y[::-1])
    alpha = ramp_y[:, None] * ramp_x[None, :]
    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    return rgba, f"素材 {width}x{height}（羽化 {fx}/{fy} 像素）"


# ---------------------------------------------------------------------------
# 适配：把素材缩放到目标尺寸，并按目标区域的明暗重新匹配
# ---------------------------------------------------------------------------
def match_lighting(rgba, target_lab, mask):
    """把素材的明暗与对比度对齐到目标区域，避免"贴纸感"。

    做法（线性、保留素材内部层次）：
        L' = mean_target + (L - mean_asset) * (std_target / std_asset)
    只在**目标区域内**统计，只作用于素材不透明处。
    """
    if target_lab is None or mask is None or not mask.any():
        return rgba
    asset_lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    alpha = rgba[:, :, 3] > 0
    if alpha.sum() < 16:
        return rgba
    asset_l = asset_lab[:, :, 0][alpha]
    target_l = target_lab[:, :, 0][mask]
    if target_l.size < 16:
        return rgba
    asset_std = float(asset_l.std()) or 1.0
    scale = float(np.clip(float(target_l.std()) / asset_std, 0.35, 2.5))
    shift = float(target_l.mean()) - float(asset_l.mean()) * scale
    out = rgba.copy()
    out_l = asset_lab[:, :, 0] * scale + shift
    out_l = np.clip(out_l, 0, 255)
    relab = asset_lab.copy()
    relab[:, :, 0] = out_l
    restored = cv2.cvtColor(relab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    out[:, :, :3] = np.where(alpha[:, :, None], restored, out[:, :, :3])
    return out


def fit_to_box(rgba, box, canvas_size, aspect=None):
    """把素材缩放到目标框内（保持自身长宽比），返回 (resized_rgba, paste_xy)。

    `box` 为画布像素坐标 (left, top, right, bottom)。
    """
    left, top, right, bottom = [int(round(v)) for v in box]
    target_w = max(8, right - left)
    target_h = max(8, bottom - top)
    asset_h, asset_w = rgba.shape[:2]
    if aspect:
        ratio = float(aspect)
    else:
        ratio = asset_w / max(1.0, asset_h)
    # 先按目标框定"最多能占多大"，再按素材自身比例收缩，避免拉伸变形
    width = min(target_w, int(round(target_h * ratio)))
    height = int(round(width / max(1e-6, ratio)))
    if height > target_h:
        height = target_h
        width = int(round(height * ratio))
    width = max(4, width)
    height = max(4, height)
    resized = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_AREA)
    centre_x = (left + right) // 2
    centre_y = (top + bottom) // 2
    paste_x = int(np.clip(centre_x - width // 2, 0, max(0, canvas_size[0] - width)))
    paste_y = int(np.clip(centre_y - height // 2, 0, max(0, canvas_size[1] - height)))
    return resized, (paste_x, paste_y)


def paste_rgba(canvas_rgb, rgba, position):
    """按 alpha 把 rgba 叠到 canvas 上（alpha 合成，不做硬贴）。"""
    x, y = position
    height, width = rgba.shape[:2]
    canvas_h, canvas_w = canvas_rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas_w, x + width), min(canvas_h, y + height)
    if x1 <= x0 or y1 <= y0:
        return canvas_rgb
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = (sub[:, :, 3:4].astype(np.float32) / 255.0)
    base = canvas_rgb[y0:y1, x0:x1].astype(np.float32)
    blended = base * (1 - alpha) + sub[:, :, :3].astype(np.float32) * alpha
    out = canvas_rgb.copy()
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def mark_blocked(block_mask, box, canvas_size, pad_ratio=0.10):
    """把一块区域记入"已占用"掩版（供后续底纹避开）。

    用途：徽记/可佩戴件先落笔，底纹后铺时必须避开它们，
    否则后铺的纹样会直接压住徽记（实测：杯子中央的金色图案被扇子盖住）。
    """
    if block_mask is None:
        block_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=bool)
    left, top, right, bottom = [int(round(v)) for v in box]
    pad_x = int((right - left) * pad_ratio)
    pad_y = int((bottom - top) * pad_ratio)
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(canvas_size[0], right + pad_x)
    bottom = min(canvas_size[1], bottom + pad_y)
    if right > left and bottom > top:
        block_mask[top:bottom, left:right] = True
    return block_mask


def tile_rgba(canvas_rgb, rgba, region_mask, spacing_ratio=None, max_count=24,
              block_mask=None, unit_scale=None):
    """重复纹样承载：按统一方向与间距平铺（§6 pattern 的承载方式）。

    实现在**目标区域内**按网格铺放，遇掩版外即停；间距由素材短边与
    `spacing_ratio` 决定（保证"看得出是同一个单元在重复"）。

    `block_mask`：**已经放过显眼件（徽记/可佩戴件）的位置**，铺纹样时必须避开。
    实测问题：杯子中央的金色徽记被后铺的扇子纹样直接压住，
    两者都看不清——"先放徽记"的顺序调整因此等于白做。
    """
    if region_mask is None or not region_mask.any():
        return canvas_rgb, 0
    if spacing_ratio is None:
        spacing_ratio = float(getattr(config, "MARK_CARRY_PATTERN_SPACING", 0.25))
    rows = np.where(region_mask.any(axis=1))[0]
    cols = np.where(region_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return canvas_rgb, 0
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    span_w, span_h = right - left, bottom - top
    asset_h, asset_w = rgba.shape[:2]
    # 单元边长：在区域宽度里铺 `MARK_CARRY_PATTERN_COLUMNS` 列
    # （实测 3 列时单元太小、整片像"贴纸墙"，而且格数一多就更容易压住徽记）。
    unit_w = max(20, int(span_w / max(2.0, float(
        getattr(config, "MARK_CARRY_PATTERN_COLUMNS", 2.0)))))
    # **按角色上的实际占比缩放**（用户指出的空间用法）：
    # `unit_scale` 来自 `mark_layout` 量出的 coverage —— 角色身上小面积点缀的花纹，
    # 落到器物上也应当是小面积重复，而不是放成大图占据视觉主体。
    # 实测踩过：金色图案在角色上只占 0.31% 面积（散布的小点缀），
    # 却因为固定尺寸被放到近 200 像素见方，成了整只杯子的视觉主体——
    # 正是用户说的"不会说占较大视觉形成视觉主体"。
    if unit_scale is not None:
        unit_w = max(20, int(unit_w * float(unit_scale)))
    unit_h = max(20, int(unit_w * asset_h / max(1, asset_w)))
    gap_x = max(6, int(unit_w * spacing_ratio))
    gap_y = max(6, int(unit_h * spacing_ratio))
    # **放不下就缩小重试**，而不是直接跳过。
    # 实测：`白色樱花图案` 因为单元偏大被判"在可见区域内铺不下"而整条跳过，
    # 于是杯子上只剩一枚徽记——看着就像"只贴了一张图"。
    rows_fit = max(1, int((span_h + gap_y) // max(1, unit_h + gap_y)))
    cols_fit = max(1, int((span_w + gap_x) // max(1, unit_w + gap_x)))
    if rows_fit * cols_fit < 2:
        shrink = max(0.35, min(
            (span_w / max(1.0, float(getattr(config, "MARK_CARRY_PATTERN_MIN_COLUMNS",
                                             2.0)))) / max(1, unit_w),
            (span_h / 2.0) / max(1, unit_h)))
        unit_w = max(18, int(unit_w * shrink))
        unit_h = max(18, int(unit_w * asset_h / max(1, asset_w)))
        gap_x = max(4, int(unit_w * spacing_ratio))
        gap_y = max(4, int(unit_h * spacing_ratio))
    unit = cv2.resize(rgba, (unit_w, unit_h), interpolation=cv2.INTER_AREA)
    # 先排一遍网格，只保留**真正放得下且未被显眼件占用**的格子；
    # 再把这些格子沿纵向**均匀铺满可用高度**。
    # 为什么需要（实测）：排除口沿之后区域变矮，固定间距会把最后一行切掉，
    # 出现"上面两个、下面空一大块"的不平衡观感。
    #
    # **每个候选格子被拒的原因都要记下来**（方向修整1 §5 P0 第 2 条）：
    # 原先只返回一个放置次数，于是"可用格子很多却铺放 0 次"只能得到
    # 一句笼统的"铺不下"，无法判断是尺寸、区域形状还是被显眼件挡住。
    slots = []
    reject_outside = 0     # 中心点不在可承载区域内
    reject_blocked = 0     # 被已放置的显眼件占用
    reject_offgrid = 0     # 网格装不下整格（贴边被切）
    y = top
    while y + unit_h <= bottom and len(slots) < max_count:
        x = left
        while x + unit_w <= right and len(slots) < max_count:
            centre = (x + unit_w // 2, y + unit_h // 2)
            blocked = False
            if block_mask is not None:
                block = block_mask[y:y + unit_h, x:x + unit_w]
                if block.size and block.mean() > float(
                        getattr(config, "MARK_CARRY_PATTERN_BLOCK_RATIO", 0.25)):
                    blocked = True
            inside = bool(region_mask[min(centre[1], region_mask.shape[0] - 1),
                                      min(centre[0], region_mask.shape[1] - 1)])
            if blocked:
                reject_blocked += 1
            elif not inside:
                reject_outside += 1
            else:
                slots.append((x, y))
            x += unit_w + gap_x
        y += unit_h + gap_y
    # 贴边被切掉的格数（网格循环条件本身排除的）
    grid_rows = max(0, int((span_h - unit_h) // max(1, unit_h + gap_y)) + 1)
    grid_cols = max(0, int((span_w - unit_w) // max(1, unit_w + gap_x)) + 1)
    reject_offgrid = max(0, grid_rows * grid_cols - (len(slots) + reject_blocked
                                                     + reject_outside))
    reject_stats = {"grid": int(grid_rows * grid_cols), "placed": 0,
                    "outside_region": reject_outside, "blocked": reject_blocked,
                    "offgrid": reject_offgrid,
                    "unit": [int(unit_w), int(unit_h)],
                    "span": [int(span_w), int(span_h)]}
    if not slots:
        # 把拒绝原因回传（第 2 个返回值仍是放置次数，调用方不用改签名）
        tile_rgba.last_reject = reject_stats
        return canvas_rgb, 0
    rows_used = sorted({slot[1] for slot in slots})
    if len(rows_used) > 1:
        span = rows_used[-1] - rows_used[0]
        step = span / (len(rows_used) - 1)
        remap = {old: int(round(top + index * step))
                 for index, old in enumerate(rows_used)}
        slots = [(x, remap[y]) for x, y in slots]
    placed = 0
    for x, y in slots:
        canvas_rgb = paste_rgba(canvas_rgb, unit, (x, y))
        placed += 1
    reject_stats["placed"] = placed
    tile_rgba.last_reject = reject_stats
    return canvas_rgb, placed


def place_scattered(canvas_rgb, rgba, region_mask, count=3, max_ratio=0.34,
                    block_mask=None, slot=0, slots=3, size_jitter=0.18):
    """把标志**零星放置**若干处（§6 里 `scattered` 用法，例如樱花飘落）。

    为什么要单独有这条承载方式（用户评价"效果很差"的直接原因之一）：
    原先所有花纹类标志都走 `tile_rgba` 做**规则网格平铺**，实测两个标志各铺 6 次、
    器物表面 94% 像素被改写，观感变成"包装纸"或"重新刷过的杯子"，
    而不是"这件器物属于这个角色"。

    做法（对应角色身上"零星覆盖"的实际用法）：
      * 单元尺寸取区域短边的 `max_ratio` 以内（**封顶**：既避免素材分辨率低时放大发糊，
        也避免大素材糊满整个器物）；
      * 位置按序号在区域内分散，带**确定性**抖动（不用随机数，保证同一输入出图可复现）；
      * 各处尺寸略有差异（`size_jitter`），更像自然分布而不是模板；
      * 仍避开 `block_mask` 已占用的位置。

    返回 (结果图, 实际放置次数)。
    """
    if region_mask is None or not region_mask.any() or rgba is None:
        return canvas_rgb, 0
    rows = np.where(region_mask.any(axis=1))[0]
    cols = np.where(region_mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return canvas_rgb, 0
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    span_w, span_h = right - left, bottom - top
    short = min(span_w, span_h)

    unit = max(24, int(short * float(max_ratio)))
    asset_h, asset_w = rgba.shape[:2]
    placed = 0
    last_audit = {}
    boxes = []
    count = max(1, int(count))
    for index in range(count):
        factor = 1.0 + (float(size_jitter) * (1 if index % 2 == 0 else -1)
                        * (0.5 + 0.5 * (index / max(1, count))))
        side = max(22, int(unit * factor))
        scale = side / max(1, max(asset_w, asset_h))
        width = max(16, int(asset_w * scale))
        height = max(16, int(asset_h * scale))
        piece = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_AREA)
        fraction = (index + 1) / (count + 1) if count > 1 else 0.5
        centre_y = top + int(span_h * fraction)
        offset = int(span_w * 0.14) * (1 if index % 2 == 0 else -1)
        centre_x = (left + right) // 2 + (0 if count == 1 else offset)
        x = int(np.clip(centre_x - width // 2, left, max(left, right - width)))
        y = int(np.clip(centre_y - height // 2, top, max(top, bottom - height)))
        if block_mask is not None:
            block = block_mask[y:y + height, x:x + width]
            if block.size and block.mean() > float(
                    getattr(config, "MARK_CARRY_PATTERN_BLOCK_RATIO", 0.25)):
                continue
        cy = min(y + height // 2, region_mask.shape[0] - 1)
        cx = min(x + width // 2, region_mask.shape[1] - 1)
        if not region_mask[cy, cx]:
            continue
        canvas_rgb, last_audit = paste_or_fuse(canvas_rgb, piece, (x, y), region_mask)
        if block_mask is not None:
            block_mask = mark_blocked(block_mask, (x, y, x + width, y + height),
                                      canvas_rgb.shape[1::-1], pad_ratio=0.06)
        placed += 1
        boxes.append([int(x), int(y), int(x + width), int(y + height)])
    place_scattered.last_audit = last_audit
    place_scattered.last_boxes = boxes
    return canvas_rgb, placed


def paste_or_fuse(canvas_rgb, rgba, position, region_mask=None):
    """把一块素材落到画布上：**优先承印融合**，融合不可用时退回 alpha 合成。

    统一放在这里，是为了让 `place_scattered` / `band_rgba` 这类"多次落笔"的承载方式
    也能享受同一套融合处理，而不是只有徽记那一条路走融合
    （否则同一件器物上会出现"部分图案是印上去的、部分是贴上去的"两种观感）。

    返回 (画布, 审计)。审计里的 `applied=False` 表示**确实退回了旧做法**，
    并写明原因——不静默降级。
    """
    try:
        import config as _config
        use_fusion = bool(getattr(_config, "MARK_FUSION_ENABLED", True))
    except Exception:                     # noqa: BLE001
        use_fusion = False
    if not use_fusion:
        return paste_rgba(canvas_rgb, rgba, position), {
            "applied": False, "reason": "MARK_FUSION_ENABLED=False"}
    try:
        from vision import surface_fusion
    except Exception as exc:              # noqa: BLE001
        return paste_rgba(canvas_rgb, rgba, position), {
            "applied": False, "reason": f"融合模块不可用: {type(exc).__name__}"}
    x, y = int(position[0]), int(position[1])
    height, width = rgba.shape[:2]
    fused, audit = surface_fusion.fuse_onto_surface(
        rgba, canvas_rgb, (x, y, x + width, y + height), region_mask=region_mask)
    if audit.get("applied"):
        return fused, {k: v for k, v in audit.items() if k != "applied"}
    return paste_rgba(canvas_rgb, rgba, position), {
        "applied": False, "reason": audit.get("reason")}


def band_rgba(canvas_rgb, rgba, region_mask):
    """边饰承载：沿目标区域的下沿铺一条重复带（§6 trim 的承载方式）。"""
    if region_mask is None or not region_mask.any():
        return canvas_rgb, 0
    rows = np.where(region_mask.any(axis=1))[0]
    cols = np.where(region_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return canvas_rgb, 0
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    asset_h, asset_w = rgba.shape[:2]
    unit_h = max(10, int((bottom - top) * 0.10))
    unit_w = max(10, int(unit_h * asset_w / max(1, asset_h)))
    unit = cv2.resize(rgba, (unit_w, unit_h), interpolation=cv2.INTER_AREA)
    # 贴着下沿（区域内的最低一行）铺
    placed, x = 0, left
    y = max(top, bottom - unit_h)
    last_audit = {}
    while x + unit_w <= right:
        column = region_mask[y:y + unit_h, x:x + unit_w]
        if column.any():
            canvas_rgb, last_audit = paste_or_fuse(canvas_rgb, unit, (x, y), region_mask)
            placed += 1
        x += unit_w
    band_rgba.last_audit = last_audit
    return canvas_rgb, placed
