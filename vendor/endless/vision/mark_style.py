# -*- coding: utf-8 -*-
"""vision/mark_style.py —— 从角色**自己的图像证据**里提取"风格特征"，用于非人物承载。

## 为什么换掉"贴标志图"的做法（用户反馈 + 实测）

上一版把 `mark_refs` 里标志的真实像素裁片直接贴到器物上。
用户评价："看起来就只是简单的单独标志图的粘贴，根本没有学习到这种角色的特征并运用上"。
这个判断是准确的：

* 贴上去的是**一张图的像素**，与器物自身的受光、材质、透视无关，
  观感必然是"贴纸"；
* 它没有用到"**这个角色的颜色关系与材质语言**"——而这才是"一眼看出是谁"的来源；
* 器物本身完全没变，等于加法只加了一个外来图层。

## 本模块的做法：把"角色的颜色与明暗关系"迁移到目标表面

判据只用**角色自己的图像证据**（逐标志裁剪 + 参考图），不写死任何角色：

1. **调色板**：对角色裁剪的像素做 k-means，取主色（按占比排序）。
   这是"这个角色的颜色关系"，而不是某一处配色的名字。
   实测角色档案里的配色是 `blue / pink / gold` 这类**色名**，
   而真正的角色特征在**具体色值与占比**里（深蓝为主、金为点缀、粉作小面积提亮）。
2. **色相映射**：把目标表面像素的**明度层次完整保留**，
   只把色相/饱和度按占比**朝角色调色板线性迁移**。
   因此器物的圆润、高光、阴影、材质纹理全部留下——它是"被角色配色重新着色"，
   而不是"被贴上图案"。
3. **标志作为点缀**：标志以**同一调色板重新着色**后小面积落在表面上，
   与整体配色同源，因此读起来是"这件器物本来就属于这个角色"。

`vision/mark_carry.py` 继续负责像素操作（缩放、alpha 合成、平铺），
本模块只负责"算出该用什么颜色"。
"""
import numpy as np
import cv2

import config


def extract_palette(paths, k=None, sample_limit=200000):
    """从角色自己的图像里提取主色，返回 [{"rgb": (r,g,b), "weight": float}, ...]。

    `paths`：角色图像路径列表（逐标志裁剪、参考图都可以）。

    **只在"有色像素"上做聚类**（实测教训，很关键）：
    第一版对所有像素做 k-means，结果调色板被**白色/高光**淹没——
    角色是浅蓝近白的头发、白瓷般的皮肤，白色占了 46%，
    于是"角色配色"实际变成"近白 + 灰紫"，迁移到白瓷杯上完全看不出变化。
    而这恰恰丢掉了最该学的东西：**这个角色的点缀色**（金、粉、深蓝）。
    因此先用饱和度阈值把近白/近灰像素排除，再聚类；
    被排除的部分比例仍如实记录在 `coverage` 里，便于判断素材是否合适。
    """
    k = int(getattr(config, "MARK_STYLE_PALETTE_K", 5) if k is None else k)
    pixels = []
    for path in paths or []:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        # 统一缩到短边 128：主色提取不需要原始分辨率，缩图能省一个数量级的像素
        height, width = bgr.shape[:2]
        scale = 128.0 / max(1, min(height, width))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (max(8, int(width * scale)),
                                   max(8, int(height * scale))),
                             interpolation=cv2.INTER_AREA)
        pixels.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3))
    if not pixels:
        return []
    stacked = np.concatenate(pixels, axis=0).astype(np.uint8)
    if stacked.shape[0] > sample_limit:
        step = int(stacked.shape[0] // sample_limit) + 1
        stacked = stacked[::step]

    # 只保留"有色"像素：HSV 的 S 高于阈值（排除纯白/纯灰），V 不过低（排除死黑）
    hsv = cv2.cvtColor(stacked.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat_min = float(getattr(config, "MARK_STYLE_SAT_MIN", 40.0))
    val_min = float(getattr(config, "MARK_STYLE_VAL_MIN", 35.0))
    keep = (hsv[:, 1].astype(np.float32) >= sat_min) & \
           (hsv[:, 2].astype(np.float32) >= val_min)
    chromatic = stacked[keep].astype(np.float32)
    coverage = float(keep.mean()) if keep.size else 0.0
    if chromatic.shape[0] < k * 4:
        # 有色像素太少（素材几乎全是白/灰）：退回全部像素，但如实标记
        chromatic = stacked.astype(np.float32)
        coverage = 1.0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.6)
    _compactness, labels, centres = cv2.kmeans(
        chromatic, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.reshape(-1), minlength=k).astype(np.float32)
    total = float(counts.sum()) or 1.0
    palette = []
    for index in np.argsort(-counts):
        rgb = tuple(int(round(v)) for v in centres[index])
        palette.append({"rgb": rgb, "weight": round(float(counts[index] / total), 4)})
    if palette:
        # coverage = 参与聚类的"有色像素"占全部像素的比例（审计用）
        palette[0] = dict(palette[0], chromatic_coverage=round(coverage, 4))
    return palette


def palette_summary(palette, limit=4):
    """调色板的人类可读摘要（日志与审计用）。"""
    parts = []
    for item in (palette or [])[:limit]:
        parts.append(f"#{item['rgb'][0]:02x}{item['rgb'][1]:02x}{item['rgb'][2]:02x}"
                     f"({item['weight']:.0%})")
    return " ".join(parts)


def _lab(rgb_array):
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB).astype(np.float32)


def restyle_surface(image_array, region_mask, palette, strength=None):
    """把角色的**配色**迁移到目标表面，作为"加法"的一部分。

    ## 三轮做法的取舍（前两轮都实测失败，记录以免重走）

    1. **只迁移色度、明度完全不动**：在浅色器物上数学上无效——
       白瓷 L≈240，在那个明度上放不下角色主色（`#3a6e9b` L≈45）所需的色度，
       实测"几乎看不出变化"。
    2. **按明度相对位置整体映射到调色板**：能上色，但把器物**洗成了淡彩块**——
       实测区域内平均 RGB 由 `[184,186,175]` 变成 `[119,130,153]`：
       明度掉了约 35%、彩度被拉平，陶瓷的受光与质感消失。
       这违反需求里"保留原图像主体的**形态**"与"保留原图像的画面风格"，
       属于**替换**而不是加法（用户看到的就是"杯子被涂成一块淡彩"）。
    3. **本版：加法式着色**——保留原像素的明度层次，只把彩度朝角色主导色偏移，
       明度仅做很轻的跟随；高光进一步少动，保住器物反光。
       效果是"白瓷沾上角色的蓝"，而不是"白瓷被刷成蓝色"。

    ## 羽化不外溢

    羽化用**内缩后**的掩版再模糊：羽化带落在器物内侧，
    不会把颜色抹到器物轮廓之外的背景上。
    （实测：不外缩时"区域外平均改动"达到 3.847，背景被染了淡淡一层色。）
    """
    if not palette:
        return image_array, {"applied": False, "reason": "没有可用的角色调色板"}
    if region_mask is None or not region_mask.any():
        return image_array, {"applied": False, "reason": "没有可承载区域"}
    strength = float(getattr(config, "MARK_STYLE_STRENGTH", 0.88)
                     if strength is None else strength)

    inside = region_mask.astype(bool)
    rgb = image_array.astype(np.float32)
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB).astype(np.float32)
    values = lab[:, :, 0][inside]
    if values.size < 16:
        return image_array, {"applied": False, "reason": "区域内像素过少"}
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))

    # 角色**主导色**（占比最大那一色）：角色之所以是那个角色靠的是它
    dominant = max(palette, key=lambda item: item["weight"])["rgb"]
    dominant_lab = cv2.cvtColor(np.array([[dominant]], dtype=np.uint8),
                                cv2.COLOR_RGB2LAB).reshape(-1, 3)[0]

    # 高光保护：区域内最亮的一小撮少迁移，否则器物失去光泽、看起来像哑光漆
    protect = float(getattr(config, "MARK_STYLE_HIGHLIGHT_PROTECT", 0.30))
    grey = lab[:, :, 0]
    highlight = np.clip((grey - high) / max(1.0, 255.0 - high), 0.0, 1.0)

    # 掩版内缩后羽化：羽化带留在器物内侧，颜色不外溢到背景
    erode_px = int(getattr(config, "MARK_STYLE_ERODE_PX", 10))
    hard = (inside.astype(np.uint8) * 255)
    if erode_px > 0:
        hard = cv2.erode(hard, np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8))
    if int((hard > 127).sum()) < 64:
        hard = inside.astype(np.uint8) * 255
    alpha = cv2.GaussianBlur((hard > 127).astype(np.float32), (0, 0),
                             float(getattr(config, "MARK_STYLE_FEATHER", 14.0)))
    alpha = np.clip(alpha, 0.0, 1.0)
    # **权重必须再乘一次 inside**（实测踩过，代价是背景被整体染色）：
    # 高斯模糊的 alpha 在"掩版薄处"不会严格归零（悬空的小块、孔洞边缘、
    # 以及噪声形成的孤立像素），于是那些区域外的像素也拿到了非零权重。
    # 实测表现：区域外被改动的像素占区域外总量的 **80%**，
    # 且被改动像素距区域边界中位 **107 像素**（远超羽化半径 14），
    # 说明不是羽化外溢，而是权重在区域外没有归零。
    # 乘上 `inside` 后，区域外的权重严格为 0——"只做加法"的硬约束由此成立。
    weight = (strength * alpha * (1.0 - protect * highlight)) * inside

    out = lab.copy()
    # 明度：保留原有层次，只叠一个很轻的偏移（默认 0.22）
    follow = float(getattr(config, "MARK_STYLE_LUMA_FOLLOW", 0.22))
    out[:, :, 0] = lab[:, :, 0] + (float(dominant_lab[0]) - float(np.median(values))) \
        * follow * weight
    # 彩度：朝角色主导色的 a/b 位移（加法，不是替换）
    for channel in (1, 2):
        shift = float(dominant_lab[channel]) - float(lab[:, :, channel][inside].mean())
        out[:, :, channel] = lab[:, :, channel] + shift * weight

    result = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    # **区域外逐像素取回原值**（硬保证，不依赖 alpha 是否恰好为 0）。
    # 实测：即使 `weight` 在区域外为 0，RGB->LAB->RGB 的往返本身不是无损的
    # （8 位量化误差，实测区域外最大改动 4~5）。而"只做加法"要求区域外**完全不动**，
    # 因此这里直接合成：区域内用着色结果，区域外保留原像素。
    result = np.where(inside[:, :, None], result, image_array).astype(np.uint8)
    stats = {
        "applied": True, "strength": strength, "luma_follow": follow,
        "palette": [{"rgb": list(item["rgb"]), "weight": item["weight"]}
                    for item in palette],
        "dominant_rgb": list(dominant),
        "region_range": [round(low, 1), round(high, 1)],
        "mean_before": [round(float(rgb[:, :, c][inside].mean()), 1) for c in range(3)],
        "mean_after": [round(float(result[:, :, c][inside].mean()), 1) for c in range(3)],
        "region_pixels": int(inside.sum()),
        "note": "加法式着色：保留原明度层次，彩度朝角色主导色偏移；羽化带内缩在器物内侧",
    }
    return np.clip(result, 0, 255).astype(np.uint8), stats


def restyle_asset(rgba, palette, strength=None):
    """把标志素材按**同一套角色调色板**重新着色，使它与整体配色同源。

    为什么要重着色（用户反馈的直接回应）：直接把参考图的原始像素贴上去，
    它带来的是**那张图自己的配色**，与器物被渲染出的配色不是一个体系，
    观感就是"贴纸"。按同一调色板着色后，标志成为"这件器物的纹样"的一部分。
    """
    if rgba is None or not palette:
        return rgba, {"applied": False, "reason": "没有调色板或素材"}
    strength = float(getattr(config, "MARK_STYLE_ASSET_STRENGTH", 0.70)
                     if strength is None else strength)
    rgb = rgba[:, :, :3]
    weights = np.array([item["weight"] for item in palette], dtype=np.float32)
    colours = np.array([item["rgb"] for item in palette], dtype=np.uint8).reshape(-1, 1, 3)
    palette_lab = cv2.cvtColor(colours, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    weights = weights / max(1e-6, float(weights.sum()))
    target = (palette_lab * weights[:, None]).sum(axis=0)

    lab = _lab(rgb)
    # 保留明度层次：(L - 中位) 不动，只把 L 的中心搬到调色板的 L 中心附近
    centre = float(np.median(lab[:, :, 0]))
    out = lab.copy()
    out[:, :, 0] = lab[:, :, 0] + (float(target[0]) - centre) * strength * 0.5
    for channel in (1, 2):
        source = float(lab[:, :, channel].mean())
        out[:, :, channel] = lab[:, :, channel] + \
            (float(target[channel]) - source) * strength
    recoloured = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    result = np.dstack([recoloured, rgba[:, :, 3]])
    return result, {"applied": True, "strength": strength,
                    "target_rgb": [int(round(v)) for v in
                                   cv2.cvtColor(np.uint8([[target[:3]]]),
                                                cv2.COLOR_LAB2RGB)[0, 0]]}
