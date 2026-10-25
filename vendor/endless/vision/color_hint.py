# -*- coding: utf-8 -*-
"""vision/color_hint.py —— 区域配色锚定（只改初始图，不改生成过程）

解决的问题（实测定位）：img2img 从源图出发时，源图最显眼的颜色区域（例如深色头发）
在潜空间里是强信号，重绘强度 0.78 也不足以把它改写成角色的配色。底模本身能画出该配色
（底模能力探针已证明），所以最短路径是**先把初值改到位**，再让扩散按原参数运行。

## 为什么是"锚定"而不是"替换色度"（这一版的核心修正）

旧实现把区域内每个像素的色度（LAB 的 a/b）**整块替换**成目标颜色的色度，只保留明度。
实测后果（三条都可在图上直接看到）：
  * 整片区域变成同一个色度 ⇒ 皮肤、发丝、衣物全部被染上一层蓝，"人物整体偏蓝"；
  * 色度替换是**加性**的（新色度与源色度无关）⇒ 暖色皮肤被拉向冷蓝，
    明暗层次虽然保住，色相层次却被抹平，看起来就是"颜色糅杂涂抹"；
  * 边界即使羽化，青色与暖色皮肤相减也会出现一条灰色过渡带 ⇒ "轮廓蓝色层"。

本版改成**结构保持的线性锚定**：
    在 LAB 空间对每个区域拟合一组 (尺度, 偏移)，把该区域的颜色分布**平移并适度缩放**
    到目标色附近，而不是替换：

        target_a = mean_a*scale + mean_bias
        src_a'   = (src_a - mean_a) * scale + target_a

    即"整体搬家 + 保留内部相对差异"。线性变换的三大好处：
      1. 源图的明暗与色相层次按比例保留（纹理不会被抹平）；
      2. 不会把灰/白像素推向目标色（(x-mean)*scale 对接近均值的像素几乎无位移，
         羽化边界因此平滑接回源图，不会再出现那条蓝色轮廓层）；
      3. 目标色只决定"搬到哪"，不决定"长什么样"，因此对任何角色通用。

目标颜色与目标区域照旧全部来自角色档案与自动定位，不写死任何角色。
"""
import numpy as np
import cv2
from PIL import Image

import config
from character import profile as profile_module

# 需要"发色"而不是"服装配色"的区域名（由 locate_regions 定名）
_HAIR_REGIONS = {"head", "hair", "face", "头部", "头发"}


def color_hint(image, masks, profile, strength=None):
    """按区域掩版把源图配色**锚定**向角色配色，返回 (改写后的图, 审计)。

    masks: {区域名: PIL "L" 掩版}；区域名只用于选择目标颜色，取值来自档案。
    profile: character.profile 的档案（提供发色与配色）。
    没有可用目标颜色时**原样返回**并说明原因，不改动图。

    模式（`config.COLOR_ANCHOR_MODE`，三种都实测过，结论写在 config 注释里）：
        "off"       不改写（默认）：实测这是画质最好、且不会把人物染蓝的选择；
        "luminance" 只把区域的**明度带**搬到目标色，色度完全不动：
                    用于"源图颜色明显更暗，只靠扩散改不动明度"的场合；
        "color"     明度 + 色度都搬（按源色与目标色的距离自适应选择门控方式）。
    """
    mode = str(getattr(config, "COLOR_ANCHOR_MODE", "off")).lower()
    audit = {"applied": False, "regions": [], "reasons": [],
             "method": "structure_preserving_anchor", "mode": mode}
    if mode == "off":
        audit["reasons"].append("配置为 off（实测：改写初值会把暖色皮肤一起推向冷色）")
        return image, audit
    target_colors = _target_colors(profile)
    if not target_colors:
        audit["reasons"].append("角色档案没有可解析的配色")
        return image, audit
    if not masks:
        audit["reasons"].append("没有区域掩版")
        return image, audit

    source = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    result = lab.copy()
    touched = np.zeros(lab.shape[:2], dtype=bool)
    for name, mask in masks.items():
        if mask is None:
            continue
        mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        array = np.asarray(mask)
        if (array > 128).mean() < 0.005:
            audit["reasons"].append(f"{name}: 掩版过小，跳过")
            continue
        color = _pick_color(name, profile, target_colors, source, array)
        if color is None:
            audit["reasons"].append(f"{name}: 没有匹配到目标颜色")
            continue
        chroma = 0.0 if mode == "luminance" else _ratio(
            config.COLOR_ANCHOR_CHROMA if strength is None else strength)
        luminance = config.COLOR_ANCHOR_LUMINANCE if name in _HAIR_REGIONS else 0.0
        if mode == "luminance" and luminance <= 0:
            luminance = config.COLOR_ANCHOR_LUMINANCE
        result, stats = _anchor_region(result, array, color, chroma, luminance)
        touched |= (array > 128)
        audit["regions"].append(dict(stats, region=name,
                                     target_rgb=[int(v) for v in color],
                                     target_note=(_hair_target(profile, target_colors)[1]
                                                  if name in _HAIR_REGIONS else "")))
    if audit["regions"]:
        audit["applied"] = True
        print("[color_hint] 已锚定初值配色(结构保持): "
              + ", ".join(f"{r['region']}->{r['target_rgb']}({r['chroma_mode']})"
                          for r in audit["regions"]))
    else:
        print(f"[color_hint] 未改写初值: {audit['reasons']}")
    # 只把**被改写过的**像素送回 RGB：LAB↔RGB 往返在 uint8 上有 ±1 的量化误差，
    # 若整图往返，掩版外会出现肉眼难见但逐位不等的改动（回归测试直接钉死了这一条）。
    out = source.copy()
    if touched.any():
        converted = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        out[touched] = converted[touched]
    return Image.fromarray(out, mode="RGB"), audit


def _ratio(value):
    """把强度夹到 [0,1]；非数值按 0 处理（不静默套用一个越界值）。"""
    try:
        return float(min(1.0, max(0.0, float(value))))
    except (TypeError, ValueError):
        return 0.0


def _target_colors(profile):
    """档案里可解析的颜色：先发色，再配色三槽（去重）。"""
    colors = []
    hair = profile_module.hair_color(profile)
    if hair:
        colors.append(("hair", [int(v) for v in hair]))
    for name, rgb in profile_module.palette_colors(profile):
        colors.append((name, [int(v) for v in rgb]))
    seen, unique = set(), []
    for name, rgb in colors:
        key = tuple(rgb)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, rgb))
    return unique


def _chroma(rgb):
    """颜色的"彩度"：LAB 里 (a,b) 到中性轴的距离。中性灰接近 0。"""
    lab = cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    return float(np.hypot(float(lab[1]) - 128.0, float(lab[2]) - 128.0))


def _hair_target(profile, colors):
    """发色锚点：档案发色**够彩**时用它，否则退回配色里最像发色的那一项。

    ## 为什么必须有这个退回（实测，本轮最关键的发现）

    角色档案里的发色是**视觉模型给的"颜色名"**，而名字会随运行漂移：
    同一个角色的浅蓝发，历史档案写成 `light blue`（→ 锚定后出图头发变蓝、正确），
    本次重建写成 **`silver`**（→ 目标色 `[192,192,200]`，彩度只有 4，
    "锚定"把一个浅蓝发的角色染成了**中性灰**，出图反而比不锚定更不像角色）。

    也就是说：把"改变发色"这件事交给一个**会被名字误导**的模块，
    结果只能靠运气。判据改成**看颜色的彩度**而不是看名字：
      * 档案发色彩度足够（真是一个有彩度的颜色）→ 按它锚定；
      * 彩度近乎中性（`silver` / `gray` / `white` / `black` 这类）→
        从该角色配色里挑**最像发色**的一项（见下）。

    `COLOR_ANCHOR_HAIR_PREFERRED` 里的名字（blue/cyan/lavender/aqua…）优先：
    浅蓝、淡紫这类"发型色"在英文配色名里几乎总落在这几个词上。
    没有命中时退回**彩度最高、且偏亮**的一项——浅色发在画面里是"亮而低饱和"的，
    所以亮度的权重与彩度相当。两项都是角色自己档案里的数据，不引入角色专用逻辑。

    返回 (目标 RGB, 说明) 或 (None, 说明)。
    """
    hair = profile_module.hair_color(profile)
    floor = float(getattr(config, "COLOR_ANCHOR_MIN_HAIR_CHROMA", 14.0))
    if hair and _chroma(hair) >= floor:
        return hair, f"档案发色（彩度 {_chroma(hair):.0f}）"
    candidates = [(name, rgb) for name, rgb in (colors or [])
                  if name != "hair" and _chroma(rgb) >= floor]
    if candidates:
        preferred = tuple(str(word).lower() for word in
                          getattr(config, "COLOR_ANCHOR_HAIR_PREFERRED", ()))
        hit = [(name, rgb) for name, rgb in candidates
               if any(word in str(name).lower() for word in preferred)]
        pool = hit or candidates
        # 偏向"亮而低饱和"的发型色：亮度权重与彩度相当
        chosen_name, chosen = max(
            pool, key=lambda item: _luminance(item[1]) * 0.5 + _chroma(item[1]))
        reason = ("档名优先" if hit else "彩度与亮度最高")
        note = (f"档案发色彩度不足（{_chroma(hair):.0f} < {floor:.0f}，"
                f"名字多为 silver/gray 一类中性色）" if hair else "档案没有发色")
        return chosen, (f"{note}，改用配色项「{chosen_name}」（{reason}，"
                        f"彩度 {_chroma(chosen):.0f} 亮度 {_luminance(chosen):.0f}）")
    if hair:
        return hair, f"档案发色（彩度 {_chroma(hair):.0f}，无可替代的配色项）"
    return None, "档案没有可用的发色"


def _pick_color(region, profile, colors, source, mask):
    """为区域选目标颜色；不确定时返回 None，绝不硬凑。

    * 头部/毛发区域：用发色锚点（见 `_hair_target`：档案发色彩度不足时退回配色）；
    * 其余区域（服装/配饰/表面）：选**与发色明度最接近**的配色，并且要求它比源区域更亮一些。
      理由：一个角色设计里头发与服装通常处在相近的明度带，用发色当锚点比"按源区域明度挑"
      更接近角色本身；后者会把浅色衣物染成配色的点缀色（实测缺陷：白裙被染成粉）。
      这样不依赖任何角色的颜色名，换角色同样成立。
    """
    hair, hair_note = _hair_target(profile, colors)
    if region in _HAIR_REGIONS and hair:
        return hair
    if not colors:
        return None
    if len(colors) == 1:
        return colors[0][1]
    anchor = _luminance(hair) if hair else None
    if anchor is not None:
        ranked = sorted(colors, key=lambda item: abs(_luminance(item[1]) - anchor))
        chosen = ranked[0][1]
        # 源区域已经很亮（例如白衣服）时不要把它压暗成配色里的深色
        if _median_luminance(source, mask) > 0.72 and _luminance(chosen) < 170:
            brighter = [rgb for _, rgb in colors if _luminance(rgb) >= 170]
            if brighter:
                chosen = max(brighter, key=_luminance)
        return chosen
    luminance = _median_luminance(source, mask)
    ranked = sorted(colors, key=lambda item: _luminance(item[1]))
    position = min(max(luminance, 0.0), 1.0) * (len(ranked) - 1)
    return ranked[int(round(position))][1]


def _median_luminance(source, mask):
    grey = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    inside = mask > 128
    if not inside.any():
        return 0.5
    return float(np.median(grey[inside])) / 255.0


def _luminance(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def region_weights(mask, feather=None):
    """区域软权重 = 羽化 alpha / 255。合成与统计都用它，保证边界一致。

    用 feather_alpha(interior=True)：过渡带落在区域**内部**，掩版外权重严格为 0，
    因此掩版外逐像素不变（测试钉死）。旧写法让 2~3 像素的过渡值渗出掩版外，
    在轮廓周围留下一圈淡淡的色边——正是"轮廓蓝色层"的一个来源。
    """
    from subject_transfer import feather_alpha
    feather = config.COLOR_ANCHOR_FEATHER if feather is None else feather
    if feather <= 0:
        return np.asarray(mask, dtype=np.float32) / 255.0
    alpha = feather_alpha(Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L"),
                          feather=feather, interior=True)
    return np.asarray(alpha, dtype=np.float32) / 255.0


def _anchor_region(lab, mask, color, chroma_ratio, luminance_ratio):
    """把区域内**主体自身颜色**的像素锚定到目标色，背景/皮肤等异色像素不动。

    这一版是第三版，前两版的失败原因都记在下面（都是实测出来的）：

    * 第一版（整块替换色度）：区域内每个像素的 a/b 直接换成目标色度。结果整片区域
      同一色相 ⇒ "人物整体偏蓝"；色相层次被抹平 ⇒ "颜色糅杂涂抹"。
    * 第二版（整块线性锚定）：把整块区域的色度分布整体搬家并缩放。比第一版好，但
      仍然**平均地**作用在所有像素上；区域掩版里同时含头发、脸、衣服，
      于是暖色皮肤被一起推向蓝色（实测 head 区 b 偏移 -24.9），脸被染成青灰——
      这就是"人物整体偏蓝 + 轮廓一圈蓝层"的直接来源。

    本版判据（只依赖像素自身统计量，不含角色/题材假设）：
      1. **发色锚点取区域内偏亮的那一部分像素的色度中位数**。角色发色通常是该区域
         最亮的颜色成分（浅蓝头发 vs 深棕源发，或深色头发 vs 更深的阴影/背景），
         而"最暗像素的色度"在这种照片里等于中性灰，拿它当锚点会把整块推向灰紫（实测）。
      2. 置信度 = 像素色度到锚点的距离经 smoothstep 衰减；距离远的像素（皮肤、
         背景、深色阴影）权重趋零，**不给它们上色**。
      3. 新色度 = 原色度 + (目标色度 - 原色度) * 置信度 * 强度：是**位移**而不是替换，
         同色像素之间的相对差异完整保留（纹理不被抹平）。
      4. 明度按"高置信度像素的中位数"整体平移到目标明度带，同样只作用于高置信度像素，
         因此脸的明暗不受影响。
    """
    import numpy as np

    inside = mask > 128
    weights = region_weights(mask)
    # `dropped` 必须在分支之外先有值（实测缺陷）：原先它只在
    # "最大连通块存在"与 `allow_chroma=False` 两条路径里被赋值，
    # 而 `allow_chroma=True` 且 `_largest_blob` 返回 None 时两条路都不走，
    # 于是下面的审计字段引用未定义变量，抛 `UnboundLocalError`——
    # 一次真实运行里它直接**中止了整批出图**，而不是"这一块不做锚定"。
    dropped = 0.0
    if int(inside.sum()) < 64:
        return lab, {"coverage": 0.0, "chroma": chroma_ratio, "luminance": luminance_ratio}
    target_lab = cv2.cvtColor(np.array([[color]], dtype=np.uint8),
                              cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

    anchor, share = _dominant_chroma(lab, inside)
    target_gap = float(np.linalg.norm(anchor - target_lab[1:]))
    radius = max(4.0, float(config.COLOR_ANCHOR_HUE_RADIUS))
    confidence = _smoothstep(
        np.sqrt(((lab[:, :, 1:3] - anchor) ** 2).sum(axis=2)), 0.0, radius)
    effective = (confidence * weights).astype(np.float32)

    # 源色与目标色接近（同色系）时用色相置信度锚定：能把颜色精确钉到目标色；
    # 差得远时（实测深棕 → 浅蓝，距离 70）锚点本身就是深棕，置信度会排除几乎所有像素
    # （实测只染到 11%），等于失效 —— 改用"偏暗 × 低饱和"门控，把脸排除掉之后再位移。
    allow_chroma = target_gap <= float(config.COLOR_ANCHOR_MAX_GAP)
    if allow_chroma:
        mode = "hue_confidence"
        tint = effective
        aim = anchor * config.COLOR_ANCHOR_CHROMA_KEEP + target_lab[1:] \
            * (1.0 - config.COLOR_ANCHOR_CHROMA_KEEP)
        source_median = float(np.median(lab[:, :, 0][inside]))
    else:
        mode = "dark_lowsat_shift"
        source_median = float(np.median(lab[:, :, 0][inside]))
        first = float(np.percentile(lab[:, :, 0][inside],
                                    config.COLOR_ANCHOR_DARK_PERCENTILE))
        ramp = max(6.0, float(config.COLOR_ANCHOR_GATE_RAMP))
        dark = _smoothstep(lab[:, :, 0], first + ramp, first - ramp)
        # 皮肤在 a 通道上明显偏红：用区域自身的 a 分位数当门槛，只保留低饱和那一部分
        a_cut = float(np.percentile(lab[:, :, 1][inside],
                                    config.COLOR_ANCHOR_SKIN_PERCENTILE))
        a_ramp = max(3.0, float(config.COLOR_ANCHOR_SKIN_RAMP))
        low_sat = _smoothstep(lab[:, :, 1], a_cut + a_ramp, a_cut - a_ramp)
        gate = dark * low_sat
        tint = (gate * weights).astype(np.float32)
        aim = target_lab[1:]
        # ---- 只保留"主体自身颜色的最大连成一片" ----
        # 实测问题：门控通过的不只是头发，还有**脸上与颈部的阴影**——它们同样是
        # "偏暗 + 低饱和"，在颜色统计上与深色头发无法区分。它们散落在掩版各处、
        # 彼此不连通，而上色之后会变成一块块孤立的深色斑（实测出图上额头/颈部/
        # 手臂上的蓝色色块就是这么来的）。
        # 判据：头发是一整片连通区域，阴影是若干小块。取最大连通块 + 补洞即可分开，
        # 这一步只用几何形态，不含任何角色或题材假设。
        keep = _largest_blob(tint >= config.COLOR_ANCHOR_KEEP_THRESHOLD)
        if keep is not None:
            # 把"最大连通块"向外扩张一档再与原门控取交集：
            # 最大块的连通范围只包含较粗的发束，细发丝与边缘是**不连通**的，
            # 只按最大块上色会在头发上留下一半蓝一半褐的碎片状边界（实测）。
            # 扩张半径由 config 控制，扩张范围仍被区域掩版本身限制（那里只有头发）。
            grow = max(1, int(round(config.COLOR_ANCHOR_BLOB_GROW * min(lab.shape[:2]))))
            if grow > 1:
                keep = cv2.dilate(keep.astype(np.uint8), np.ones((grow * 2 + 1,) * 2, np.uint8)) > 0
            keep = keep & (tint >= config.COLOR_ANCHOR_GROW_THRESHOLD)
            dropped = float(((tint >= config.COLOR_ANCHOR_KEEP_THRESHOLD) & ~keep).mean())
            tint = (tint * keep).astype(np.float32)
        else:
            dropped = 0.0

    # ---- 覆盖范围扩展（实测踩过：只染到掩版的 1/3）----
    # 第三轮实测：门控最终只选中头部掩版内 **33.7%** 的像素（占整幅 7.4%），
    # 于是出图里头发出现"一块蓝 + 一片褐"的硬边拼接（人工核对确认）。
    # 扩展必须**有界**：头部掩版里同时含脸（实测左半 L 中位 109，是皮肤），
    # 无条件按掩版扩张会把脸染蓝——那正是 19 轮里"人物整体偏蓝"的成因。
    # 因此只把"与种子同色系"的像素并进来：暗（不发亮）且不偏红（不是皮肤）。
    stats_extra = {}
    expanded = _expand_seed(tint, lab, inside)
    if expanded is not None:
        added = float(((expanded > 0.5) & (tint <= 0.5)).sum()) / max(
            1.0, float(inside.sum()))
        tint = expanded
        stats_extra["expanded_added_ratio"] = round(added, 4)
    # 边缘羽化：硬边是"平色块"观感的一半来源（另一半是覆盖不足）。
    # 只模糊 tint 自身，掩版外仍由 weights 压到 0。
    feather = int(getattr(config, "COLOR_ANCHOR_TINT_FEATHER", 0) or 0)
    if feather > 1:
        kern = feather * 2 + 1
        tint = cv2.GaussianBlur(tint, (kern, kern), 0)
        tint = (tint * weights).astype(np.float32)
    stats_extra["tinted_ratio_in_mask"] = round(
        float(((tint > 0.5) & inside).sum()) / max(1.0, float(inside.sum())), 4)

    out = lab.copy()
    tinted = tint * chroma_ratio
    for index, channel in enumerate((1, 2)):
        out[:, :, channel] = lab[:, :, channel] + (float(aim[index]) - lab[:, :, channel]) * tinted
    stats = {"coverage": round(float(inside.mean()), 4),
             "chroma": chroma_ratio, "luminance": luminance_ratio,
             "anchor_ab": [round(float(v), 1) for v in anchor],
             "anchor_share": round(float(share), 3),
             "target_ab": [round(float(v), 1) for v in target_lab[1:]],
             "anchor_gap": round(target_gap, 1),
             "chroma_mode": mode,
             "source_L_median": round(source_median, 1),
             "target_L": round(float(target_lab[0]), 1),
             "dropped_blobs_ratio": round(dropped, 4),
             "tinted_ratio": round(float((tint > 0.5).mean()), 4)}
    stats.update(stats_extra)
    if luminance_ratio > 0:
        # 明度朝目标色位移，只作用在与色度同一批像素上（门控不足会把脸的明暗也带动）。
        #
        # 两处**实测必需的**约束（第三、四轮各踩一次）：
        #  1. `COLOR_ANCHOR_LUMINANCE_KEEP` 原本只在 config 里定义、**代码从未使用**
        #     （死配置）。现在按它的文档含义生效：围绕区域中位数把原有明度**层次**
        #     缩放保留，1.0 = 层次完整保留。
        #  2. 位移量必须有上限。源发实测 L 中位 16、目标色 L=122，
        #     不设上限时一次把头发推到 122 —— 出图整头变成**惨白的浅灰**
        #     （第四轮人工核对），因为 strength=0.58 的扩散会把"已经很亮"的初值
        #     当成既成事实、不再把它压暗回来。
        #     发色锚定的职责是**给出色相**，明度交给扩散与提示词，
        #     因此位移上限收到 `COLOR_ANCHOR_MAX_LUMINANCE_SHIFT`（默认 55）。
        strength = (tint * luminance_ratio).astype(np.float32)
        if float(strength.max()) > 0:
            keep = float(getattr(config, "COLOR_ANCHOR_LUMINANCE_KEEP", 1.0))
            keep = min(1.0, max(0.0, keep))
            centred = source_median + (lab[:, :, 0] - source_median) * keep
            raw_shift = float(target_lab[0]) - source_median
            cap = abs(float(getattr(config, "COLOR_ANCHOR_MAX_LUMINANCE_SHIFT", 55.0)))
            shift = max(-cap, min(cap, raw_shift))
            shifted = centred + shift * strength
            out[:, :, 0] = lab[:, :, 0] * (1 - strength) + shifted * strength
            stats["luminance_keep"] = keep
            stats["luminance_shift"] = round(shift, 1)
            stats["luminance_shift_raw"] = round(raw_shift, 1)
    return out, stats


def _expand_seed(tint, lab, inside):
    """把锚定种子扩展到"同色系"的邻近像素（有界，不越过肤色/亮色）。

    为什么需要：第三轮实测门控只选中掩版内 33.7% 的像素，
    出图里头发出现"一块蓝 + 一片褐"的硬边拼接。

    为什么必须**有界**：头部掩版里同时含脸（实测左半 L 中位 109，是皮肤），
    无条件按掩版扩张会把脸染蓝——那正是 19 轮里"人物整体偏蓝"的成因。

    判据只用种子自身的统计量（不含角色/题材假设）：
      * **足够暗**：L 低于种子 L 的高分位（头发比皮肤暗）；
      * **不偏红**：a 不超过种子 a 的中位数 + 一点余量（皮肤在 a 通道上明显偏红）；
      * **邻近区域掩版**：落在外扩一档的掩版内即可，不必严格在掩版内。
    达不到"足够暗 + 不偏红"的像素保持原样——**宁可少染，也不要把皮肤染进去**。

    为什么允许扩到掩版之外一档（实测）：SAM 按视觉模型的头部框细化出来的掩版
    只覆盖右侧那一大团头发（框 x 从 0.294 起），**左侧的长发完全在掩版外**，
    于是同一头头发出现"右侧蓝、左侧褐"的拼接。
    而颜色判据（暗 + 不偏红）本身已经把皮肤与白衣排除了——
    真正在做约束的是颜色而不是掩版边界，所以"邻近"这一步不会引入肤色。
    外扩幅度由 `COLOR_ANCHOR_EXPAND_REACH` 控制，仍然只作用于紧邻掩版的像素。
    """
    import numpy as np
    seed = tint > 0.5
    if seed.sum() < 64:
        return None
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    seed_L = L[seed]
    seed_a = a[seed]
    L_cut = float(np.percentile(seed_L, config.COLOR_ANCHOR_EXPAND_L_PERCENTILE))
    L_ramp = max(4.0, float(config.COLOR_ANCHOR_EXPAND_L_RAMP))
    a_cut = float(np.median(seed_a)) + float(config.COLOR_ANCHOR_EXPAND_A_MARGIN)
    a_ramp = max(2.0, float(config.COLOR_ANCHOR_EXPAND_A_RAMP))
    dark_enough = _smoothstep(L, L_cut + L_ramp, L_cut - L_ramp)
    not_warm = _smoothstep(a, a_cut + a_ramp, a_cut - a_ramp)
    grown = (dark_enough * not_warm).astype(np.float32)
    reach = float(getattr(config, "COLOR_ANCHOR_EXPAND_REACH", 0.0) or 0.0)
    allowed = inside
    if reach > 0:
        radius = max(1, int(round(reach * min(lab.shape[:2]))))
        allowed = cv2.dilate(inside.astype(np.uint8),
                             np.ones((radius * 2 + 1,) * 2, np.uint8)) > 0
    grown[~allowed] = 0.0
    # 保留原有种子权重（种子处至少不降），避免把已经选对的地方削弱
    return np.maximum(tint, grown).astype(np.float32)


def _largest_blob(binary):
    """取二值图里最大的连通块并补洞；没有前景时返回 None。

    用于把"头发那一整片"与"散落的阴影小块"分开（两者在颜色统计上无法区分，
    但形态差异明显：头发连成一片，阴影是若干小块）。
    """
    import numpy as np
    import cv2
    if binary is None or not binary.any():
        return None
    array = binary.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=8)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    kept = (labels == largest).astype(np.uint8)
    # 补洞：最大块内部的孔（发丝间的高光）也要算进这一片，
    # 否则上色会在头发里留下未处理的斑点
    height, width = kept.shape
    flood = kept.copy()
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return np.clip(kept + holes, 0, 1).astype(bool)


def _dominant_chroma(lab, inside):
    """区域内"主体自身颜色"的色度锚点：取偏亮那一半像素的色度中位数。

    为什么是偏亮的一半而不是最亮的一小撮：头发上常有过曝高光（实测画面右侧有强烈
    暖色反光），只取最亮的 5% 会把反光的暖色当成发色。取"亮度中位数以上的像素"
    既避开了阴影与背景，也避开了极端高光；再用中位数而非均值，进一步抗离群点。
    """
    import numpy as np
    lightness = lab[:, :, 0][inside]
    chroma = lab[:, :, 1:3][inside]
    cut = float(np.median(lightness))
    chosen = lightness >= cut
    if int(chosen.sum()) < 32:
        chosen = np.ones_like(lightness, dtype=bool)
    anchor = np.median(chroma[chosen], axis=0).astype(np.float32)
    return anchor, float(chosen.mean())


def _smoothstep(distance, low, high):
    """平滑衰减：距离 <= low 时为 1，>= high 时为 0，中间用三次曲线过渡。"""
    import numpy as np
    if high <= low:
        return (distance <= low).astype(np.float32)
    x = np.clip((distance - low) / (high - low), 0.0, 1.0)
    return (1.0 - x * x * (3.0 - 2.0 * x)).astype(np.float32)


def _smoothstep(distance, low, high):
    """平滑衰减：距离 <= low 时为 1，>= high 时为 0，中间用三次曲线过渡。"""
    import numpy as np
    if high <= low:
        return (distance <= low).astype(np.float32)
    x = np.clip((distance - low) / (high - low), 0.0, 1.0)
    return (1.0 - x * x * (3.0 - 2.0 * x)).astype(np.float32)
