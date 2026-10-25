# -*- coding: utf-8 -*-
"""contour：把"模型画的平面纹样"**按曲面几何**承印到器物表面。

## 这一步解决的是用户指出的根本问题

用户（2026-10-08）：
  "你把标志图片规整得虽小复制排列，这并不能体现模型学习了角色的视觉特征
   并自然得运用这些"

实测结论（见 `histories/scripts/_probe_concept_canvas.py` 与
`_probe_diffusion_paint.py`）：

  * 让模型**直接把纹样画到摄影器物的曲面**上 -> 画不出来（只得到一层淡杂色）；
  * 让模型在**平面画布**上画角色纹样 -> **画得出来**（樱花飘落 + 金色丝带的
    完整纺织品图案、金色六边徽记），但配色偏向"泛用的日式粉／米白"，
    不是这个角色的深蓝／淡蓝。

因此正确的分工是：

  **模型**负责"画什么"（纹样的形状语言，这是它在训练里学到的）；
  **几何**负责"怎么落到器物上"（曲面映射、随受光变化、按角色调色板着色）。

本模块只做"几何"这一半。它与 `vision/mark_carry.py` 的区别：
`mark_carry` 是**把一张图原样贴上去**（所以看起来像贴纸），
本模块会**逐像素按柱面重新投影**并**按表面明暗调制**，
因此纹样会随器物圆润而收窄、随高光而变淡——这是"看起来是印上去的"的关键。
"""
import cv2
import numpy as np

import config


def cylindrical_map(image, mask, strength=None):
    """把平面图按**柱面**重新投影到目标区域，返回与原图同尺寸的新图。

    做法（纯几何，可复核）：
      1. 取目标区域的外接框，认为器物在这个框内是一段**圆柱面**；
      2. 对框内每个像素，按它到中心轴的横向距离计算归一化横坐标
         `u = asin(x_norm)`（圆柱的透视：越靠边越被压缩）；
      3. 从平面图上按 `u` 采样，因此图案在两侧自然收窄，
         而不是等宽平移——这就是"贴纸"与"承印"最直观的差别。

    `strength`：映射强度（0=不映射，1=完全按柱面采样）。
    """
    if mask is None or not mask.any():
        return image
    strength = float(getattr(config, "SURFACE_CYLINDER_STRENGTH", 1.0)
                     if strength is None else strength)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return image
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    width = right - left
    if width < 8:
        return image

    height_index, width_index = np.mgrid[top:bottom, left:right]
    # 归一化横坐标 -1 ~ 1
    x_norm = (width_index - (left + right) / 2.0) / max(1.0, width / 2.0)
    # 柱面投影：u = sin(asin(x)*k)/... 用最简形式 x -> sin(pi/2 * x) 的逆
    # 这里用 asin 的几何意义：像素越靠边，对应的表面弧长变化越慢
    angle = np.clip(x_norm, -1.0, 1.0) * (np.pi / 2.0)
    mapped = np.sin(angle)
    source_x = ((mapped * 0.5 + 0.5) * (width - 1)).astype(np.float32)
    source_x = left + source_x
    source_y = height_index.astype(np.float32)
    # 只在区域内采样（区域外的像素稍后会被原图覆盖）
    map_x = source_x[None].repeat(bottom - top, axis=0) if source_x.ndim == 1 else source_x
    map_y = source_y
    warped = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # 按强度与原图混合
    result = image.copy()
    region_view = warped
    blended = (image[top:bottom, left:right].astype(np.float32) * (1.0 - strength)
               + region_view.astype(np.float32) * strength)
    result[top:bottom, left:right] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def modulate_by_surface(base, printed, mask, strength=None):
    """让印上去的纹样**跟随器物的明暗**，而不是保持自身恒定亮度。

    做法：以器物原像素的**明度相对位置**作为调制系数——
    亮处（受光）纹样更亮、暗处（背光/边缘）纹样更暗，
    因此图案会随器物的圆润与高光自然起伏。
    这是"看起来印在物体上"最要紧的一条：贴纸的亮度不随物体变化。

    `base`：器物原图（提供明暗）；`printed`：已承印的图；`mask`：器物区域。
    """
    if mask is None or not mask.any():
        return printed
    strength = float(getattr(config, "SURFACE_SHADING_STRENGTH", 0.55)
                     if strength is None else strength)
    base_grey = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    values = base_grey[mask]
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    span = max(1.0, high - low)
    relative = np.clip((base_grey - low) / span, 0.0, 1.0)
    # 以中灰为 1.0：亮处 >1（提亮），暗处 <1（压暗）
    factor = 1.0 + (relative - 0.5) * 2.0 * strength
    printed_grey = cv2.cvtColor(printed, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ratio = np.where(printed_grey > 1.0, base_grey / np.maximum(printed_grey, 1.0), 1.0)
    ratio = np.clip(ratio, 0.6, 1.6)
    out = printed.astype(np.float32)
    adjust = (1.0 - ratio) * 0.5 + 0.5          # 0.5 = 不变
    delta = (factor - 1.0)[:, :, None] * 40.0
    out = out + delta * mask[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def recolour_to_palette(printed, mask, palette, strength=None):
    """把生成的纹样**朝角色调色板**着色。

    为什么需要：实测概念 LoRA 在平面画布上生成的纹样配色是"泛用日式"的
    （粉樱花 + 米白），而角色是深蓝系。若直接用这张图，器物会变成
    "日式花纹杯子"而不是"这个角色的杯子"。因此按角色调色板把纹样的
    色相拉回来——形状来自模型，颜色来自角色证据。

    与 `vision/mark_style.restyle_surface` 的区别：那一个是给**器物原像素**
    着色（保留明度层次、只偏移彩度）；本函数是给**生成出来的纹样**着色，
    因此必须保留纹样自身的明暗结构（花瓣的渐变、描边），
    只把整体色相朝角色主色旋转。
    """
    if printed is None or mask is None or not mask.any() or not palette:
        return printed
    strength = float(getattr(config, "SURFACE_RECOLOUR_STRENGTH", 0.6)
                     if strength is None else strength)
    dominant = max(palette, key=lambda item: item["weight"])["rgb"]
    target = cv2.cvtColor(np.array([[dominant]], dtype=np.uint8),
                          cv2.COLOR_RGB2LAB).reshape(-1, 3)[0]

    lab = cv2.cvtColor(printed, cv2.COLOR_RGB2LAB).astype(np.float32)
    inside = mask.astype(bool)
    out = lab.copy()
    for channel in (1, 2):
        source_mean = float(lab[:, :, channel][inside].mean())
        shift = (float(target[channel]) - source_mean) * strength
        out[:, :, channel] = lab[:, :, channel] + shift * inside
    result = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    # 区域外逐像素取回原值（硬保证只动器物）
    return np.where(inside[:, :, None], result, printed).astype(np.uint8)


def flatten_luminance_relation(reference, printed, mask, strength=None):
    """把纹样的**明暗层次**换成参照物的层次，避免纹样自带一套打光。

    做法：用参照物（器物原图）的明度梯度替换纹样明度梯度的低频部分，
    保留纹样自身的高频细节（花瓣边缘、描边）。
    这样"纹样是被这个物体的光照射出来的"，而不是自带打光贴上去的。
    """
    if mask is None or not mask.any():
        return printed
    strength = float(getattr(config, "SURFACE_LUMINANCE_TRANSFER", 0.45)
                     if strength is None else strength)
    ref_grey = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    printed_lab = cv2.cvtColor(printed, cv2.COLOR_RGB2LAB).astype(np.float32)
    low_ref = cv2.GaussianBlur(ref_grey, (0, 0), 18.0)
    low_printed = cv2.GaussianBlur(printed_lab[:, :, 0], (0, 0), 18.0)
    detail = printed_lab[:, :, 0] - low_printed
    # 参照物的低频（明暗分布）与纹样的低频按强度混合
    target_low = low_printed * (1.0 - strength) + low_ref * strength
    new_l = target_low + detail
    out = printed_lab.copy()
    out[:, :, 0] = np.where(mask, new_l, printed_lab[:, :, 0])
    result = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return np.where(mask[:, :, None], result, printed).astype(np.uint8)


def design_prompt(folder, palette=None):
    """构造"让模型画出这个角色纹样"的提示词。

    **关键在于必须写进角色的具体配色**（本轮实测的决定性结论）：
      * V0 泛泛要求"角色纹样" -> 得到泛用日式粉樱花 + 灰底，**蓝色占比 0.0%**；
      * V1 把配色写进去（deep navy blue / light blue / gold）->
        **蓝色占比 81.6%**，且最接近角色主导色 `#3a6e9b`；
      * V3 堆了更多纹样词反而变差（0.1%）——词太多会互相干扰。
    因此这里只写"配色 + 少量纹样词"，并且配色**从角色证据实测的调色板**来，
    不写死任何角色名。

    为什么不用 `marks.txt` 的原始中文描述直接拼：那些是给"定位标志"用的，
    里面混着发色/瞳色/刘海等**人物属性**，拼进纹样提示词会把人物画出来。
    这里只取颜色名与少量纹样词，且颜色名来自实测调色板。
    """
    from character import marks as marks_module
    trigger = config.get_trigger(folder)

    def _colour_name(rgb):
        """把 RGB 归到最近的常用色名（只用一张固定表，不依赖模型）。"""
        table = [("deep navy blue", (58, 110, 155)), ("navy blue", (45, 70, 110)),
                 ("light blue", (156, 160, 202)), ("sky blue", (140, 180, 220)),
                 ("gold", (200, 165, 70)), ("light pink", (218, 193, 208)),
                 ("dark purple", (59, 48, 75)), ("white", (245, 245, 245)),
                 ("black", (30, 30, 30))]
        best, best_distance = table[0][0], None
        for name, reference in table:
            distance = sum((int(rgb[i]) - reference[i]) ** 2 for i in range(3))
            if best_distance is None or distance < best_distance:
                best, best_distance = name, distance
        return best

    # **"浅粉/肤色"类颜色不放进纹样提示词**（实测观感问题）：
    # 角色调色板里有一档是浅粉（占比 21%），它主要来自肤色与腮红——
    # 那是**人物**的颜色，不是纹样的颜色。把它写进提示词后生成出来的
    # 纹样掺进大片粉色，成品上出现一团粉色噪点，观感比不加还差。
    # 因此这里只取"纹样会用到"的色（冷色与金色），粉/肤/白都排除。
    skip = {"light pink", "white"}
    colour_words = []
    for item in (palette or [])[:4]:
        name = _colour_name(item["rgb"])
        if name in skip or name in colour_words:
            continue
        colour_words.append(name)
    colour_clause = ", ".join(colour_words) if colour_words else "deep navy blue"

    # **不把标志的中文名拼进提示词**（实测教训）：
    # CLIP 对中文几乎没有理解力，`金色祥云纹`、`中心带图案的黑色前胸甲` 这类词
    # 会白占掉本已紧张的 77 token 额度，还把真正有用的英文词挤掉。
    # 而且这些名字里混着"前胸甲"这种**人体部位**词，会把人物画出来。
    # 纹样类型只用**英文**的通用词（配色才是决定性的，见上文实测）。
    pattern_clause = ", cherry blossom motifs, cloud lines"

    return (f"{trigger}, japanese textile pattern, {colour_clause}, gold accents"
            f"{pattern_clause}, flat design")


def generate_design(pipe, folder, palette=None, device="cuda", seed=None, size=768,
                    text_pipe=None):
    """用**已挂载概念权重**的管线生成一张该角色的平面纹样，返回 (PIL 图, 提示词)。

    这一步是"模型负责画什么"的落点：纹样的形状语言来自概念 LoRA，
    而不是从标志图复制。实测泛泛提示词只会得到泛用图案，因此提示词里
    **必须包含从角色证据实测出的配色**（见 `design_prompt`）。

    `text_pipe`：**由调用方显式传入的文生图管线**（推荐）。
    为什么不让本函数自己装配：`test.py` 里的管线是 img2img，直接对它调用文生图
    会在 `image` 参数上失败（实测 `image must be passed and be one of PIL image...`）；
    而"从 img2img 的 components 现场装配 txt2img"在 `test.py` 的调用路径下
    也失败过一次（同样的 TypeError）。由调用方在自己那边装配好、传进来最可靠。
    未传时才退回现场装配。
    两条管线共享同一份 UNet 与文本编码器，因此**已挂载的 LoRA 一起生效**，无需重载权重。
    """
    import torch
    if text_pipe is None:
        import model_utils
        components = {"pipe": pipe, "device": device,
                      "dtype": next(pipe.unet.parameters()).dtype}
        text_pipe, _with_controlnet = model_utils.build_text2img(
            components, with_controlnet=False)
    prompt = design_prompt(folder, palette)
    negative = getattr(config, "SURFACE_DESIGN_NEGATIVE",
                       "photo, photograph, 3d, person, girl, human, text, watermark, "
                       "blurry, low quality, worst quality")
    # **生成纹样时必须把权重强度调成"只让概念分支说话"**（本轮实测的关键一条）。
    #
    # 实测现象：在 `test.py` 的管线里直接生成，得到的是一张**柔和的人物特写**
    # （紫粉色调、丝带结），完全不是纺织品图案；而单独只挂概念权重、scale=1.0 时
    # 同一个提示词能生成干净的蓝底金花纺织图案。
    # 原因是主链路里 **身份权重 scale=0.9 在主导**，概念只有 0.55——
    # 身份分支学的是"这个角色长什么样"，它会把画面拉回人物。
    # 因此这里临时把身份压到 0、概念提到 1，生成完再恢复，
    # 保证"纹样"这一步只由**学到了视觉特征的概念分支**决定。
    import model_utils
    restore = []
    for name, target in (("default", float(getattr(config, "SURFACE_DESIGN_IDENTITY_SCALE", 0.0))),
                         ("concept", float(getattr(config, "SURFACE_DESIGN_CONCEPT_SCALE", 1.0)))):
        try:
            if name == "default":
                restore.append((name, float(getattr(config, "LORA_SCALE", 0.9))))
            else:
                restore.append((name, float(getattr(config, "CONCEPT_NON_HUMAN_SCALE", 0.55))))
            model_utils.set_adapter_scale(text_pipe, target, adapter_name=name)
        except Exception:      # noqa: BLE001
            pass
    try:
        generator = torch.Generator("cuda" if device == "cuda" else "cpu").manual_seed(
            int(getattr(config, "SURFACE_DESIGN_SEED", 20261008) if seed is None else seed))
        image = text_pipe(prompt=prompt, negative_prompt=negative,
                          num_inference_steps=int(getattr(config, "NUM_INFERENCE_STEPS", 28)),
                          guidance_scale=float(getattr(config, "GUIDANCE_SCALE", 7.0)),
                          height=int(size), width=int(size), generator=generator).images[0]
    finally:
        for name, value in restore:
            try:
                model_utils.set_adapter_scale(text_pipe, value, adapter_name=name)
            except Exception:      # noqa: BLE001
                pass
    return image, prompt


def apply(design, base, mask, palette=None):
    """把生成的平面纹样承印到器物上：柱面 -> 受光 -> 角色配色 -> 明暗关系。

    `design`：模型生成的平面纹样（RGB ndarray，尺寸不限，会被缩放到区域大小）
    `base`  ：器物原图（RGB ndarray）
    `mask`  ：器物区域（bool ndarray，与 base 同尺寸）
    返回 (承印结果, 审计)。
    """
    if design is None or mask is None or not mask.any():
        return base, {"applied": False, "reason": "缺少纹样或器物区域"}
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    height, width = bottom - top, right - left
    if height < 8 or width < 8:
        return base, {"applied": False, "reason": "器物区域过小"}

    design_image = design
    if design_image.shape[0] != height or design_image.shape[1] != width:
        design_image = cv2.resize(design_image, (width, height),
                                  interpolation=cv2.INTER_AREA)

    # 只把纹样放进器物区域内，区域外先保留原图（后续各步也不越界）
    canvas = base.copy()
    region_mask = mask[top:bottom, left:right]
    stage = design_image
    audit = {}

    # 1) 柱面重投影：图案随器物圆润而横向收窄
    if bool(getattr(config, "SURFACE_CYLINDER", True)):
        stage = _cylinder_region(stage, region_mask,
                                 float(getattr(config, "SURFACE_CYLINDER_STRENGTH", 1.0)))
        audit["cylinder"] = getattr(config, "SURFACE_CYLINDER_STRENGTH", 1.0)

    printed = canvas.copy()
    printed[top:bottom, left:right] = stage

    if palette:
        printed = recolour_to_palette(printed, mask, palette,
                                      getattr(config, "SURFACE_RECOLOUR_STRENGTH", 0.6))
        audit["recolour"] = getattr(config, "SURFACE_RECOLOUR_STRENGTH", 0.6)
    if bool(getattr(config, "SURFACE_SHADING", True)):
        printed = flatten_luminance_relation(base, printed, mask,
                                             getattr(config, "SURFACE_LUMINANCE_TRANSFER", 0.45))
        printed = modulate_by_surface(base, printed, mask,
                                      getattr(config, "SURFACE_SHADING_STRENGTH", 0.55))
        audit["shading"] = {"luminance": getattr(config, "SURFACE_LUMINANCE_TRANSFER", 0.45),
                            "modulate": getattr(config, "SURFACE_SHADING_STRENGTH", 0.55)}
    # 边缘羽化：让纹样与器物交界自然过渡，而不是一条硬边
    #
    # **必须同时做到两件事**（实测观感问题）：
    #   1. 沿器物**轮廓**裁剪——只在掩版内落纹样。早期只用外接矩形，
    #      出图上纹样是一块**矩形**，上下是硬直边，明显不是印在器物上；
    #   2. 轮廓边缘**羽化**，让纹样在器物边缘处自然淡出。
    fade = float(getattr(config, "SURFACE_EDGE_FADE", 6.0))
    soft = mask.astype(np.float32)
    if fade > 0:
        # 向内收缩再模糊：羽化带落在器物内侧，不会溢出到背景
        shrink = int(max(1, round(fade)))
        soft_u8 = cv2.erode((mask.astype(np.uint8) * 255),
                            np.ones((shrink * 2 + 1, shrink * 2 + 1), np.uint8))
        if int((soft_u8 > 127).sum()) > 64:
            soft = cv2.GaussianBlur((soft_u8 > 127).astype(np.float32), (0, 0), fade)
    soft = np.clip(soft, 0.0, 1.0)[:, :, None]
    printed = (printed.astype(np.float32) * soft
               + base.astype(np.float32) * (1.0 - soft)).astype(np.uint8)
    audit["edge_fade"] = fade
    audit["edge_feather"] = int(getattr(config, "SURFACE_EDGE_FEATHER", 6))
    audit["applied"] = True
    audit["region"] = [int(left), int(top), int(right), int(bottom)]
    return printed, audit


def _pad_to(region_image, base, top, left):
    """把区域小图放回 base 同尺寸的画布（其余位置填 base 的像素）。"""
    canvas = base.copy()
    height, width = region_image.shape[:2]
    canvas[top:top + height, left:left + width] = region_image[:canvas.shape[0] - top,
                                                               :canvas.shape[1] - left]
    return canvas


def _cylinder_region(region_image, region_mask, strength):
    """在**区域小图**上做柱面重投影（横向按 sin 收窄采样）。

    与 `cylindrical_map` 的区别：本函数直接对区域小图操作，
    不需要与整图同尺寸，调用更简单。
    """
    height, width = region_image.shape[:2]
    if width < 8 or strength <= 0:
        return region_image
    output = np.zeros_like(region_image)
    xs = np.arange(width, dtype=np.float32)
    x_norm = (xs - (width - 1) / 2.0) / max(1.0, width / 2.0)
    angle = np.clip(x_norm, -1.0, 1.0) * (np.pi / 2.0)
    mapped = np.sin(angle) * 0.5 + 0.5
    # 混合：strength=0 时保持线性（不映射）
    position = mapped * strength + (xs / max(1, width - 1)) * (1.0 - strength)
    map_x = (position * (width - 1)).astype(np.float32)
    map_x_2d = np.tile(map_x[None, :], (height, 1))
    map_y_2d = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    warped = cv2.remap(region_image, map_x_2d, map_y_2d,
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # 只保留区域内的部分：区域外的像素稍后会被原图覆盖，这里不必处理
    return warped
