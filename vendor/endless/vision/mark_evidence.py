# -*- coding: utf-8 -*-
"""vision/mark_evidence.py —— 标志证据提取（唯一实现）

规格来源：docs/统一角色标志与自动证据提取_技术调整说明_20261004.md §5 自动识别流程。

处理链（与规格一致）：
    逐图整体观察 -> 按标志提出候选区域 -> 从原始分辨率裁剪 -> 局部核验
    -> 可选分割 -> 跨图匹配 -> 统一证据库

本模块只负责"看"：把标志库里的每条描述拿到每一张原图上问一遍，
产出**逐图证据**（框、可见状态、可见描述、真实裁剪文件）。
不负责训练、不决定生成参数，也不把模型自报的 visibility 当成校准概率。

硬约束（规格 §2 第 6 行、§5.4）：
    * 拿不到有效证据框就不进入标志监督，**不回退到中心裁剪**；
    * 看不到且无法判断原因记 unknown，不自动归因为遮挡；
    * 裁剪必须来自原图原始分辨率的像素，放大只帮助阅读已有像素。

本模块需要 VLM（以及可选的 SAM2）。显存策略：与扩散段**分时**占用，
因此调用方必须在扩散模型加载之前跑完本模块并卸载 VLM（`extract` 内部完成）。
"""
import os

import cv2
import numpy as np

import config

from character import marks as marks_module


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _clip_box(box, min_side=None):
    """把归一化框收进 [0,1] 并保证最小边长；不可用返回 None。"""
    min_side = config.MARK_LOCATE_MIN_SIDE if min_side is None else min_side
    if not box or len(box) != 4:
        return None
    left, top, right, bottom = (float(max(0.0, min(1.0, v))) for v in box)
    if right - left < min_side or bottom - top < min_side:
        return None
    return (left, top, right, bottom)


def _expand(box, pad):
    """把框按边长比例外扩（保留少量上下文，规格 §6）。"""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    return (max(0.0, left - width * pad), max(0.0, top - height * pad),
            min(1.0, right + width * pad), min(1.0, bottom + height * pad))


def _crop(image, box, pad=0.0):
    """按归一化框从原图裁出一块（像素来自原图，不放大碎片）。"""
    box = _clip_box(_expand(box, pad) if pad else box,
                    min_side=config.MARK_LOCATE_MIN_SIDE)
    if box is None:
        return None, None
    width, height = image.size
    left, top, right, bottom = box
    pixels = (int(round(left * width)), int(round(top * height)),
              max(int(round(right * width)), int(round(left * width)) + 1),
              max(int(round(bottom * height)), int(round(top * height)) + 1))
    return image.crop(pixels), pixels


def _flatten_on_white(image):
    """把带透明通道的描边图压到白底上（否则透明像素会变黑，VLM 看不出颜色）。

    用户提供的单独标志图多为 PNG 带 alpha 的矢量描边，`convert("RGB")` 会把
    透明区变成黑色，于是"金色线条"被读成"黑色底上的金线"，颜色描述全错。
    """
    if image.mode in ("RGBA", "LA", "P"):
        converted = image.convert("RGBA")
        from PIL import Image as _Image
        canvas = _Image.new("RGB", converted.size, (255, 255, 255))
        canvas.paste(converted, mask=converted.split()[-1])
        return canvas
    return image.convert("RGB")


def _nontransparent_bbox(image):
    """带 alpha 图里"实际有内容"的范围；没有 alpha 时返回全图。"""
    if image.mode not in ("RGBA", "LA", "P"):
        return (0, 0) + image.size
    converted = image.convert("RGBA")
    bbox = converted.split()[-1].point(lambda v: 255 if v > 8 else 0).getbbox()
    return bbox or ((0, 0) + converted.size)


# ---------------------------------------------------------------------------
# 结构化（中文大白话 -> 结构化 + 规范化英文）
# ---------------------------------------------------------------------------
def structure_mark(client, image, description, is_reference=False):
    """用 VLM 把用户的中文描述结构化成可进入条件的字段。

    VLM 不可用时返回 {"appearance_source": "text_only"}，只保留原文——
    规格要求"识别、学习、迁移位置与最终渲染分别验收"，这里宁可留缺口也不猜结构。
    """
    from vlm.vlm_utils import MARK_STRUCTURE_PROMPT
    result = {"appearance_source": "text_only"}
    if client is None or not getattr(client, "available", False):
        return result
    if is_reference:
        note = ("The user's note about this exact mark (may be Chinese, treat it as the "
                "primary evidence): " + str(description or "").strip())
    else:
        note = ("This image is a crop from one character reference picture; the named mark "
                "appears inside it. The user's note about this exact mark (may be Chinese, "
                "treat it as the primary evidence): " + str(description or "").strip())
    prompt = (f"{MARK_STRUCTURE_PROMPT}\n{note}\n"
              "Now output that JSON object with the fields filled in.")
    try:
        raw = client.analyze_image(image, prompt, temperature=0, max_new_tokens=384)
    except Exception as exc:
        print(f"[mark_evidence] 结构化失败（{description[:20]}）: "
              f"{type(exc).__name__}: {exc}")
        return result
    if not raw:
        return result
    result["appearance_source"] = "vlm"
    for key in ("outline", "internal_topology", "color_relation", "repeat_unit",
                "material", "relative_scale", "description_en", "name_en"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip() and value.strip().lower() not in (
                "unknown", "none", "n/a", "null"):
            result[key] = value.strip()
    for key in ("name_zh",):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
    mark_class = str(raw.get("mark_class") or "").strip().lower()
    if mark_class in marks_module.MARK_CLASSES:
        result["mark_class"] = mark_class
    role = str(raw.get("transfer_role") or "").strip().lower()
    if role in ("emblem", "wearable", "prop", "pattern", "palette"):
        result["transfer_role"] = role
    return result


# ---------------------------------------------------------------------------
# 逐图证据
# ---------------------------------------------------------------------------
def _content_bbox_on_background(image, tolerance=None):
    """按**主导背景色**求内容外接框；看不出背景时退回非透明外接框。

    为什么必须做这一步（本轮实测的根因，代价很大）：
    `reference_crop` 原先只按**透明**范围裁（`_nontransparent_bbox`），
    而用户给的参考图是**白底不透明**的（例如 `images/021.png` =
    白底 + 黑色樱花 + 金色花心）。对不透明图，"非透明外接框"就是**整张图**，
    于是 `ref_*.png` 原样保留整幅画面；如果那张参考图是**整张插画**，
    承载阶段就会把整幅角色立绘平铺到器物上——实测出图上出现了
    九宫格的完整角色插画，这是最严重的异常。

    做法：取图像四边像素的**中位色**当作主导背景色，把与它接近的像素视作背景，
    返回剩余内容的外接框。这是纯统计判据，不依赖语义模型。
    """
    bbox, _content, _background, _largest = _analyse_reference(image, tolerance)
    return bbox


def _analyse_reference(image, tolerance=None):
    """分析参考图：返回 (内容外接框, 内容掩版, 背景色, 平面度)。

    判"这张参考图是**干净标志**还是**整张插画**"这件事，实测排除了两个判据：

      * **内容占画布比例**：区分不了。`021.png`（白底黑樱花）占 85% 且完全可用，
        `006.jpg`（角色立绘）占 100% 不可用——比例几乎一样；
      * **最大内容连通块占比**：也区分不了。`022.png` 是**细线条金色徽记**，
        线条彼此分离，最大连通块只占内容的 18%；而插画占 99%。
        两者刚好**反着来**，用它判定会把好素材（细线徽记）误判为插画。

    真正稳定的差别是**"平面图形" vs "渲染插画"**：
      * 干净标志是**平面矢量感**的——颜色数极少（白底 + 一两种色），
        色阶是阶跃的；
      * 插画有渐变、阴影、光照，颜色数成百上千。
    因此这里量"**去底后内容里的显著颜色数**"：标志通常 < 40，插画远高于此。
    这个判据只看直方图，与形状、连通性、占比都无关。
    """
    import numpy as _np
    array = np.asarray(image.convert("RGB")).astype(np.int16)
    height, width = array.shape[:2]
    if array.size == 0:
        return (0, 0, width, height), None, None, 0.0
    border = np.concatenate([
        array[0, :, :], array[-1, :, :], array[:, 0, :], array[:, -1, :]], axis=0)
    background = np.median(border, axis=0)
    distance = np.abs(array - background).max(axis=2)
    cut = float(config.MARK_REF_BG_TOLERANCE if tolerance is None else tolerance)
    content = distance > cut
    if content.sum() < 64:
        bbox = _nontransparent_bbox(image)
        return bbox, content, background, 0.0
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    bbox = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)

    # 显著颜色数：把颜色量化到 16 级的立方体后再数非空格子，抗轻微噪声
    pixels = array[content]
    if pixels.size == 0:
        return bbox, content, background, 0.0
    if pixels.shape[0] > 200000:
        step = pixels.shape[0] // 200000 + 1
        pixels = pixels[::step]
    quantised = (pixels // 16).astype(np.int32)
    keys = quantised[:, 0] * 256 * 256 + quantised[:, 1] * 256 + quantised[:, 2]
    distinct = int(np.unique(keys).size)
    return bbox, content, background, float(distinct)


def reference_crop(folder, rel_path, description):
    """把单独标志参考图裁成"只有这个标志"的一张图并落盘。

    用户给的参考图有两类，必须分别对待（本轮实测）：
      * **干净标志图**（白底 + 图案本身，如 `021.png` 的黑色樱花）：
        按主导背景色去掉白底后裁到内容外接框即可得到干净素材；
      * **整张插画**（如 `006.jpg` 是角色立绘）：这类图**不是**可用的贴图素材，
        它只是"这个标志长什么样"的样例。裁出来仍是一整幅画，
        因此这里如实标注 `is_artwork`，由承载阶段决定不用它。

    规格要求每个标志保存"真实裁剪路径"，因此仍然落盘——但**不假装它是干净素材**。
    """
    import image_utils
    path = os.path.join(config.DATASET_DIR, folder, rel_path.replace("/", os.sep))
    if not os.path.isfile(path):
        return None, "参考图不存在"
    image = image_utils.load_image(path)
    bbox, content, _background, distinct = _analyse_reference(image)
    box = bbox
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return None, "参考图内容范围过小"
    cropped = image.crop(box)
    # **判"插画"看"平面图形 vs 渲染插画"，不看占比也不看连通性**（实测两轮才理清）：
    #   * `021.png`（白底黑樱花）占 85% 却完全可用；`006.jpg`（立绘）占 100% 不可用
    #     -> 占比区分不了；
    #   * `022.png` 是**细线条金色徽记**，最大连通块只占内容 18%；
    #     插画占 99% -> 连通性刚好反着来，会把好素材误判为插画。
    # 稳定的判据是**显著颜色数**：干净标志是平面矢量感（白底 + 一两种色），
    # 插画有渐变/阴影/光照，颜色数成百上千。
    max_colors = float(getattr(config, "MARK_REF_MAX_COLORS", 60.0))
    coverage = float(content.mean())
    note = ""
    if distinct > max_colors:
        note = (f"去底后显著颜色数 {distinct:.0f} > {max_colors:.0f}，"
                f"判为**渲染插画**而不是平面标志；该参考图只作样例，不用于平铺")
    elif coverage > 0.97:
        note = (f"内容占整幅 {coverage:.0%}，几乎无背景分隔，"
                f"判为**整图**而不是孤立标志")
    padding = int(round(max(cropped.size) * config.MARK_REF_CROP_PAD))
    from PIL import Image as _Image
    canvas = _Image.new("RGB", (cropped.size[0] + 2 * padding,
                                cropped.size[1] + 2 * padding), (255, 255, 255))
    canvas.paste(_flatten_on_white(cropped), (padding, padding))
    target = os.path.join(marks_module.reference_crop_dir(folder),
                          f"ref_{os.path.splitext(os.path.basename(rel_path))[0]}.png")
    image_utils.save_image(canvas, target)
    return target, note


def _location_query(folder, rel_path, mark, image, client, subject):
    """按标志的实体形态决定查询方式。

    * locatable ：问"这个标志在不在这张图里、在哪"（要框 + 可见状态）；
    * describable：问"这个外观属性在不在这张图里、这里看起来是什么"（只要可见状态与描述，
      因为配色/发色/瞳色/发型剪裁没有唯一的位置框，逼模型给框只会得到退化的宽框）。

    返回 (查询结果, 查询文本, 允许区域列表)。
    """
    description = str(mark.get("description") or "")
    english = str(mark.get("description_en") or "").strip()
    regions = [str(r) for r in (mark.get("regions") or [mark.get("region") or "whole"])]
    hint = "" if regions == ["whole"] else ", ".join(regions)
    if str(mark.get("entity_mode") or "locatable") == "describable":
        target = (f"{english or description} (the character's appearance property: "
                  f"{description})")
        return client.observe_trait(image, target, region_hint=hint), target, regions
    if regions == ["whole"]:
        # 角色级通用象征标志：搜索整个图集（用户人工标注约定的第一条）。
        target = (f"the character mark described as: {english or description}. "
                  "It is part of this character's overall design, so it may appear "
                  "anywhere on the character or the outfit")
    else:
        target = (f"the character mark described as: {english or description}. "
                  f"It belongs to the {', '.join(regions)} of the character")
    return client.locate_mark(image, target, region_hint=hint), target, regions


def _region_boxes(client, image, subject, rel_path):
    """该图的主体区域框（人物：头/躯干；非人物：身体/头/颈）。

    只用于**位置可信度检查**：规格 §2 第 4 行明确要求人体区域只能做搜索范围，
    不能充当标志定位结果。这里正是把它当范围用。
    """
    from vision.subject_analyzer import locate_regions
    try:
        boxes = locate_regions(client, image, subject) or {}
    except Exception as exc:
        print(f"[mark_evidence] {rel_path}: 区域框不可用（{type(exc).__name__}: {exc}）")
        return {}
    return {name: info.get("box") for name, info in boxes.items() if info.get("box")}


def _plausible_location(box, regions, region_boxes) -> tuple:
    """标志框是否落在它声明/暗示的部位里。

    返回 (是否可信, 说明)。拿不到区域框时返回可信（无依据就不否决），
    但会在说明里写清"没有区域框可核对"。

    为什么需要这一步：实测里小徽记会被定位到完全不相干的地方（模型对"金色祥云纹"
    这样泛化的描述会给一个"大概在这"的宽框）。这类证据一旦进入标志监督，
    学的就是背景而不是标志，因此宁可记 unknown。

    通过检查时也返回说明（而不是空串）：否则事后无法区分"检查过并通过"与"根本没检查"。
    判据精度如实写明——区域框只有 head / body（视觉模型对精细部位给不出稳定框），
    因此这条检查只能区分"头/脸/眼部"与"躯干及以下"，**不能**区分胸与腰。
    """
    if box is None or not regions or regions == ["whole"]:
        return True, "该标志不限定部位，未做位置可信度检查"
    if not region_boxes:
        return True, "没有区域框可核对，未做位置可信度检查"
    allowed = set()
    for region in regions:
        allowed.update(_REGION_TO_PART.get(region, ("head", "body", "neck")))
    candidates = [region_boxes[name] for name in allowed if region_boxes.get(name)]
    if not candidates:
        return True, "该部位没有可核对的区域框，未做位置可信度检查"
    left, top, right, bottom = box
    area = max(1e-9, (right - left) * (bottom - top))
    overlap = 0.0
    for other in candidates:
        width = max(0.0, min(right, other[2]) - max(left, other[0]))
        height = max(0.0, min(bottom, other[3]) - max(top, other[1]))
        overlap = max(overlap, (width * height) / area)
    if overlap < config.MARK_REGION_OVERLAP_MIN:
        return False, (f"定位框只有 {overlap:.0%} 落在声明的部位 {regions} 内"
                       f"（低于 {config.MARK_REGION_OVERLAP_MIN:.0%}），已记为 unknown")
    # 通过检查时也留一条审计：否则事后无法区分"检查过并通过"与"根本没检查"。
    # 判据精度也必须写清楚——区域框只有 head / body（视觉模型对精细部位给不出稳定框），
    # 所以这条检查实际只能区分"头/脸/眼部"与"躯干及以下"，**不能**区分胸与腰。
    return True, (f"位置检查通过（框 {overlap:.0%} 落在 {sorted(allowed)} 范围内；"
                  f"区域框只有 head/body，仅能区分头部与躯干，不能区分胸与腰）")


def extract(folder, client=None, image_limit=0, library=None, observations=None,
            reuse_boxes=False) -> dict:
    """跑完整个证据提取链，写出统一标志库与逐图证据。返回结果摘要。

    参数：
        client      已加载的 VLMClient；None 时只做参考图落盘与逐图观察（不定位）
        image_limit >0 时只用前 N 张图（冒烟测试）
        library     已有标志库（None=重新解析两份 txt）
        observations 已有的逐图证据（None=重新提取）
        reuse_boxes True=复用已记录的真实框**只重建裁剪文件**，不再问模型。
                    用途：裁剪文件被清理/迁移后重新生成像素证据（实测踩过：
                    测试清理把 crop_*.png 删掉之后，逐图证据里的路径全部指向不存在的文件，
                    而"已有可用证据就跳过"的逻辑又不会重新裁剪，于是永远补不回来）。

    返回 {"library": ..., "observations": [...], "reasons": [...], "stats": {...}}
    """
    import image_utils
    import dataset as dataset_module

    reasons = []
    library = library or marks_module.load_library(folder) or marks_module.parse_marks(folder)
    if not library.get("marks"):
        reasons.append("标志库为空：dataset/<角色>/marks.txt 与 mark_refs.txt 都没有可用描述")
        library["input_fingerprint"] = marks_module.input_fingerprint(folder)
        marks_module.save_library(folder, library)
        marks_module.save_observations(folder, [])
        return {"library": library, "observations": [], "reasons": reasons,
                "stats": {"marks": 0, "observations": 0}}

    # ---- 0. 单独标志参考图：本身就是一条确定的证据配对 ----
    observations = list(observations or [])
    by_reference = {}
    for reference in library.get("references") or []:
        for link in library.get("links") or []:
            if link.get("rel_path") != reference.get("rel_path"):
                continue
            target_mark = next((m for m in library["marks"]
                                if m["mark_id"] == link.get("mark_id")), None)
            if target_mark is None:
                continue
            # **用解析后的实际路径**，不要用用户写的原始 rel_path。
            # 实测事故：`mark_refs.txt` 写 `images/005.png`、磁盘上是 `005.jpg`，
            # 跨扩展名解析修好之后，这里如果仍然传 `rel_path`（原始写法），
            # `reference_crop` 拼出来的路径依旧不存在，
            # 4 条单独标志参考图**全部继续报"参考图不存在"**——
            # 上游修好了、下游还在用旧值，问题表现为"修了但没生效"。
            crop_path, note = reference_crop(
                folder, reference.get("resolved_rel_path") or reference["rel_path"],
                reference.get("description") or "")
            if crop_path is None:
                reasons.append(f"单独标志参考图不可用（{reference['rel_path']}）: {note}")
                continue
            by_reference[target_mark["mark_id"]] = {
                "rel_path": reference["resolved_rel_path"],
                "crop_path": crop_path,
                "description": reference.get("description") or "",
                "mark_id": target_mark["mark_id"],
                "visibility": "visible",
                "completeness": "complete",
                "box": None,
                "source": "mark_refs.txt",
                "note": "用户提供的单独标志参考图：本身就是一条确定的证据配对",
            }
            evidence = {
                "rel_path": reference["resolved_rel_path"],
                "kind": "reference_image",
                "crop_path": os.path.relpath(crop_path, config.PROJECT_ROOT).replace("\\", "/"),
                "box": None,
                "visibility": "visible",
                "completeness": "complete",
                "quality": "reference",
                "source": "mark_refs.txt",
                "model": "user_input",
                "prompt_version": config.MARK_PROMPT_VERSION,
            }
            if evidence not in target_mark["evidence"]:
                target_mark["evidence"].append(evidence)
            target_mark["evidence_confidence"] = "reference_image"
            break

    # ---- 1. 逐图整体观察：覆盖全部有效原图（规格 §2 第 7 行）----
    images = dataset_module.scan_images(folder)
    if image_limit and image_limit > 0:
        images = images[:image_limit]
    image_rows = {}
    captions = dataset_module.load_character_captions(folder)
    for path in images:
        name = os.path.basename(path)
        tags = [str(t) for t in (captions.get(name) or [])]
        image_rows[name] = {
            "rel_path": name, "view": next((t for t in tags if t.endswith("view")), ""),
            "framing": next((t for t in tags if t in (
                "full body shot", "upper body", "face close-up", "clothing close-up",
                "feet close-up", "accessory close-up", "pattern close-up")), ""),
            "tags": tags, "source": "captions.txt" if tags else "unlabelled",
        }
    print(f"[mark_evidence] 逐图观察覆盖 {len(image_rows)} 张原图（全部有效原图，不只代表图）")

    # ---- 2. 按标志提出候选区域 -> 3. 原始分辨率裁剪 -> 4. 局部核验 ----
    existing = {}
    for row in observations:
        if row.get("source") != "mark_refs.txt":
            existing.setdefault((row.get("rel_path"), row.get("mark_id")), row)

    located, unlocated = 0, 0
    if reuse_boxes and observations:
        # 只重建裁剪文件：逐图证据里的框就是真实证据，重新裁一次即可，
        # 不必再问模型（也避免"已有证据就跳过"的逻辑把这一步永久跳过）。
        # **必须在这里直接返回**：继续往下走会进入"VLM 不可用 -> 全部记 unknown"的分支，
        # 把已经提取好的逐图证据整批覆盖掉（实测踩过：205 条好证据被 462 条
        # "VLM 不可用"覆盖）。
        rebuilt = _rebuild_crops(folder, library, observations)
        library["input_fingerprint"] = marks_module.input_fingerprint(folder)
        print(f"[mark_evidence] 已按已记录的真实框重建 {rebuilt} 条裁剪文件")
        return {"library": library, "observations": list(observations),
                "reasons": reasons,
                "stats": {"marks": len(library["marks"]),
                          "observations": len(observations), "rebuilt_crops": rebuilt,
                          "located": sum(1 for r in observations
                                         if r.get("visibility") in ("visible", "partial")),
                          "unlocated": sum(1 for r in observations
                                           if r.get("visibility") not in
                                           ("visible", "partial")),
                          "with_crop": sum(1 for r in observations if r.get("crop_path")),
                          "reference_crops": len(by_reference)}}
    sam = None
    if client is not None and getattr(client, "available", False) and config.USE_SAM:
        try:
            from vision.segmentation import load_sam2
            sam = load_sam2()
        except Exception as exc:
            reasons.append(f"SAM2 不可用（{type(exc).__name__}: {exc}），只保留 VLM 框")
    try:
        for index, path in enumerate(images, start=1):
            name = os.path.basename(path)
            row = image_rows[name]
            image = image_utils.load_image(path)
            subject = {"subject_type": "human" if any(
                t in ("full body shot", "upper body", "face close-up") for t in row["tags"])
                else "unknown", "visible_regions": [], "framing": row["framing"],
                "style": ""}
            if client is None or not getattr(client, "available", False):
                for mark in library["marks"]:
                    key = (name, mark["mark_id"])
                    existing[key] = _observation(
                        name, mark, visibility="unknown",
                        note="VLM 不可用，未做定位；保留缺口而不是猜位置",
                        source="vlm_unavailable")
                continue
            # 区域框只算一次（同一张图的所有标志共用），作为"位置可信度检查"的范围
            region_boxes = _region_boxes(client, image, subject, name)

            # ---- 阶段 A：把这张图里所有还没有证据的标志**逐条**问一遍 ----
            # 先把候选全部收齐再统一裁决（阶段 B），不能问到哪条就定哪条：
            # 实测（2026-10-05）视觉模型会把**同一个框**回报给多条不同的标志
            # （002.png 里腰结 m10、背后蝴蝶结 m17、扇子 m18 都拿到 0.655,0.454,1.0,0.649；
            # 003.png 里胸前甲 m12 拿到的是项圈 m11 的框）。逐条即时落库时，
            # 后一条会把前一条正确的框"顶掉"，而且冲突只能靠部位检查事后否决——
            # 那会连**正确的那一条**一起丢掉。
            candidates = []
            for mark in library["marks"]:
                if str(mark.get("entity_mode") or "locatable") != "locatable":
                    continue
                key = (name, mark["mark_id"])
                if key in existing and existing[key].get("visibility") in (
                        "visible", "partial") and existing[key].get("box"):
                    continue                       # 已有可用证据，不重复问（省显存时间）
                if _evidence_is_sufficient(mark, existing):
                    # 已经有足够多张不同原图的真实裁剪：训练侧每个标志最多只用
                    # MARK_MAX_CROPS_PER_MARK 条（规格 §6 的均衡抽样），
                    # 继续问下去不会增加训练语料，只增加耗时。
                    continue
                located_result, query, regions = _location_query(
                    folder, name, mark, image, client, subject)
                if not located_result:
                    existing[key] = _observation(
                        name, mark, visibility="unknown",
                        note="模型未返回可解析结果", source="vlm")
                    unlocated += 1
                    continue
                visibility = located_result.get("visibility") or "unknown"
                box = _clip_box(located_result.get("box"))
                if box is None and visibility in ("visible", "partial"):
                    visibility = "unknown"        # 声称可见却拿不到有效框：不进入监督
                if box is None:
                    existing[key] = _observation(
                        name, mark, visibility=visibility,
                        visible_description=located_result.get("description") or "",
                        note="", source="vlm",
                        model=os.path.basename(config.VLM_PATH))
                    unlocated += 1
                    print(f"[mark_evidence] {index}/{len(images)} {name} "
                          f"{mark['mark_id']} -> {visibility}", flush=True)
                    continue
                candidates.append({
                    "mark": mark, "key": key, "box": box, "visibility": visibility,
                    "regions": regions,
                    "description": located_result.get("description") or "",
                    "plausible": _plausible_location(box, regions, region_boxes),
                })

            # ---- 阶段 B：按框分组做"一个框只归一条标志"的确定性裁决 ----
            assignments, collisions, index_groups = _resolve_collisions(candidates)
            # `_resolve_collisions` 返回的是候选**下标**分组，这里必须映射回候选对象
            # （直接把下标列表传给消歧函数会得到 'int' object is not subscriptable）。
            groups = [[candidates[i] for i in group] for group in index_groups]
            # 撞框组先问一次消歧：模型可能确实分得清"腰结"与"背后蝴蝶结"，
            # 只是第一次提问时用了同一个热区。问不出来就保持 unknown（不挑赢家）。
            resolved = {}
            for group in groups:
                if len(group) < 2:
                    continue
                resolved.update(_disambiguate_collision(client, image, group, region_boxes,
                                                        index, len(images)))
            for item in candidates:
                mark, key = item["mark"], item["key"]
                plausible, why = item["plausible"]
                resolution = assignments.get(mark["mark_id"])
                note_parts = []
                location_audit = why if plausible else ""
                visibility, box = item["visibility"], item["box"]
                if not plausible:
                    visibility, box = "unknown", None
                    note_parts.append(why)
                elif resolution == "collision" and mark["mark_id"] in resolved:
                    # 消歧成功：这一条拿到了框，组内其他成员仍然记 unknown
                    box = resolved[mark["mark_id"]]
                    location_audit = (f"撞框消歧后由模型指认（组："
                                      f"{collisions.get(mark['mark_id'], '?')}）")
                    note_parts.append("撞框组内由模型消歧指认（其余成员保持 unknown）")
                elif resolution == "collision":
                    # 多条标志抢同一个框、且消歧问不出结果：**都不采信**，
                    # 如实记 unknown 与冲突说明（规格 §5.7）。
                    visibility, box = "unknown", None
                    note_parts.append(
                        f"与同一张图上的另一条标志定位到同一个区域"
                        f"（撞框组 {collisions.get(mark['mark_id'], '?')}），无法确定归属")
                crop_path, pixels, quality = None, None, ""
                if box is not None:
                    cropped, pixels = _crop(image, box, pad=config.MARK_CONTEXT_PAD)
                    if cropped is None:
                        quality = "框不可用"
                    else:
                        quality, crop_why = _crop_quality(cropped)
                        if quality == "ok":
                            target = os.path.join(
                                marks_module.reference_crop_dir(folder),
                                f"crop_{os.path.splitext(name)[0]}_{mark['mark_id']}.png")
                            image_utils.save_image(cropped, target)
                            crop_path = os.path.relpath(target, config.PROJECT_ROOT) \
                                .replace("\\", "/")
                        else:
                            note_parts.append(crop_why)
                existing[key] = _observation(
                    name, mark, visibility=visibility, box=box,
                    visible_description=item["description"],
                    crop_path=crop_path, pixels=pixels, quality=quality,
                    note="；".join(note_parts), location_audit=location_audit, source="vlm",
                    model=os.path.basename(config.VLM_PATH))
                if visibility in ("visible", "partial"):
                    located += 1
                else:
                    unlocated += 1
                print(f"[mark_evidence] {index}/{len(images)} {name} "
                      f"{mark['mark_id']} -> {visibility}"
                      f"{'（裁剪 ' + str(pixels[2] - pixels[0]) + '×' + str(pixels[3] - pixels[1]) + '）' if pixels else ''}"
                      f"{'  ' + note_parts[0] if note_parts else ''}", flush=True)
    finally:
        if sam is not None:
            try:
                from vision.segmentation import unload_sam
                unload_sam(sam)
            except Exception:
                pass

    # ---- 5. 跨图匹配：把同一标志的证据收进它的 evidence 列表（真实裁剪，逐图记录）----
    # 注意：逐图可见性统计在阶段 B 已经按"撞框组"裁决过，这里只负责把**采信过的**
    # 证据挂到标志上，不能反过来再放宽可见状态（否则撞框裁决会被这一步抹掉）。
    for mark in library["marks"]:
        locatable = str(mark.get("entity_mode") or "locatable") == "locatable"
        for key, row in existing.items():
            if key[1] != mark["mark_id"]:
                continue
            entry = {
                "rel_path": row["rel_path"], "kind": "original_crop",
                "crop_path": row.get("crop_path"), "box": row.get("box"),
                "visibility": row["visibility"], "completeness": row.get("completeness"),
                "quality": row.get("quality") or "", "source": row.get("source"),
                "model": row.get("model") or "", "prompt_version": config.MARK_PROMPT_VERSION,
            }
            if not locatable:
                # 外观属性没有位置框：把该图的可见描述作为证据保留（可以写进条件），
                # 但绝不产出一条"假的裁剪样本"（规格 §2 第 6 行）。
                entry["kind"] = "trait_observation"
                entry["visible_description"] = row.get("visible_description") or ""
                if entry["visible_description"] and entry not in mark["evidence"]:
                    mark["evidence"].append(entry)
                continue
            if row.get("crop_path") and entry not in mark["evidence"]:
                mark["evidence"].append(entry)
        usable = [e for e in mark["evidence"] if e.get("crop_path")]
        if usable:
            mark["evidence_confidence"] = ("reference_image"
                                           if any(e["kind"] == "reference_image" for e in usable)
                                           else "image_evidence")

    # ---- 6. 每条标志的可见性统计与缺口 ----
    unresolved = []
    for mark in library["marks"]:
        locatable = str(mark.get("entity_mode") or "locatable") == "locatable"
        counts = {}
        for key, row in existing.items():
            if key[1] != mark["mark_id"]:
                continue
            counts[row["visibility"]] = counts.get(row["visibility"], 0) + 1
        mark["visibility_counts"] = counts
        mark["observed_image_count"] = counts.get("visible", 0) + counts.get("partial", 0)
        if not locatable:
            continue                     # 外观属性的缺口在下面按"每图可见状态"单独报
        if not [e for e in mark["evidence"] if e.get("crop_path")]:
            unresolved.append(
                f"标志「{mark['name']}」({mark['mark_id']}) 没有任何可用裁剪证据"
                f"（逐图可见状态：{_counts_text(counts)}）：不进入标志监督、不进入条件文本，"
                f"训练不会为它编造位置")
        elif str(mark.get("scope")) == "whole" and not mark["observed_image_count"]:
            unresolved.append(
                f"标志「{mark['name']}」被声明为角色级通用象征标志，"
                f"但没有任何一张图给出可见证据（{_counts_text(counts)}）")

    observations = [existing[key] for key in sorted(existing, key=lambda k: (k[0], k[1]))]
    library["unresolved_gaps"] = unresolved
    library["input_fingerprint"] = marks_module.input_fingerprint(folder)
    stats = {"marks": len(library["marks"]), "observations": len(observations),
             "located": located, "unlocated": unlocated,
             "with_crop": sum(1 for row in observations if row.get("crop_path")),
             "reference_crops": len(by_reference)}
    return {"library": library, "observations": observations, "reasons": reasons,
            "stats": stats, "images": image_rows}


def _counts_text(counts):
    return ", ".join(f"{k}={v}" for k, v in sorted((counts or {}).items())) or "无记录"


def _iou(left, right) -> float:
    """两个归一化框的交并比。"""
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    inter = width * height
    if inter <= 0:
        return 0.0
    area_left = max(1e-9, (left[2] - left[0]) * (left[3] - left[1]))
    area_right = max(1e-9, (right[2] - right[0]) * (right[3] - right[1]))
    return inter / (area_left + area_right - inter)


# 部位 -> 区域框名。只用于"这条标志该在哪个范围里"的可信度检查；
# 区域框只有 head / body（视觉模型对精细部位给不出稳定框）。
_REGION_TO_PART = {"head": ("head",), "face": ("head",), "eyes": ("head",),
                   "neck": ("head", "body"), "chest": ("body",), "torso": ("body",),
                   "waist": ("body",), "back": ("body",), "arms": ("body",),
                   "hands": ("body",), "legs": ("body",), "feet": ("body",),
                   "surface": ("body", "head")}


def resolve_output_path(rel_path):
    """把落盘的证据路径解析成绝对路径。

    实测踩过：`crop_path` 是以 `config.OUTPUT_DIR` 为基准的相对路径
    （`os.path.relpath(mark_evidence.reference_crop_dir(folder), config.OUTPUT_DIR)`），
    而读取端曾经一律拼 `config.PROJECT_ROOT`，于是得到
    `.../output/marks/...` 这种把 output 重复一次的路径，直接报"图片文件不存在"。
    这里按"绝对路径 -> PROJECT_ROOT -> OUTPUT_DIR -> 当前目录"依次尝试，
    并把最终结果**校验存在**；都不存在时返回 None，由调用方记缺口。
    """
    text = str(rel_path or "").strip()
    if not text:
        return None
    text = text.replace("\\", "/")
    if os.path.isabs(text):
        return text if os.path.isfile(text) else None
    for base in (config.PROJECT_ROOT, config.OUTPUT_DIR, os.getcwd()):
        candidate = os.path.normpath(os.path.join(base, text))
        if os.path.isfile(candidate):
            return candidate
    return None


def _rebuild_crops(folder, library, observations) -> int:
    """按逐图证据里已记录的真实框重新裁剪落盘，返回重建条数。

    为什么需要单独一条路径：正常提取流程里"已有可用证据就跳过"，
    因此裁剪文件一旦丢失（被清理、迁移、或测试删掉），那条证据的 `crop_path`
    就永远指向不存在的文件——而可见状态与框都还在，流程不会报错，
    直到下游读文件时才失败（实测：留出对照因此退化为"证据不足"）。
    这里用**已记录的框**重新裁一次，属于重放而不是重新识别。
    """
    import image_utils

    marks_by_id = {mark["mark_id"]: mark for mark in library.get("marks") or []}
    rebuilt = 0
    for row in observations:
        box = row.get("box")
        rel_path = row.get("rel_path")
        mark = marks_by_id.get(row.get("mark_id"))
        if not box or not rel_path or mark is None:
            continue
        if row.get("visibility") not in ("visible", "partial"):
            continue
        image_path = os.path.join(config.DATASET_DIR, folder, "images", rel_path)
        if not os.path.isfile(image_path):
            continue
        cropped, pixels = _crop(image_utils.load_image(image_path), box,
                                pad=config.MARK_CONTEXT_PAD)
        if cropped is None:
            continue
        quality, _why = _crop_quality(cropped)
        if quality != "ok":
            continue
        target = os.path.join(
            marks_module.reference_crop_dir(folder),
            f"crop_{os.path.splitext(rel_path)[0]}_{mark['mark_id']}.png")
        image_utils.save_image(cropped, target)
        row["crop_path"] = relative_crop_path(target)
        row["pixels"] = list(pixels)
        row["quality"] = "ok"
        rebuilt += 1
    return rebuilt


def _evidence_is_sufficient(mark, existing) -> bool:
    """这条标志是否已经攒够训练用的真实裁剪。

    判据只看**已有可用裁剪的条数**（quality=ok 且带 crop_path），
    因为训练侧每个标志最多取 `MARK_MAX_CROPS_PER_MARK` 条（规格 §6 的均衡抽样）。
    攒够之后继续逐图提问不会增加训练语料，只增加耗时——22 张图 × N 条标志的
    全量问答是小时级开销，而其中大部分答案不会被用到。

    注意：这里只影响"要不要再问"，**不影响已经得到的可见性统计**；
    没有攒够的标志仍然会在每一张图上被问到。
    """
    usable = 0
    for row in existing.values():
        if row.get("mark_id") != mark.get("mark_id"):
            continue
        if row.get("quality") == "ok" and row.get("crop_path"):
            usable += 1
    return usable >= int(config.MARK_MAX_CROPS_PER_MARK)


# 撞框消歧：把"同一区域里到底哪一条标志"变成一次**显式提问**。
#
# 为什么需要（2026-10-05 实测）：模型对相似标志会固定回同一个热区——
# 001.png 里"腰前绳结"与"背后蝴蝶结"、"金色祥云纹"与"裙上祥云纹"、
# "金色项圈"与"脖上金色挂坠"各成一组撞框。直接全部记 unknown 会把
# **本来正确的那一条**也丢掉（腰结在 001 上是真的可见的）。
# 与其把歧义留在沉默里，不如把候选项摆给模型让它选一次；
# 仍然选不出来（null / 非法 / 又撞）就保持 unknown。
MARK_DISAMBIGUATE_PROMPT = (
    "In this image, which ONE of the following character marks is the one visible at the "
    "region around {region}?\n{candidates}\n"
    "Return ONLY JSON:\n"
    '{{"chosen": "<exact id from the list>" or null,'
    '"reason":"one short ENGLISH phrase explaining what you actually see there"}}\n'
    "Choose an id only when you can actually see that specific mark there. "
    "If you cannot tell which one it is, return null. Never guess."
)


def _disambiguate_collision(client, image, group, region_boxes, index, total):
    """对一个撞框组做一次消歧提问；返回 {mark_id: box 或 None}。

    只有组内成员的**声明部位彼此兼容**才值得问——如果两条标志属于互斥的部位
    （例如一条在头部、一条在躯干），那么无论模型怎么答都不该采信，
    因为同一块像素不可能同时是两处的东西。
    """
    if client is None or not getattr(client, "available", False) or len(group) < 2:
        return {}
    allowed_sets = []
    for item in group:
        allowed = set()
        for region in item["regions"]:
            allowed.update(_REGION_TO_PART.get(region, ("head", "body", "neck")))
        allowed_sets.append(allowed)
    if not all(allowed == allowed_sets[0] for allowed in allowed_sets):
        print("[mark_evidence] 撞框组内的声明部位互斥，不做消歧（同一块像素不可能属于两处）",
              flush=True)
        return {}
    box = group[0]["box"]
    region_text = (f"x {box[0]:.2f}~{box[2]:.2f}, y {box[1]:.2f}~{box[3]:.2f} "
                   f"(normalized, 0~1)")
    lines = []
    for item in group:
        mark = item["mark"]
        descriptor = (str(mark.get("description_en") or "").strip()
                      or str(mark.get("description") or "").strip())
        lines.append(f'- id "{mark["mark_id"]}": {mark.get("name") or ""} — {descriptor}')
    prompt = MARK_DISAMBIGUATE_PROMPT.format(region=region_text,
                                             candidates="\n".join(lines))
    try:
        result = client.analyze_image(image, prompt, temperature=0, max_new_tokens=160)
    except Exception as exc:
        print(f"[mark_evidence] 撞框消歧失败: {type(exc).__name__}: {exc}", flush=True)
        return {}
    chosen = result.get("chosen") if isinstance(result, dict) else None
    print(f"[mark_evidence] {index}/{total} 撞框消歧：模型选择 {chosen!r}"
          f"（理由：{str(result.get('reason') or '')[:60]}）", flush=True)
    chosen = _extract_mark_id(chosen, [item["mark"]["mark_id"] for item in group])
    if chosen is None:
        # 模型返回了不在候选里的 id（或整段说明文字）：视为无法判定，不做任何采纳
        print("[mark_evidence] 撞框消歧没有给出候选内的 id，保持 unknown", flush=True)
        return {}
    return {chosen: box}


def _extract_mark_id(answer, valid_ids):
    """从模型回答里取出候选内的 mark_id。

    实测（2026-10-05）：模型有时不返回裸 id，而是把整行候选原样抄回来，例如
    `id "m19_b35777a2": 白色樱花图案 — 白色樱花图案`。要求"完全相等"会把这些
    **本来正确**的指认全部丢掉（当轮就有 3 次），因此这里只做一件事：
    在回答里找候选 id，返回其中**最长**的那个（id 含哈希后缀，不会互相包含，
    取最长只是为了在极端情况下确定性地挑一个）。找不到任何候选 id 才返回 None。
    """
    if not isinstance(answer, str):
        return None
    text = answer.strip().strip('"').strip()
    found = [mark_id for mark_id in valid_ids if mark_id in text]
    if not found:
        return None
    return max(found, key=len)


def _resolve_collisions(candidates) -> tuple:
    """一组候选框里，"一个框只归一条标志"的确定性裁决。

    为什么必须做（2026-10-05 实测）：视觉模型会把**同一个框**回报给多条不同的标志。
    逐条即时落库时，后一条会覆盖/顶掉前一条正确的框，而冲突只能靠部位检查事后否决
    ——那会连正确的那一条一起丢掉（002.png 里腰结拿到正确的框，却因为"背后蝴蝶结"
    也报了同一个框而被撞成 unknown）。

    裁决规则（保守、可复现、不挑赢家）：
      * 两条以上标志的框 IoU 超过 `MARK_BOX_COLLISION_IOU` -> 归为**同一个撞框组**；
      * 组内先做一次显式消歧提问（`_disambiguate_collision`）：模型能指认出
        具体是哪一条就采纳那一条；指认不出来则**组内全部记 unknown**。
    注意：这里**不做"程序挑一个赢家"**。让其中一条拿走那个框，等于把 1/N 的猜测
    写进训练监督（规格 §5.4：得到 mask/框不等于识别正确）。

    返回 ({mark_id: "ok"|"collision"}, {mark_id: 组描述}, [组]）。
    `groups` 里的每组是候选下标列表（组内顺序即发现顺序），供消歧阶段使用。
    用 mark_id 而不是对象 id 做键：对象 id 在对象被回收后可能被复用，
    对"这是哪条标志"这种长期有效的裁决不安全。
    """
    groups, used = [], set()
    for index, item in enumerate(candidates):
        if index in used:
            continue
        group = [index]
        for other in range(index + 1, len(candidates)):
            if other in used:
                continue
            if _iou(item["box"], candidates[other]["box"]) >= config.MARK_BOX_COLLISION_IOU:
                group.append(other)
                used.add(other)
        used.add(index)
        groups.append(group)
    assignments, collisions = {}, {}
    for group in groups:
        if len(group) < 2:
            assignments[candidates[group[0]]["mark"]["mark_id"]] = "ok"
            continue
        names = [str(candidates[i]["mark"].get("name") or "?") for i in group]
        label = "、".join(names[:4]) + ("…" if len(names) > 4 else "")
        for i in group:
            mark_id = candidates[i]["mark"]["mark_id"]
            assignments[mark_id] = "collision"
            collisions[mark_id] = label
        print(f"[mark_evidence] 撞框：{len(group)} 条标志定位到同一区域"
              f"（{label}），先做一次消歧", flush=True)
    return assignments, collisions, groups

def _crop_quality(cropped) -> tuple:
    """裁剪是否足以进入标志监督。

    判据按**原始像素**而不是归一化尺寸：角色图里的小结绳、纹样本来就只占几十像素，
    按比例否决会把真实的标志证据全部丢掉（实测：一张 1242×1656 的图里，
    腰间结绳的定位框是 96×55 像素，比例只有 0.077，但内容是清楚的）。
    真正要挡住的是"框其实没框住东西"：原始像素过低时放大只会把噪声当证据。
    """
    if min(cropped.size) < 8:
        return "useless", f"裁剪仅 {cropped.size[0]}×{cropped.size[1]} 像素，无法作为证据"
    if min(cropped.size) < config.MARK_MIN_SOURCE_PIXELS:
        return "useless", (f"原始裁剪仅 {cropped.size[0]}×{cropped.size[1]} 像素，"
                           f"低于 {config.MARK_MIN_SOURCE_PIXELS} 的硬下限，"
                           f"放大只会把噪声当证据，不进入标志监督")
    scale = config.MARK_MIN_CROP_SIDE / float(min(cropped.size))
    if scale > config.MARK_MAX_UPSCALE:
        return "useless", (f"需要放大 {scale:.1f} 倍才能达到工作尺寸，超过 "
                           f"{config.MARK_MAX_UPSCALE} 倍上限，不进入标志监督")
    return "ok", ""


def relative_crop_path(absolute) -> str:
    """把落盘的裁剪文件表示成**相对 PROJECT_ROOT** 的路径（全项目统一一种写法）。

    实测踩过：写端用 `os.path.relpath(target, config.OUTPUT_DIR)`（得到
    `marks/ayaka/xxx.png`），读端却拼 `config.PROJECT_ROOT`，于是解析到
    `<项目>/marks/ayaka/xxx.png`——把 output 这一层整个丢掉了，
    结果所有 54 条裁剪都"文件不存在"，留出对照直接退化为"证据不足"。
    统一成相对 PROJECT_ROOT 之后，读端只要拼一次就不会错。
    """
    return os.path.relpath(absolute, config.PROJECT_ROOT).replace("\\", "/")


def _observation(rel_path, mark, visibility="unknown", box=None, visible_description="",
                 crop_path=None, pixels=None, quality="", note="", source="vlm",
                 model="", location_audit=""):
    """一条"标志 × 图像"的证据记录（规格 §4 的逐图状态）。"""
    completeness = {"visible": "complete", "partial": "partial"}.get(visibility, "unknown")
    return {
        "mark_id": mark["mark_id"],
        "mark_name": mark.get("name") or "",
        "entity_mode": mark.get("entity_mode") or "locatable",
        "rel_path": rel_path,
        "visibility": visibility,
        "completeness": completeness,
        "box": list(box) if box else None,
        "pixels": list(pixels) if pixels else None,
        "crop_path": crop_path,
        "visible_description": visible_description,
        "quality": quality,
        "location_audit": location_audit,
        "source": source,
        "model": model or os.path.basename(config.VLM_PATH),
        "prompt_version": config.MARK_PROMPT_VERSION,
        "note": note,
    }


def training_crops(folder, max_per_mark=None, observations=None, library=None) -> list:
    """把真实标志裁剪整理成**训练可用**的条目（规格 §6：按标志均衡抽样）。

    返回 [{"mark_id", "rel_path", "box", "crop_path", "kind", "quality", ...}]。

    规则（每条都有实测理由）：
      * 只有 visible / partial 且拿到**真实框**的图才产出条目；未知或框不可用的不产出，
        也**不回退到中心裁剪**（规格 §2 第 6 行）；
      * 每个标志最多保留 max_per_mark 条，按"不同原图"优先——同一标志的多个视角比
        同一张图的重复裁剪更有价值，也避免大面积服装淹没小徽记；
      * 小裁剪（低于 MARK_MIN_CROP_SIDE）不进入监督：放大碎片只会给模型看模糊像素。
    """
    max_per_mark = config.MARK_MAX_CROPS_PER_MARK if max_per_mark is None else max_per_mark
    library = library or marks_module.load_library(folder)
    if not library:
        return []
    observations = observations if observations is not None \
        else marks_module.load_observations(folder)
    by_mark = {}
    for row in observations:
        if row.get("visibility") not in ("visible", "partial"):
            continue
        if not row.get("box") or not row.get("crop_path"):
            continue
        if str(row.get("quality") or "") != "ok":
            continue
        by_mark.setdefault(row.get("mark_id"), []).append(row)
    entries = []
    for mark in library.get("marks") or []:
        if str(mark.get("entity_mode") or "locatable") != "locatable":
            continue
        rows = by_mark.get(mark["mark_id"]) or []
        # 先按"不同原图"去重，再按可见描述的信息量排序（有描述的更可能带结构信息）
        seen_images, ordered = set(), []
        for row in sorted(rows, key=lambda r: (r["rel_path"] not in seen_images,
                                               len(str(r.get("visible_description") or "")) == 0)):
            if row["rel_path"] in seen_images:
                continue
            seen_images.add(row["rel_path"])
            ordered.append(row)
        for row in ordered[:max_per_mark]:
            entries.append({
                "mark_id": mark["mark_id"], "mark_name": mark.get("name") or "",
                "mark_class": mark.get("mark_class") or "detail",
                "region": mark.get("region") or "whole",
                "regions": list(mark.get("regions") or [mark.get("region") or "whole"]),
                "rel_path": row["rel_path"], "box": list(row["box"]),
                "crop_path": row["crop_path"], "visibility": row["visibility"],
                "kind": "mark_crop", "quality": row.get("quality") or "ok",
            })
    return entries


def reference_crops(folder, library=None) -> list:
    """单独标志参考图的真实裁剪（每条参考图与描述是一对确定的配对）。"""
    library = library or marks_module.load_library(folder)
    if not library:
        return []
    entries = []
    for mark in library.get("marks") or []:
        for evidence in mark.get("evidence") or []:
            if evidence.get("kind") != "reference_image" or not evidence.get("crop_path"):
                continue
            entries.append({
                "mark_id": mark["mark_id"], "mark_name": mark.get("name") or "",
                "mark_class": mark.get("mark_class") or "detail",
                "region": "whole", "regions": ["whole"],
                "rel_path": evidence.get("rel_path"), "box": None,
                "crop_path": evidence["crop_path"], "visibility": "visible",
                "kind": "reference_image", "quality": "reference",
            })
    return entries


def benchmark(folder, marks=None, limit=None, client=None, library=None,
              observations=None) -> dict:
    """标志识别对照：把逐图证据汇总成规格 §8 要求的可比较指标。

    指标（规格 §8 明列）：
      * 每条标志的**召回率**：在它确实出现的图上被判为可见的比例；
      * **可见状态准确性**：visible/partial 与 unknown 的分工是否站得住；
      * **定位质量**：可用框的比例、框的尺寸分布（过宽的框说明模型只是在"猜大致位置"）；
      * **错误合并率**：库里的标志数 vs 参考图条目数（合并过多会把两个装饰当成一个）；
      * 时间与峰值显存（由调用方计时，不在这里测）。

    "确实出现"的判据只能来自人工核验或独立参考，这里不假装有 ground truth：
    没有人工标注时，本函数明确报告**无监督统计**，不给准确率数字（规格 §8：
    不能只让同一个 VLM 对自己提取的内容打分）。
    """
    library = library or marks_module.load_library(folder) or parse_marks_safe(folder)
    observations = observations if observations is not None \
        else marks_module.load_observations(folder)
    rows = [row for row in observations
            if not marks or row.get("mark_id") in set(marks)]
    per_mark = {}
    for mark in library.get("marks") or []:
        if marks and mark.get("mark_id") not in set(marks):
            continue
        if str(mark.get("entity_mode") or "locatable") != "locatable":
            continue
        own = [row for row in rows if row.get("mark_id") == mark.get("mark_id")]
        if limit:
            own = own[:limit]
        counts = {}
        for row in own:
            counts[row.get("visibility")] = counts.get(row.get("visibility"), 0) + 1
        visible = counts.get("visible", 0)
        partial = counts.get("partial", 0)
        usable = [row for row in own if row.get("quality") == "ok" and row.get("crop_path")]
        sizes = []
        for row in usable:
            pixels = row.get("pixels")
            if pixels:
                sizes.append((pixels[2] - pixels[0], pixels[3] - pixels[1]))
        per_mark[mark["mark_id"]] = {
            "name": mark.get("name"),
            "region": mark.get("region"),
            "mark_class": mark.get("mark_class"),
            "images_checked": len(own),
            "visible": visible,
            "partial": partial,
            "unknown": counts.get("unknown", 0),
            "absent": counts.get("absent", 0),
            "out_of_view": counts.get("out_of_view", 0),
            "occluded": counts.get("occluded", 0),
            "usable_crops": len(usable),
            "crop_median_size": (sorted(sizes)[len(sizes) // 2] if sizes else None),
            "box_too_wide": sum(1 for size in sizes if size[0] * size[1] > 0.25),
        }
    merged = sum(1 for link in library.get("links") or [] if link.get("merged"))
    total_refs = len(library.get("references") or [])
    return {
        "character": folder,
        "marks": len(library.get("marks") or []),
        "observations": len(observations),
        "reference_entries": total_refs,
        "merged_reference_entries": merged,
        "wrong_merge_rate": (round(merged / float(total_refs), 4) if total_refs else None),
        "per_mark": per_mark,
        "ground_truth": False,
        "note": ("没有人工核验集时这些是无监督统计，**不是准确率**；"
                 "规格 §8 要求用一次性人工核验集算召回率与误认率，"
                 "并配合独立规则/视觉核对与人工盲评（不能只让同一个 VLM 自评）"),
    }


def parse_marks_safe(folder):
    """解析标志库但**不写盘**（供只读的基准统计使用）。"""
    try:
        return marks_module.parse_marks(folder)
    except Exception as exc:
        print(f"[mark_evidence] 标志库解析失败: {type(exc).__name__}: {exc}")
        return {"marks": [], "warnings": [], "references": [], "links": []}


def benchmark_text(report) -> str:
    """把基准报告渲染成人类可读表格。"""
    lines = [f"标志识别对照: {report['character']}",
             f"  标志 {report['marks']} 个，逐图证据 {report['observations']} 条，"
             f"参考条目 {report['reference_entries']} 条（并入已有标志 "
             f"{report['merged_reference_entries']} 条）"]
    if report.get("wrong_merge_rate") is not None:
        lines.append(f"  错误合并率（参考条目被并入的比例）: "
                     f"{report['wrong_merge_rate']:.0%}")
    lines.append("")
    lines.append(f"  {'标志':<22}{'可见':>5}{'部分':>5}{'未知':>5}{'无':>5}"
                 f"{'可用裁剪':>9}  中位裁剪尺寸")
    for mark_id, row in sorted(report["per_mark"].items()):
        size = row["crop_median_size"]
        lines.append(f"  {str(row['name'])[:20]:<22}{row['visible']:>5}{row['partial']:>5}"
                     f"{row['unknown']:>5}{row['absent']:>5}{row['usable_crops']:>9}  "
                     f"{f'{size[0]}×{size[1]}' if size else '—'}")
    lines.append("")
    lines.append(f"  {report['note']}")
    return "\n".join(lines)


def reidentification_probe(folder, client, library=None, observations=None,
                           samples_per_mark=2, library_path=None) -> dict:
    """**留出重识别对照**：用真实裁剪去问模型"这个标志在不在这张图里"。

    这是"Qwen3-VL-4B 能不能胜任标志识别"这个问题唯一诚实的本地判据。
    做法（规格 §8：识别对照至少覆盖小徽记、重复纹样、侧背视角、遮挡及易混装饰）：
      1. 取某个标志的一条**真实裁剪**（来自原图，是它自己的证据）；
      2. 用**另一张**已经记录过该标志可见状态的整图提问（留出，不用它自己那张）；
      3. 比较模型答案与已记录的逐图状态。
    如果模型连"这个标志在不在这张图里"都答不对，那么"自动发现标志"这条链路
    在它身上就不成立，需要换模型，而不是继续调提示词。

    返回 {"cases": [...], "agreement": float|None, "per_mark": {...}}。
    `agreement` 是**有监督对照**（前提是那份逐图状态本身可信），因此同时报告样本数。
    """
    library = library or marks_module.load_library(folder) or parse_marks_safe(folder)
    observations = observations if observations is not None \
        else marks_module.load_observations(folder)
    if not library or client is None or not getattr(client, "available", False):
        return {"cases": [], "agreement": None, "note": "缺少标志库或 VLM 不可用"}
    import image_utils
    by_mark = {}
    for row in observations:
        by_mark.setdefault(row.get("mark_id"), []).append(row)
    cases = []
    for mark in library.get("marks") or []:
        if str(mark.get("entity_mode") or "locatable") != "locatable":
            continue
        rows = by_mark.get(mark["mark_id"]) or []
        with_crop = [row for row in rows if row.get("quality") == "ok" and row.get("crop_path")]
        others = [row for row in rows if row.get("visibility") in
                  ("visible", "partial", "unknown", "absent", "out_of_view", "occluded")]
        if not with_crop or not others:
            continue
        reference = with_crop[0]
        ref_path = resolve_output_path(reference["crop_path"])
        if ref_path is None:
            continue
        ref_image = image_utils.load_image(ref_path)
        for row in others[:samples_per_mark]:
            if row["rel_path"] == reference["rel_path"]:
                continue
            image_path = os.path.join(config.DATASET_DIR, folder, "images", row["rel_path"])
            if not os.path.isfile(image_path):
                continue
            image = image_utils.load_image(image_path)
            # 用"裁剪图 + 整图"一起问：这正是模型在自动发现时的处境
            probed = _probe_one(client, ref_image, image, mark)
            if not probed:
                continue
            expected = row.get("visibility")
            expected_positive = expected in ("visible", "partial")
            actual_positive = probed.get("visibility") in ("visible", "partial")
            cases.append({
                "mark_id": mark["mark_id"], "name": mark.get("name"),
                "reference_image": reference["rel_path"],
                "probe_image": row["rel_path"],
                "expected": expected, "actual": probed.get("visibility"),
                "expected_positive": expected_positive, "actual_positive": actual_positive,
                "agreed": expected_positive == actual_positive,
                "description": probed.get("description") or "",
            })
    if not cases:
        return {"cases": [], "agreement": None,
                "note": "证据不足，无法做留出对照（需要至少两条逐图证据）"}
    # 参照状态里 `unknown` 的含义是"我们没定位到"，**不是"这里没有"**。
    # 把它算进一致率会把"模型找到了而我们因为撞框没敢采信"记成模型出错
    # （实测：35 个样本里 27 个的参照就是 unknown，一致率被压到 34%，
    # 但其中模型答"可见"的多数属于**模型比我们更敢下判断**）。
    # 因此把"可核对的样本"与"无法核对的样本"分开报告。
    verifiable = [case for case in cases if case["expected"] != "unknown"]
    unverifiable = [case for case in cases if case["expected"] == "unknown"]
    agreed = sum(1 for case in verifiable if case["agreed"])
    positive = [case for case in verifiable if case["expected_positive"]]
    positive_agreed = sum(1 for case in positive if case["agreed"])
    all_agreed = sum(1 for case in cases if case["agreed"])
    return {
        "cases": cases,
        "agreement": (round(agreed / float(len(verifiable)), 4) if verifiable else None),
        "verifiable_samples": len(verifiable),
        "unverifiable_samples": len(unverifiable),
        "agreement_on_positive_samples": (round(positive_agreed / float(len(positive)), 4)
                                          if positive else None),
        "positive_samples": len(positive),
        "agreement_including_unknown": round(all_agreed / float(len(cases)), 4),
        "note": ("一致率只在**可核对的样本**（参照状态不是 unknown）上计算；"
                 "参照为 unknown 的样本单列——它表示我们没定位到，而不是那里没有。"
                 "参照状态本身也需要人工核验，因此这是**相对一致性**，不是绝对准确率（规格 §8）"),
    }


def _probe_one(client, ref_image, image, mark):
    """一次留出提问：给参考裁剪与整图，问这个标志在不在这张整图里。"""
    prompt = (
        "Image 1 is a reference crop of ONE character mark. Image 2 is a full picture. "
        "Decide whether that same mark is actually visible in image 2.\n"
        f"The mark is: {mark.get('description_en') or mark.get('description')}\n"
        "Return ONLY JSON:\n"
        '{"visibility":"visible|partial|occluded|out_of_view|absent|unknown",'
        '"visible_description":"what is visible of it in image 2, in ENGLISH, under 20 words"}'
    )
    try:
        from vlm.vlm_utils import parse_json
        raw = client.generate_with_images(
            [ref_image, image], prompt, max_new_tokens=160, temperature=0)
        result = parse_json(raw)
    except Exception as exc:
        print(f"[mark_evidence] 留出对照失败: {type(exc).__name__}: {exc}")
        return {}
    if not result:
        return {}
    visibility = str(result.get("visibility") or "unknown").strip().lower()
    if visibility not in ("visible", "partial", "occluded", "out_of_view", "absent", "unknown"):
        visibility = "unknown"
    return {"visibility": visibility,
            "description": str(result.get("visible_description") or "").strip()}


def summarize(result) -> str:
    stats = result.get("stats") or {}
    lines = [f"[mark_evidence] 标志 {stats.get('marks', 0)} 个，"
             f"逐图证据 {stats.get('observations', 0)} 条，"
             f"定位成功 {stats.get('located', 0)}，未定位 {stats.get('unlocated', 0)}，"
             f"有真实裁剪 {stats.get('with_crop', 0)} 条，"
             f"单独标志参考 {stats.get('reference_crops', 0)} 张"]
    for reason in result.get("reasons") or []:
        lines.append(f"  ! {reason}")
    for gap in (result.get("library") or {}).get("unresolved_gaps") or []:
        lines.append(f"  ! 证据缺口: {gap}")
    return "\n".join(lines)


def main():
    """只读入口：统计当前角色已有的标志证据，不加载模型。

    python -m vision.mark_evidence [角色]
    """
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else config.CHARACTER_ID
    result = benchmark(folder)
    print(benchmark_text(result))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
