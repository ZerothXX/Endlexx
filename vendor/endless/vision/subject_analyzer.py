# -*- coding: utf-8 -*-
"""
vision/subject_analyzer.py —— 输入主体理解（规格 §7）

职责：
- SUBJECT_PROMPT 不在此重复定义：vlm/vlm_utils.py 中已有唯一版本（§7 JSON 结构），
  本文件直接引用，保证训练/推理侧提示词永远一致（单一定义原则）。
- analyze_subject(client, image) -> dict：调用 VLM 分析输入图片的主体并补全字段。
  任何失败（client 为 None / analyze_image 抛异常 / 返回空 dict / subject_type 非法）
  都返回 {"subject_type": "unknown"} 兜底（规格 §27 失败回退 / §32 unknown 不崩溃）。
- get_style_token(style) -> str：规格 §17“原画风保持”的 style -> prompt 片段映射，
  供下游 prompt 构建使用；无匹配返回空串（不注入任何风格约束）。

接口约定（与 vlm/vlm_utils.py 的 VLMClient 对齐）：
    client.analyze_image(image, system_prompt, max_new_tokens=None, temperature=None) -> dict
"""

import os
import sys
from copy import deepcopy

# 保证从任意位置直接运行本文件时都能 import 到项目根目录的模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from vlm.vlm_utils import SUBJECT_PROMPT  # noqa: E402  引用唯一提示词定义（§7）
import config  # noqa: E402
from character.transfer_constraints import region_visible  # noqa: E402

# =========================
# §7 默认主体语义（字段与 SUBJECT_PROMPT 完全对齐）
# =========================
# analyze_subject 返回前用本结构补全：VLM 未给出的键填 None / []，
# 保证下游（迁移策略 / prompt 构建）读取任何键都不会 KeyError。
DEFAULT_SUBJECT = {
    "subject_type": "unknown",     # human | animal | object | unknown（§7 四选一）
    "pose": None,                  # 姿态，如 standing / sitting
    "body_view": "unknown",
    "head_view": "unknown",
    "proportions": "unknown",
    "framing": "unknown",
    "hands": "unknown",
    "eyes": "unknown",
    "expression": "unknown",
    "coverage": {},
    "species": None,               # 动物种类，如 cat（仅 animal 时有意义）
    "object_category": None,       # 物品类别，如 cup / bag（仅 object 时有意义）
    "material": None,              # 材质，如 fur / ceramic / fabric
    "style": None,                 # 画风，如 pixel art / realistic photo / 3D render / anime illustration
    "structure": [],               # 主体主要结构部件列表
    "visible_regions": [],         # 可见区域列表
    "attachment_regions": [],      # 可佩戴/悬挂配饰的区域（§7 示例：head / ear / neck / bag surface）
    "surface_regions": [],         # 适合表面纹样的平坦区域（§7 示例：cup body / handle）
}

# 合法的 subject_type 取值（§7）
VALID_SUBJECT_TYPES = ("human", "animal", "object", "unknown")

# 睁闭眼复核提问（见 confirm_closed_eyes）
EYE_VERIFY_PROMPT = (
    'Look only at the eyes of the single main person in this image. '
    'Return JSON {"eyes": "open"} or {"eyes": "closed"}. '
    'Treat squinting or partially open eyes as "open".')


def confirm_closed_eyes(client, image, result):
    """用一次独立复问核对"闭眼"判断；只有复核一致才保留该结论。

    实测问题：主体分析把《蒙娜丽莎》读成 "closed eyes"。该结论一旦进入正向/负向
    提示词，就会把原本睁眼的人物强行画成闭眼（100% 可复现的明显错误）。这里对
    "闭眼"这一条做二次确认：复问回答 open 时改写为 "open eyes"，并记录审计字段。
    复核本身失败时不改结论（保持原行为），不引入新的静默降级。
    """
    if client is None or result.get("eyes") != "closed eyes":
        return result
    try:
        raw = client.analyze_image(image, EYE_VERIFY_PROMPT, max_new_tokens=32, temperature=0)
    except Exception as exc:  # 复核失败不改变原结论
        result["eyes_verification"] = f"verification_failed: {type(exc).__name__}"
        return result
    answer = str((raw or {}).get("eyes", "")).strip().lower() if isinstance(raw, dict) else ""
    if answer.startswith("open"):
        result["eyes"] = "open eyes"
        result["eyes_verification"] = "second_query_reported_open"
    elif answer.startswith("closed"):
        result["eyes_verification"] = "second_query_agreed_closed"
    else:
        result["eyes_verification"] = f"inconclusive:{answer or 'empty'}"
    return result


def analyze_subject(client, image) -> dict:
    """分析输入图片的主体，返回补全字段后的 dict（绝不抛异常）。

    参数：
        client: VLMClient 实例或 None（None 表示 VLM 不可用，直接走 unknown 兜底）
        image : PIL.Image（RGB）或图片路径字符串

    返回：
        补全后的 §7 结构 dict；任何失败路径返回 DEFAULT_SUBJECT 副本
        （subject_type="unknown"，规格 §27/§32）。
    """
    # client 不可用（VLM 关闭 / 加载失败）：直接兜底，不打印噪音
    if client is None:
        return deepcopy(DEFAULT_SUBJECT)

    try:
        raw = client.analyze_image(image, SUBJECT_PROMPT, temperature=0)
    except Exception as e:
        # analyze_image 本身承诺不抛异常，但防御一层，绝不崩溃（§32）
        print(f"[subject_analyzer] 告警: 主体分析调用异常，回退 unknown。原因: {type(e).__name__}: {e}")
        return deepcopy(DEFAULT_SUBJECT)

    # analyze_image 返回 {}（推理失败 / 输出无法解析为 JSON）→ unknown 兜底
    if not isinstance(raw, dict) or not raw:
        return deepcopy(DEFAULT_SUBJECT)

    return confirm_closed_eyes(client, image, normalize_subject(raw))


def normalize_subject(raw):
    """Validate VLM/cache data without sharing mutable defaults or inventing anatomy."""
    if not isinstance(raw, dict):
        return deepcopy(DEFAULT_SUBJECT)

    # 只保留已知字段（VLM 偶发的多余键不进入结果），缺失键由 DEFAULT_SUBJECT 补全
    result = deepcopy(DEFAULT_SUBJECT)
    for key in DEFAULT_SUBJECT:
        if key in raw and raw[key] is not None:
            result[key] = raw[key]

    # subject_type 合法性校验：不在四选一内一律视为 unknown（§7/§32）
    for key, default in DEFAULT_SUBJECT.items():
        if key not in {"coverage", "visible_regions", "structure", "attachment_regions", "surface_regions"}:
            value = result[key]
            result[key] = value.strip().lower() if isinstance(value, str) else default
    if result["subject_type"] not in VALID_SUBJECT_TYPES:
        result["subject_type"] = "unknown"
    for key in ("body_view", "head_view"):
        if result[key] not in {"front view", "three-quarter view", "side view", "back view"}:
            result[key] = "unknown"
    if result["proportions"] not in {"chibi", "natural", "stylized"}:
        result["proportions"] = "unknown"
    if result["eyes"] not in {"open eyes", "closed eyes", "one eye closed"}:
        result["eyes"] = "unknown"
    if result["expression"] not in {"smiling", "open-mouth smile", "neutral expression", "frowning"}:
        result["expression"] = "unknown"
    for key in ("visible_regions", "structure", "attachment_regions", "surface_regions"):
        value = result[key]
        result[key] = list(dict.fromkeys(v.strip().lower() for v in value
                           if isinstance(v, str) and v.strip()
                           and v.strip().lower() not in {"unknown", "none", "n/a"})) if isinstance(value, list) else []
    coverage = result.get("coverage")
    result["coverage"] = ({k: v for k, v in coverage.items()
                           if k in {"legs", "chest"} and isinstance(v, str) and v in
                           {"bare", "covered", "out_of_frame", "exposed", "unknown"}}
                          if isinstance(coverage, dict) else {})
    if result["subject_type"] != "human":
        for key in ("body_view", "head_view", "proportions"):
            result[key] = "unknown"
        result["coverage"] = {}
        result["hands"] = "unknown"
    elif result["framing"] in {"upper body", "bust", "portrait", "close-up"}:
        # A cropped portrait cannot provide evidence of exposed legs.
        result["coverage"]["legs"] = "out_of_frame"
        excluded = {"legs", "leg", "feet", "foot", "thighs", "knees", "lower legs", "lower body", "full body"}
        if result["hands"] == "hidden":
            excluded |= {"hand", "hands"}
        for key in ("visible_regions", "structure", "attachment_regions"):
            result[key] = [v for v in result[key] if v not in excluded]
    return result


def locate_regions(client, image, subject):
    """定位需要"更强角色化"的源图区域，返回 {区域名: {"box":..., "target":...}}。

    只负责"大致在哪"：用视觉模型给出头发/头与躯干服装的边界框。
    像素级掩版由 region_masks() 用 SAM 点提示细化——视觉模型的框经常过宽
    （实测头发框会给到半张画幅），直接拿框当掩版会把背景一起重绘。
    区域不可见或定位失败时**不猜框**，直接不返回该区域。
    """
    if client is None or not getattr(client, "available", False):
        return {}
    subject_type = subject.get("subject_type", "unknown")
    targets = {}
    # 区域划分按主体类型给：人物有"头发"这一独立区域，动物/物品没有，
    # 对它们问"头发"只会得到错框（实测）；它们需要的是"身体/表面"这一块。
    if subject_type == "human":
        if region_visible(subject, "head") or not subject.get("visible_regions"):
            targets["head"] = ("the person's hair and head", "hair")
        if region_visible(subject, "torso") or region_visible(subject, "arms"):
            targets["body"] = ("the person's clothing on the torso", "clothing")
    else:
        targets["body"] = ("the {0}'s body or surface".format(subject_type or "subject"),
                           "{0} surface".format(subject_type or "subject"))
        # 非人物额外要"头部"：附着区域必须挂在头上、而不是按全身外接框比例去猜。
        # 实测：用全身比例定位会依次落到耳间、眼睛上、背上（头小身大的卧姿最明显）。
        targets["head"] = ("the {0}'s head".format(subject_type or "subject"), "head")
        if subject_type == "animal":
            # 动物的"加法"要把配饰放在**颈部**。实测：靠"头部外接框下沿 + 固定高度"
            # 这类几何启发式定位颈部并不可靠——卧姿猫身上那条带子会落到额头/眼睛上
            # （出图出现一条横贯面部的色带）。视觉模型对"脖子在哪"有稳定的语义认知，
            # 因此直接问它一个颈部区域，再交给 SAM 细化，不再用几何推导。
            targets["neck"] = ("the {0}'s neck area (where a collar would be worn)"
                               .format(subject_type), "neck")
    found = {}
    for name, (prompt, label) in targets.items():
        box = client.locate(image, prompt)
        located = box is not None
        if box is None and name == "head":
            # 头部定位失败时退回"整幅上半部"，这是构图上的保守假设，不涉及身份臆测
            box = (0.0, 0.0, 1.0, 0.45)
            label = "hair (fallback upper frame)"
        if box is not None:
            # `located` 记录"这个框是模型真的定位到的，还是退化假设"。
            # 下游据此决定能不能把颜色锚定作用在它上面：退化框（整幅上半部）
            # 会把线稿人物整个上半身都框进去，锚定之后整幅人物被刷成角色色
            # （实测：线稿样例的裙子、脸颊、手臂全被涂蓝）。
            found[name] = {"box": box, "target": label, "located": located}
            print(f"[subject_analyzer] 局部区域 {name}: {label} -> "
                  f"{tuple(round(v, 3) for v in box)}"
                  f"{'' if located else '（退化假设，不用于配色锚定）'}")
    return found


def _tune_mask(mask, close=None, dilate=None):
    """掩版形态学修补：闭运算填洞 -> 外扩 -> 只保留最大连通块。

    为什么需要：SAM 给出的边界在贴边发丝/衣褶处会漏掉一小块，局部重绘后那块仍是
    源图像素，看起来就是"重绘区域里有脏斑块"（实测症状）。补一层形态学比再问一次
    模型便宜，也不会引入新的不确定判断。
    """
    import numpy as np
    import cv2
    close = int(config.REGION_MASK_CLOSE if close is None else close)
    dilate = int(config.REGION_MASK_DILATE if dilate is None else dilate)
    array = (np.asarray(mask.convert("L")) > 128).astype(np.uint8)
    if close > 1:
        array = cv2.morphologyEx(array, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    if dilate > 1:
        array = cv2.dilate(array, np.ones((dilate, dilate), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        array = (labels == largest).astype(np.uint8)
    from PIL import Image as _Image
    return _Image.fromarray(array * 255, mode="L")


def region_masks(image, sam, regions):
    """用 SAM 点提示把"大致框"细化成像素级掩版，返回 {区域名: (mask|None, audit)}。

    头发：提示点取框内上部（发区），负例点取框底两角（通常是肩颈或背景）
          ——实测能稳定圈住整片头发，同时不把脸一起框进去。
    服装：提示点取"发区以下、框内中下部"的衣身位置，负例点取框外上方（头发）
          与框外下方（腿/背景）——只用一个框中心点会把脸和头发一起圈进来（实测）。
    细化失败或覆盖比例不合理时返回 None，由调用方退回矩形掩版或跳过——不静默换实现。
    """
    if sam is None or not regions:
        return {}
    from vision.segmentation import sam_point_maskbox
    width, height = image.size
    results = {}
    head_box = (regions.get("head") or {}).get("box")
    for name, info in regions.items():
        box = info.get("box")
        if not box:
            continue
        left, top, right, bottom = box
        px = (left * width, top * height, right * width, bottom * height)
        span_x, span_y = px[2] - px[0], px[3] - px[1]
        if name == "head":
            points = [(0.5 * (px[0] + px[2]), px[1] + 0.25 * span_y)]
            labels = [1]
            points += [(px[0] + 0.10 * span_x, px[1] + 0.92 * span_y),
                       (px[0] + 0.90 * span_x, px[1] + 0.92 * span_y)]
            labels += [0, 0]
        else:
            # 有头部框时，把"发区底边"当作服装区的上界，避免把脸和头发当衣服
            upper = top
            if head_box:
                upper = min(0.98, max(top, head_box[1] + 0.75 * (head_box[3] - head_box[1])))
            upper_px = upper * height
            if upper_px >= px[3] - 4:
                results[name] = (None, {"region": name, "refined": False,
                                        "reason": "头部区域占满该框，没有可用的衣身位置"})
                continue
            centre = (0.5 * (px[0] + px[2]), upper_px + 0.45 * (px[3] - upper_px))
            points = [centre]
            labels = [1]
            points += [(centre[0], max(0.0, upper_px - 0.05 * height)),        # 上方＝头发
                       (0.5 * (px[0] + px[2]), min(height - 1.0, px[3] + 0.08 * height))]
            labels += [0, 0]
        mask = sam_point_maskbox(image, sam, points, labels=labels, box=px)
        if mask is None:
            results[name] = (None, {"region": name, "refined": False,
                                    "reason": "SAM 未能给出该区域的掩版"})
            continue
        mask = _tune_mask(mask)
        covered = sum(mask.histogram()[128:]) / float(width * height)
        accepted = 0.01 < covered < 0.60
        results[name] = (mask if accepted else None,
                         {"region": name, "refined": accepted,
                          "coverage": round(covered, 4), "tuned": True,
                          "reason": "" if accepted else "SAM 掩版覆盖比例不合理，已丢弃"})
        print(f"[subject_analyzer] SAM 细化 {name}: 覆盖 {covered:.1%} "
              f"{'采用' if accepted else '丢弃（比例不合理）'}")
    # 服装掩版必须排除头发：SAM 的"人"提示会把长发一起圈进来（实测）。
    # 这里用已确定的头发掩版做一次确定性剔除，而不是再问一次模型。
    head = results.get("head", (None, None))[0]
    body_mask, body_audit = results.get("body", (None, None))
    if head is not None and body_mask is not None:
        body_mask = _subtract(body_mask, head)
        body_audit = dict(body_audit or {})
        body_audit["coverage"] = round(
            sum(body_mask.histogram()[128:]) / float(width * height), 4)
        body_audit["minus_hair"] = True
        results["body"] = (body_mask, body_audit)
        print(f"[subject_analyzer] SAM 细化 body: 已剔除头发，覆盖 "
              f"{body_audit['coverage']:.1%}")
    return results


def _subtract(mask, remove):
    """mask 减去 remove，并只保留最大连通块（去掉零散碎点）。"""
    import numpy as np
    import cv2
    array = (np.asarray(mask.convert("L")) > 128).astype(np.uint8)
    removal = (np.asarray(remove.convert("L").resize(mask.size)) > 128).astype(np.uint8)
    if removal.any():
        removal = cv2.dilate(removal, np.ones((5, 5), np.uint8), iterations=1)
    array = np.where(removal > 0, 0, array).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        array = (labels == largest).astype(np.uint8)
    from PIL import Image as _Image
    return _Image.fromarray(array * 255, mode="L")


def subject_silhouette(image, masks, sam=None, reference=None):
    """整块主体在**生成结果里真正被改写**的像素范围（人物合成的掩版）。

    为什么不能直接用"点名区域"（头发+衣服）的并集：实测人物样例里它只覆盖画面 34%，
    而人物本身约占 55%，脸、脖子、手臂、飘散的长发都不在其中；合成后这些地方留在源图里，
    于是成品出现"写实脸 + 动漫衣服"的媒介拼接，并沿掩版边界留下一圈硬折线
    （实测右肩那条台阶状描边来自视觉模型的矩形框）。

    做法：把"生成结果 vs 被改写的源图"的差异并进区域并集，再用区域邻域限制范围。
    被点名的区域本来就该被改写，而改写又只可能发生在主体上，因此
    "差异显著 ∩ 区域邻域"正好给出贴着**头发与衣服真实轮廓**的有机范围。

    处理顺序（每一步都对应一个已实测的缺陷）：
      1. 差异 ∩ 区域邻域（半径由 SUBJECT_SILHOUETTE_NEIGHBOUR_PAD 控制）；
      2. **补洞**：SAM 常把脸/眼睛排成洞，不补就会出现"生成结果里嵌着源图斑块"；
      3. 轻度闭运算 + 开运算：抹掉锯齿与零星碎点，但**不做"只留最大连通块"**
         ——那会把与主体断开的发梢/手臂整块丢掉，合成时又退回源图，形成浅色残带；
      4. 不外扩：外扩留给合成阶段（那里才知道羽化带要落在哪一侧）。

    reference 为空时退回"纯区域并集"路径，并在日志里写明来源，不猜阈值。
    sam 参数保留给"差异法失败"时的兜底调用方，当前实现不依赖它。
    """
    import numpy as _np
    import cv2
    from PIL import Image as _Image

    if image is None or not masks:
        return None
    width, height = image.size
    union = _np.zeros((height, width), dtype=_np.uint8)
    for mask in masks.values():
        if mask is None:
            continue
        layer = mask.convert("L")
        if layer.size != image.size:
            layer = layer.resize(image.size, _Image.NEAREST)
        union |= (_np.asarray(layer) > 128).astype(_np.uint8)
    if union.sum() < 64:
        return None
    source = "区域掩版并集"
    # 差异法曾用于"把没被点名但确实被改写的主体像素也纳入掩版"。实测否决：
    # 差异是逐像素判定，而区域掩版并集本身带孔（SAM 把发丝间的高光判为非主体），
    # 两者叠加后掩版整体呈**筛网状**，合成时筛孔处退回源图，出图出现
    # "闪粉/星点"一样的伪影撒在肩与头发上（三张人物样例都能看到）。
    # 现在只用区域并集：内部本来就实心，形状由主体轮廓决定；
    # 没被点名的部位（脸、脖子、手臂）由合成阶段的"向外扩张"覆盖，
    # 而"向外扩张 + 向外羽化"是确定性的形态学操作，不会引入星点。
    # `reference` 参数保留（调用方仍会传），但不再参与判定。

    # 补洞 + 只留最大连通块（两件事必须都做，顺序也有讲究）：
    #   * **补洞**：区域掩版并集本身带孔（SAM 会把发丝间的高光/背景判为非主体），
    #     差异又是"逐像素"的，于是掩版整体呈**筛网状**——合成时筛孔处退回源图，
    #     出图看起来就是"闪粉/星点"一样的东西撒在肩与头发上（实测）。
    #     从画幅四边泛洪，泛不到的背景 0 像素就是"被主体包住的洞"，全部补上。
    #   * **只留最大连通块**：差异法在细碎发丝处还会产生孤立的小块，
    #     丢掉它们、只保留主体那一整块，形状更干净；主体边缘的细发丝由合成阶段的
    #     "向外扩张 + 向外羽化"覆盖，不需要在这里逐个保留。
    flood = union.copy()
    flood_mask = _np.zeros((height + 2, width + 2), dtype=_np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(_np.uint8)
    filled = _np.clip(union + holes, 0, 1).astype(_np.uint8)
    # 平滑边界：SAM 的并集在发丝、毛发这类高频区域边缘是**锯齿+斑驳**的，
    # 直接用它在合成时会留下一圈"星点/闪粉"状伪影（实测三张人物样例都能看到）。
    # 这里先闭运算接上断口、再开运算去掉凸起，最后取最大连通块——
    # 只剩平滑的一段外轮廓，羽化之后边界落在背景上，不再有星点。
    kernel = _np.ones((9, 9), _np.uint8)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if count > 1:
        largest = 1 + int(_np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        filled = (labels == largest).astype(_np.uint8)
    if filled.sum() < 64:
        print("[subject_analyzer] 主体轮廓为空，放弃使用")
        return None
    # 再过一次补洞：取最大连通块之后可能又留下新的小孔
    flood2 = filled.copy()
    flood_mask2 = _np.zeros((height + 2, width + 2), dtype=_np.uint8)
    cv2.floodFill(flood2, flood_mask2, (0, 0), 1)
    filled = _np.clip(filled + (flood2 == 0).astype(_np.uint8), 0, 1).astype(_np.uint8)
    coverage = float(filled.mean())
    if coverage < config.SUBJECT_MASK_MIN_COVERAGE or coverage > config.SUBJECT_MASK_MAX_COVERAGE:
        print(f"[subject_analyzer] 主体轮廓覆盖率 {coverage:.1%} 不合理，放弃使用")
        return None
    print(f"[subject_analyzer] 主体轮廓掩版: 覆盖率 {coverage:.1%}（来源: {source}）")
    return _Image.fromarray(filled * 255, mode="L")


def difference_region(reference, rendered, threshold=None):
    """生成结果相对参考图**确实被改写**的像素（二值 ndarray）。

    用途：给合成掩版一个贴着真实轮廓的形状。判据是逐像素颜色距离超过阈值；
    阈值来自 config（与"只在确有变化处采用生成结果"共用同一组配置项，
    避免两处各有一套数字）。
    """
    import numpy as _np
    if reference is None or rendered is None or reference.size != rendered.size:
        return None, 0.0
    limit = config.SUBJECT_SILHOUETTE_DIFF_DISTANCE if threshold is None else threshold
    base = _np.asarray(reference.convert("RGB")).astype(_np.int16)
    drawn = _np.asarray(rendered.convert("RGB")).astype(_np.int16)
    distance = _np.abs(base - drawn).sum(axis=2)
    changed = distance > float(limit)
    return changed, float(changed.mean())


def attachment_masks(subject_mask, head_box=None, head_mask=None, neck_box=None):
    """非人物"加法"用的附着区域：头顶配饰 + 颈圈 + 胸饰，全部锚定在**头部**上。

    为什么必须限定区域（实测结论）：把概念权重作用在整个主体上是**整体换肤**
    （猫被整只染成角色配色），那是"替换"不是"加法"。用户要的是
    "给猫加一条角色配色的项圈 + 头部标志配饰"，所以只在这几块地方生成。

    定位方式（四轮实测才定下来，记录以免重走）：
    * ❌ 用全身外接框的高度比例：头小身大的卧姿主体上会依次落到耳间 / 眼睛上 / 背上；
    * ❌ 用"最窄行=脖子"或"宽度达峰值=肩线"：卧姿正对镜头时没有可分辨的颈部收窄段；
    * ❌ 把区域做成 1~3% 宽的细横带：扩散模型只会填一块平色（实测淡色带）；
    * ✅ **锚定头部**（head_box/head_mask 由视觉模型框 + SAM 细化给出），
      在其上取紧凑团块（配饰形状）与其下取一圈颈圈，再从颈圈往下取胸饰。

    每个区域都只取主体掩版自身的像素，因此形状贴轮廓、不会把背景圈进来。
    """
    import numpy as _np
    from PIL import Image as _Image, ImageDraw as _ImageDraw

    if subject_mask is None:
        return {}
    mask = subject_mask.convert("L").point(lambda v: 255 if v >= 128 else 0)
    bbox = mask.getbbox()
    if not bbox:
        return {}
    coverage = _np.asarray(mask) > 128
    left, top, right, bottom = bbox
    body_w, body_h = right - left, bottom - top
    if body_w < 24 or body_h < 24:
        return {}

    head = _head_bounds(coverage, head_mask, head_box, bbox)
    if head is None:
        return {}
    h_left, h_top, h_right, h_bottom = head

    # 头部区域的**实际像素**（不是外接框）：头的外接框横跨两耳之间的空隙，
    # 直接用框会在空隙处生成东西（实测：装饰落在耳间空隙，与主体不重叠）。
    head_rows = coverage[h_top:h_bottom, :]
    cols = _np.flatnonzero(head_rows.any(axis=0))
    if cols.size == 0:
        return {}
    head_cov = _np.zeros_like(coverage)
    head_cov[h_top:h_bottom, cols[0]:cols[-1] + 1] = head_rows[:, cols[0]:cols[-1] + 1]
    h_cols = _np.flatnonzero(head_cov.any(axis=0))
    hc_left, hc_right = int(h_cols[0]), int(h_cols[-1]) + 1
    head_w = max(6, hc_right - hc_left)
    head_h = max(6, h_bottom - h_top)

    result = {}
    # 1) 头顶配饰（默认关闭，见 config 说明）：动物头顶只有两只耳朵、中间是背景，
    #    SAM 掩版在该处是两条细窄像素，任何团块都会被裁空或退化成平色条（实测）。
    if config.ATTACH_HEAD_ORNAMENT:
        orn_w = max(12, int(head_w * config.ATTACH_ORNAMENT_WIDTH))
        orn_h = max(10, int(head_h * config.ATTACH_ORNAMENT_HEIGHT))
        orn_cx = (hc_left + hc_right) // 2
        orn_cy = h_top + int(head_h * 0.42)
        om = _Image.new("L", mask.size, 0)
        _ImageDraw.Draw(om).ellipse((orn_cx - orn_w // 2, orn_cy - orn_h // 2,
                                     orn_cx + orn_w // 2, orn_cy + orn_h // 2), fill=255)
        orn = (_np.asarray(om) > 128) & head_cov
        if orn.any():
            result["ornament"] = _Image.fromarray((orn * 255).astype(_np.uint8), mode="L")

    # 2) 颈圈：位置优先用**视觉模型给的颈部框**（`neck_box`），它是语义定位；
    #    拿不到时才退回"从头下沿向下扫描收窄段"的几何启发式。
    #    实测教训：几何启发式在卧姿猫身上会把带子放到额头/眼睛上
    #    （出图出现一条横贯面部的色带，且那一条恰好是"猫脸最宽的一段"被当成了脖子）。
    collar_top, collar_bottom = _neck_span(coverage, head, bbox, head_w, neck_box)
    _keep(_row_span(coverage, collar_top, collar_bottom, config.ATTACH_SIDE_INSET),
          coverage, result, "collar")

    # 3) 胸饰：颈圈之下、更宽的一块（实测宽区域最容易生成出可辨认的衣物/饰物）
    chest_top = collar_bottom
    chest_bottom = min(bottom, chest_top + max(8, int(head_h * config.ATTACH_CHEST_BAND)))
    _keep(_row_span(coverage, chest_top, chest_bottom, config.ATTACH_SIDE_INSET),
          coverage, result, "chest")
    return result


def _neck_span(coverage, head, body_bbox, head_w, neck_box=None):
    """找出"颈部"所在的行区间（颈圈带的位置）。

    **首选语义定位**：`neck_box` 是视觉模型给出的颈部框（归一化坐标）——它回答的是
    "脖子在哪"这个语义问题。实测必要性：纯几何启发式（"头下沿往下找最窄行"）
    在卧姿的猫身上会把**猫脸最宽的一段**当成脖子，带子于是落到额头/眼睛上，
    出图出现一条横贯面部的色带。

    几何启发式只作为兜底：从头下沿往下扫，取行宽收窄的一段；都找不到时退化为固定高度。
    """
    import numpy as _np
    height = coverage.shape[0]
    left, top, right, bottom = body_bbox
    h_left, h_top, h_right, h_bottom = head
    head_h = max(6, h_bottom - h_top)

    # ---- 首选：视觉模型的颈部框（只取它给出的纵向区间；横向仍由主体轮廓决定）----
    if neck_box:
        y0 = max(h_top, int(float(neck_box[1]) * height))
        y1 = min(bottom, int(float(neck_box[3]) * height))
        # 视觉模型给出的框常常偏小（实测：纵向只占画面 25%，形成的带子只有 6.6% 画面宽，
        # 扩散模型在这么薄的一条里给不出可辨认的配饰形状，只会填一块浅色）。
        # 因此以框中心为基准上下各扩一档，让配饰有足够的面积成形。
        expand = float(getattr(config, "ATTACH_NECK_EXPAND", 1.6))
        if expand > 1.0:
            centre = 0.5 * (y0 + y1)
            half = max(4.0, 0.5 * (y1 - y0) * expand)
            y0 = max(h_top, int(centre - half))
            y1 = min(bottom, int(centre + half))
        if y1 - y0 >= max(4, int(head_h * 0.10)):
            return y0, y1
        if y0 > h_top:
            # 框太扁（只给了一条线）时，围绕它上下各留一点厚度
            pad = max(4, int(head_h * 0.12))
            return max(h_top, y0 - pad), min(bottom, y0 + pad)

    limit = min(bottom, h_bottom + max(8, int(head_h * (config.ATTACH_COLLAR_BAND
                                                        + config.ATTACH_CHEST_BAND))))
    rows = coverage[:, left:right].sum(axis=1).astype(float)
    best = None
    for y in range(h_bottom, limit):
        width = rows[y]
        if width < 4:
            continue
        if width <= head_w * 1.15:
            length = 1
            for z in range(y + 1, limit):
                if rows[z] < 4 or rows[z] > head_w * 1.15:
                    break
                length += 1
            if best is None or width < best[0]:
                best = (width, y, length)
    if best is not None:
        _, start, length = best
        band = max(6, min(int(head_h * config.ATTACH_COLLAR_BAND), max(6, length)))
        return start, min(bottom, start + band)
    band = max(6, int(head_h * config.ATTACH_COLLAR_BAND))
    return h_bottom, min(bottom, h_bottom + band)


def _head_bounds(coverage, head_mask, head_box, body_bbox):
    """求头部的像素范围。优先用**主体自身的轮廓**推断，视觉模型/SAM 的头部定位只作参考。

    实测教训：视觉模型给的"头部"框经常过长（有时拉到画幅底部），
    SAM 按这个框细化出来的掩版会覆盖整只动物（实测覆盖率 39%、外接框≈主体）——
    两者都不能直接信。可靠信号在主体轮廓里：

        * 主体顶部 1/8 的**中位行宽** ≈ 头部宽度（动物的头是顶部最窄的一段）；
        * 从顶部往下找**第一行宽度明显超过头宽**（× 该比例）的位置 —— 那就是头/身分界。

    分界若找不到（例如杯、柱状物体这种没有头身之分的），退化为"主体高度的一个比例"。
    这样不依赖任何单个模型输出的准确性，换角色、换主体类型都成立。
    """
    import numpy as _np
    left, top, right, bottom = body_bbox
    body_h = max(1, bottom - top)
    body_w = max(1, right - left)
    max_head_h = max(12, int(body_h * config.ATTACH_HEAD_MAX_FRACTION))

    row_width = coverage[:, left:right].sum(axis=1).astype(float)
    probe = row_width[top:top + max(2, int(body_h * 0.125))]
    head_w = float(_np.median(probe)) if probe.size else 0.0
    if head_w < 3:
        head_w = float(_np.percentile(row_width[top:bottom], 25)) or 3.0

    head_bottom = None
    threshold = head_w * config.ATTACH_HEAD_JUMP_RATIO
    for y in range(top, bottom):
        if row_width[y] > threshold:
            head_bottom = y
            break
    if head_bottom is None or head_bottom - top < 6:
        head_bottom = top + max(12, int(body_h * config.ATTACH_HEAD_FALLBACK))
    head_bottom = min(bottom, top + min(head_bottom - top, max_head_h))
    if head_bottom - top < 6:
        return None

    # 头部横向范围：取头部区段内实际出现过主体的列
    head_rows = coverage[top:head_bottom, :]
    cols = _np.flatnonzero(head_rows.any(axis=0))
    if cols.size == 0:
        return None
    return (int(cols[0]), int(top), int(cols[-1]) + 1, int(head_bottom))


def _row_span(coverage, y0, y1, inset):
    """在 [y0, y1) 内按行取主体自身的横向范围（左右各收 inset），得到贴轮廓的带。"""
    import numpy as _np
    height = coverage.shape[0]
    y0 = max(0, min(int(y0), height - 1))
    y1 = max(y0 + 1, min(int(y1), height))
    band = _np.zeros_like(coverage, dtype=_np.uint8)
    for y in range(y0, y1):
        row = _np.flatnonzero(coverage[y])
        if row.size == 0:
            continue
        start, end = int(row[0]), int(row[-1]) + 1
        if end - start < 6:
            continue
        cut = int((end - start) * inset)
        start, end = start + cut, end - cut
        band[y, start:end] = coverage[y, start:end]
    return band * 255          # 归一成 0/255 灰度，调用方按 128 判有效


def _keep(layer, coverage, result, name):
    """把区域裁进主体掩版后收进结果；空区域不返回。

    注意 layer 的取值已经是 0/255 的灰度，不能再乘一次 255（会溢出成乱的强度）。
    """
    import numpy as _np
    from PIL import Image as _Image
    if layer is None:
        return
    array = layer if isinstance(layer, _np.ndarray) else _np.asarray(layer.convert("L"))
    clipped = _np.where(coverage, array, 0).astype(_np.uint8)
    if (clipped > 128).any():
        result[name] = _Image.fromarray(clipped, mode="L")


def get_style_token(style) -> str:
    """规格 §17：把 VLM 分析的画风（style）映射为 prompt 风格片段。

    匹配规则（不区分大小写，子串匹配）：
        "pixel"              -> "pixel art, preserve pixelated visual style"
        "photo"/"realistic"  -> "photorealistic, preserve photographic appearance"
        "3d"/"render"        -> "3D rendered appearance"
        "anime"/"illustration"/"2d" -> "anime style"
        无匹配               -> ""（不强加任何风格约束）

    注意顺序：pixel -> photo/realistic -> 3d/render -> anime/illustration/2d，
    各分支关键词互不包含（如 "3d render" 命中 3d 分支而非 2d 分支），
    因此顺序不影响结果，按优先级排列仅为可读性。
    """
    if not style or not isinstance(style, str):
        return ""
    s = style.strip().lower()
    if "pixel" in s:
        return "pixel art, preserve pixelated visual style"
    if "painting" in s:
        return "oil painting" if "oil" in s else "painting"
    if "3d" in s or "render" in s:
        return "3D rendered appearance"
    if "photo" in s or "realistic" in s:
        return "photorealistic, preserve photographic appearance"
    if "anime" in s or "illustration" in s or "2d" in s:
        return "anime style"
    return ""


if __name__ == "__main__":
    """自检块（PyCharm 直接运行，不加载任何模型，纯 CPU）。"""
    print("=" * 60)
    print("vision/subject_analyzer.py 自检")
    print("=" * 60)

    # 1. get_style_token 全分支断言（§17）
    cases = [
        ("pixel art", "pixel art, preserve pixelated visual style"),
        ("PIXEL ART", "pixel art, preserve pixelated visual style"),
        ("realistic photo", "photorealistic, preserve photographic appearance"),
        ("photorealistic portrait", "photorealistic, preserve photographic appearance"),
        ("3D render", "3D rendered appearance"),
        ("3d", "3D rendered appearance"),
        ("anime illustration", "anime style"),
        ("2D flat illustration", "anime style"),
        ("", ""),
        (None, ""),
        ("watercolor sketch", ""),
    ]
    all_ok = True
    for style, expect in cases:
        got = get_style_token(style)
        ok = (got == expect)
        all_ok = all_ok and ok
        print(f"[1] style={style!r:32s} -> {got!r}  {'OK' if ok else 'FAIL(期望 ' + expect + ')'}")

    # 2. analyze_subject(None, img) 兜底：unknown 且字段补全（§32）
    from PIL import Image
    dummy_img = Image.new("RGB", (64, 64), (0, 0, 0))
    r = analyze_subject(None, dummy_img)
    ok = isinstance(r, dict) and r["subject_type"] == "unknown"
    ok = ok and all(k in r for k in ("pose", "species", "material", "style",
                                     "structure", "attachment_regions", "surface_regions"))
    all_ok = all_ok and ok
    print(f"[2] analyze_subject(None, img) -> {r}  {'OK' if ok else 'FAIL'}")

    # 3. analyze_image 返回 {} 的假 client 兜底（§27）
    class _FakeClient:
        def analyze_image(self, image, system_prompt, max_new_tokens=None, temperature=None):
            return {}
    r2 = analyze_subject(_FakeClient(), dummy_img)
    ok = r2["subject_type"] == "unknown"
    all_ok = all_ok and ok
    print(f"[3] analyze_image 返回 {{}} 的假 client -> {r2}  {'OK' if ok else 'FAIL'}")

    # 4. 正常结果：字段透传 + 缺失键补全 + 非法 subject_type 归 unknown
    class _GoodClient:
        def analyze_image(self, image, system_prompt, max_new_tokens=None, temperature=None):
            return {"subject_type": "animal", "species": "cat", "pose": "sitting"}
    r3 = analyze_subject(_GoodClient(), dummy_img)
    ok = (r3["subject_type"] == "animal" and r3["species"] == "cat"
          and r3["material"] is None and r3["structure"] == [])
    all_ok = all_ok and ok
    print(f"[4] 部分字段 client -> {r3}  {'OK' if ok else 'FAIL'}")

    class _BadTypeClient:
        def analyze_image(self, image, system_prompt, max_new_tokens=None, temperature=None):
            return {"subject_type": "robot", "style": "anime"}
    r4 = analyze_subject(_BadTypeClient(), dummy_img)
    ok = r4["subject_type"] == "unknown" and r4["style"] == "anime"
    all_ok = all_ok and ok
    print(f"[5] 非法 subject_type -> {r4}  {'OK' if ok else 'FAIL'}")

    print(f"SUBJECT_PROMPT 引用自 vlm/vlm_utils.py，长度={len(SUBJECT_PROMPT)}")
    print("=" * 60)
    print("全部通过" if all_ok else "存在 FAIL")
