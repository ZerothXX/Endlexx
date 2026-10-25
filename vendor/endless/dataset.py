# -*- coding: utf-8 -*-
"""dataset.py —— 训练数据入口（唯一实现）

数据来源：dataset/<角色>/images/ 的原图 + captions.txt 的自动标注。
样本只由**原图**构成，其中一部分是原图上的可信局部裁剪（学配饰、图案、花纹等细节），
其余是整图（学位置与组合关系）。不使用程序化合成图当训练数据（方案 §4.5、§5）。

训练标签：`<触发词>, <角色条件>, <质量词>, <该图标签>`
    触发词承载"这个角色"；角色条件来自 character/profile.py 的语义档案；
    该图标签来自 captions.txt（视角/构图）。三者都不写死任何具体角色。

数据集与清单：训练前由 data_registry 一次性读取并写出不可变清单，
本模块只接收普通路径与标签列表，不携带数据库连接（方案 §12）。
"""
import os
import re

from PIL import Image, ImageOps

import config
import image_utils
import text_budget
from character import cards as cards_module
from character import marks as marks_module
from character import profile as profile_module

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# 样本类型（唯一契约，供 dataset/train/tests 共用）：
#   original  整图样本：学角色整体外观、组合与位置关系
#   crop      局部裁剪样本：学标志的形状与内部结构（身份分支与概念分支都用它）
# 两个分支的区别由 regions["branch"] 记录，**不用两种样本类型区分**——
# 早期实现里概念分支写 "concept"、裁剪判断只认 "detail"，于是概念样本静默按整图训练
# （docs/统一角色标志与自动证据提取_技术调整说明_20261004.md §2 第 1 行）。
SAMPLE_ORIGINAL = "original"
SAMPLE_CROP = "crop"

# 档案参数的哨兵：不传 = 允许读盘；显式传 None / {} = 禁用身份条件（规格 §2 第 2 行）。
_PROFILE_UNSET = object()


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


# ---------------------------------------------------------------------------
# captions
# ---------------------------------------------------------------------------
def load_captions(caption_file: str) -> dict:
    """解析 captions.txt，返回 {文件名: [标签...]}。

    格式：每行 `文件名, 标签1, 标签2`。空行跳过；缺逗号的行打印警告后跳过
    （不静默吞掉），因为那通常意味着自动标注写了半行。
    """
    captions = {}
    if not os.path.isfile(caption_file):
        return captions
    with open(caption_file, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            if "," not in line:
                print(f"[dataset] 警告: {caption_file} 第 {line_no} 行缺少逗号，已跳过: {line!r}")
                continue
            filename, _, rest = line.partition(",")
            captions[filename.strip()] = [t.strip() for t in rest.split(",") if t.strip()]
    return captions


def save_captions(caption_file: str, captions: dict) -> None:
    """原子写回 captions.txt（数据准备完成后一次性发布）。"""
    os.makedirs(os.path.dirname(os.path.abspath(caption_file)), exist_ok=True)
    temp = caption_file + ".tmp"
    with open(temp, "w", encoding="utf-8") as stream:
        for name in sorted(captions, key=_natural_key):
            stream.write(f"{name}, {', '.join(captions[name])}\n")
    os.replace(temp, caption_file)


def build_text(trigger: str, clause: str, tags, quality: str = None) -> str:
    """合成训练 prompt：触发词 + 角色条件 + 质量词 + 该图标签。"""
    parts = [trigger]
    if clause:
        parts.append(clause)
    quality = config.TRAIN_QUALITY_TAGS if quality is None else quality
    if quality:
        parts.append(quality)
    if tags:
        parts.append(", ".join(str(t) for t in tags))
    return ", ".join(p for p in parts if p)


def image_visible_regions(tags, view="") -> set:
    """由一张图的既有标签推出"这张图上能看到哪些部位区域"。

    只用已经存在的证据（captions.txt 的视角/构图标签），不新增观察：
      * face close-up        -> 头/脸
      * upper body           -> 头/脸、躯干、手臂
      * full body shot       -> 头/脸、躯干、手臂、腿、脚
      * clothing / feet / pattern / accessory close-up -> 只给对应局部
      * 没有任何标签         -> 空集合（**不猜**：拿不到可见部位就不追加外观描述）
    """
    text = " ".join(str(t).lower() for t in (tags or [])) + " " + str(view or "").lower()
    if "face close-up" in text:
        return {"head", "face", "eyes"}
    if "clothing close-up" in text:
        return {"torso", "arms", "waist"}
    if "feet close-up" in text:
        return {"feet", "legs"}
    if "pattern close-up" in text or "accessory close-up" in text:
        return set()            # 局部特写图不知道是哪一块，不追加（宁可少写）
    # **背面/侧背视角必须先处理**（实测缺陷）：背面图看不到脸，也就不该拿到
    # "蓝眼睛""刘海"这类外观短语。原先没有这一条，`full body`/`upper body` 会
    # 无条件给出 head/face/eyes，于是背面图也会被写上眼睛的描述
    # （实测 captions.txt 里 017.png 是背面图却带着 `blue eyes`）。
    back = ("back view" in text) or ("from behind" in text) or ("back-facing" in text)
    if back:
        if "full body" in text:
            return {"head", "torso", "arms", "waist", "legs", "feet"}
        if "upper body" in text:
            return {"head", "torso", "arms", "waist"}
        return set()
    if "full body" in text:
        return {"head", "face", "eyes", "torso", "arms", "waist", "legs", "feet"}
    if "upper body" in text:
        return {"head", "face", "eyes", "torso", "arms", "waist"}
    return set()


# 外观短语 -> 它需要哪个部位可见才有意义。
# 这是"逐图可见证据"规则的核心：背面图不写"蓝眼睛"，特写图不写"长裙"，
# 因为那些内容在这张图上根本看不到，写进去就是给模型一个错误的图像-文本配对。
#
# **顺序即最长前缀优先**（实测缺陷）：原实现用 `phrase.startswith(key)` 遍历字典，
# 于是 `colour scheme light blue, dark blue, red` 会先命中 `"colour"` 而不是
# `"colour scheme"`；更严重的是 `eyes` 这类短键会先于 `hairstyle` 命中，
# 实测**所有**外观短语都被映射成 `torso`——"可见性门控"因此形同虚设。
_APPEARANCE_REGION = (
    ("colour scheme", "torso"),
    ("hairstyle", "head"),
    ("hair", "head"),
    ("eyes", "eyes"),
    ("eye", "eyes"),
    ("outfit", "torso"),
)


def appearance_phrase_with_region(profile) -> list:
    """`appearance_phrase` 的带区域版本：[(短语, 需要可见的区域), ...]。"""
    pairs = []
    for phrase in appearance_phrase(profile):
        lowered = str(phrase).lower()
        region = "torso"
        for key, target in _APPEARANCE_REGION:
            if key in lowered:          # 子串匹配，不再要求短语以键开头
                region = target
                break
        pairs.append((phrase, region))
    return pairs


def clean_captions(folder, profile=None) -> int:
    """清理标注文件里的历史冗余，返回改动的图片数。

    要清什么、为什么（2026-10-05 实测）：早期版本的 `appearance_phrase` 会把
    档案配色拆成裸颜色词（`white`、`blue`）追加到标注；后来改成写
    `colour scheme blue, white, pink`，但**旧内容没有被摘掉**，于是同一行里出现
    多份重复（实测 `001.png` 的标注里 `colour scheme blue, white, pink` 出现 3 次、
    `white`/`pink` 各出现多次）。这些冗余把提示词顶到 159/77 token，
    而它们提供的信息量为零。

    只做两件事，都可逆、都不新增信息：
      1. **去重**（大小写不敏感，保留首次出现）；
      2. 摘掉与档案配色**完全同名**的裸颜色词（那是被 `colour scheme ...` 覆盖的旧写法）。
    视角/构图标签与其它人工标注一律不动。
    """
    folder = folder or config.CHARACTER_ID
    profile = profile if profile is not None else profile_module.load_profile(folder)
    path = captions_path(folder)
    captions = load_captions(path)
    if not captions:
        return 0
    bare_colours = set()
    palette = (profile or {}).get("color_palette") or {}
    for value in palette.values():
        for token in re.split(r"[^A-Za-z]+", str(value).lower()):
            if token:
                bare_colours.add(token)
    changed = 0
    for name, tags in captions.items():
        cleaned, seen = [], set()
        for tag in tags:
            text = str(tag).strip()
            key = text.lower()
            if not text or key in seen:
                continue
            if key in bare_colours and "colour scheme" not in key:
                continue
            seen.add(key)
            cleaned.append(text)
        if cleaned != list(tags):
            captions[name] = cleaned
            changed += 1
    if changed:
        save_captions(path, captions)
        print(f"[dataset] 已清理 {changed} 张图的标注冗余（去重 + 摘除被 colour scheme "
              f"覆盖的裸颜色词）")
    return changed


def enrich_captions(folder=None, profile=None) -> int:
    """把角色档案里**已观察到的**外观特征追加到**确实看得见它们的**那些图的标注上。

    为什么必须做（实测定位到的根因）：`vlm.generate_tags` 按设计只产出
    "视角 + 构图"两类短标签（`CAPTION_PROMPT` 明确如此），因此**颜色信息从来没有
    进入过训练条件**。触发词只能靠图集做视觉平均，无法被"发色/配色"这类文本条件
    显式调用——实测推理时提示词里写着 `hair color light blue`，权重仍然把头发画成
    训练集里占主导的深棕色。

    为什么必须**按图判断可见性**（规格 §2 第 5 行）：
    早期实现把同一段外观短语无差别追加到每一张图，于是背面图的条件里出现
    "blue eyes"、特写图的条件里出现"long skirt"——这些内容在那张图上根本不可见，
    等于给模型一批错误的图像-文本配对，既拖慢收敛也让它学不准。

    现在只追加该图**确有可见部位**支持的短语；拿不到可见部位（无标签的图）就
    完全不追加，宁可少写也不写错。重复调用是幂等的（已含有的片段不再追加）。

    同时受**完整训练提示词的 token 预算**约束（2026-10-05 实测的严重漏项）：
    早期只检查"标注本身"，而真正的训练条件是
    `触发词 + 身份条件 + 质量词 + 该图标注`（加上逐图标志条件）。
    实测有 21 条训练标注到了 **176/77 token**——超出一倍多，
    `train.py` 的预算检查会**直接抛 ValueError 让训练起不来**。
    因此这里在写入前先把**完整提示词**拼出来量一次，超预算的短语不写入。
    """
    folder = folder or config.CHARACTER_ID
    profile = profile or profile_module.load_profile(folder)
    if not profile:
        return 0
    # 先清历史冗余，再按预算补全：否则冗余会占掉预算，导致该加的加不进去。
    clean_captions(folder, profile)
    pairs = appearance_phrase_with_region(profile)
    if not pairs:
        return 0
    # 条件里除了标注，还有触发词、身份条件、质量词与逐图标志条件，必须一起算。
    card = cards_module.build_card(profile)
    base = build_text(card.get("trigger") or config.get_trigger(folder),
                      card.get("clause") or "", [])
    mark_by_image = {}
    if config.MARK_TRAIN:
        for name in load_captions(captions_path(folder)):
            mark_by_image[name] = marks_module.by_image_condition(folder, name)
    path = captions_path(folder)
    captions = load_captions(path)
    if not captions:
        print(f"[dataset] 标注文件为空或不存在，跳过特征补全: {path}")
        return 0
    changed = 0
    skipped_budget = []
    removed_budget = []
    for name, value in captions.items():
        tags = value if isinstance(value, list) else [
            t.strip() for t in str(value).split(",") if t.strip()]
        # 先**归零**：把本函数可能写入过的外观短语全部摘掉，再按当前预算重新加。
        # 为什么必须能"减"：早期版本（没有预算检查）已经把外观短语写进了 captions.txt，
        # 那些标注实测到 176/77 token；如果只会"加"和"跳过已存在"，
        # 这些超预算标注永远留在文件里，训练入口每次都直接抛错（实测踩过）。
        # 摘掉是安全的：这些短语全部由 `appearance_phrase` 从档案派生，
        # 档案还在，随时能重新加回来；摘的匹配是精确字符串，不会误删 captions.txt
        # 里本来就有的标签。
        known = {phrase.strip().lower() for phrase, _region in pairs}
        kept_tags = [t for t in tags if str(t).strip().lower() not in known]
        if len(kept_tags) != len(tags):
            removed_budget.append(name)
        visible = image_visible_regions(kept_tags)
        if not visible:
            if len(kept_tags) != len(tags):
                captions[name] = kept_tags
                changed += 1
            continue
        lowered = {str(t).strip().lower() for t in kept_tags}
        additions = [phrase for phrase, region in pairs
                     if region in visible and phrase.strip().lower() not in lowered]
        base_tags = list(kept_tags)
        if not additions:
            if len(kept_tags) != len(tags):
                captions[name] = kept_tags
                changed += 1
            continue
        # 逐条试加：加进去之后完整提示词仍要在预算内，否则这一条不写（并如实报告）
        accepted = list(kept_tags)
        tail = [part for part in (mark_by_image.get(name),) if part]
        for phrase in additions:
            candidate = build_text(card.get("trigger") or config.get_trigger(folder),
                                   card.get("clause") or "",
                                   accepted + [phrase] + tail)
            if text_budget.fits(candidate, config.LABEL_TOKEN_BUDGET):
                accepted.append(phrase)
            else:
                skipped_budget.append(f"{name}: {phrase}")
        if accepted != base_tags or len(kept_tags) != len(tags):
            captions[name] = accepted
            changed += 1
    if changed:
        save_captions(path, captions)
        print(f"[dataset] 已按**完整提示词预算**重写 {changed} 张图的标注"
              f"（逐图按可见部位判断，且整条提示词不超 {config.LABEL_TOKEN_BUDGET} token）")
    if removed_budget:
        print(f"[dataset] 已从 {len(removed_budget)} 张图的标注里摘除超预算的外观短语"
              f"（它们由档案派生，预算够时会自动加回）")
    if skipped_budget:
        print(f"[dataset] {len(skipped_budget)} 条外观短语因完整提示词会超出 "
              f"{config.LABEL_TOKEN_BUDGET} token 预算而未写入"
              f"（避免训练入口直接报错）: {skipped_budget[:3]}")
    skipped = len(captions) - changed
    if skipped:
        print(f"[dataset] {skipped} 张图未改动（图上没有对应可见部位、预算不足或不缺该描述）")
    return changed


def appearance_phrase(profile) -> list:
    """档案里可直接用作标注的外观短语（只复述已观察到的内容，不臆造）。"""
    appearance = (profile or {}).get("appearance") or {}
    palette = (profile or {}).get("color_palette") or {}
    clothing = (profile or {}).get("clothing") or {}
    phrase = []
    hair = str(appearance.get("hair") or "").strip()
    if hair:
        phrase.append(f"{hair} hair")
    if str(appearance.get("hair_style") or "").strip():
        phrase.append(f"{str(appearance['hair_style']).strip()} hairstyle")
    eyes = str(appearance.get("eyes") or "").strip()
    if eyes:
        phrase.append(f"{eyes} eyes")
    colours = [str(v).strip() for v in palette.values() if str(v).strip()]
    if colours:
        phrase.append("colour scheme " + ", ".join(list(dict.fromkeys(colours))[:3]))
    style = str(clothing.get("style") or "").strip()
    if style:
        # 服装描述可能很长，取前一小段作为可迁移的服装特征。
        # 预算说明：训练提示词 = 触发词 + 身份条件 + 质量词 + 该图标注，上限 77 token；
        # 实测某一张图的标注把总长顶到 78 而被训练入口直接拒绝（这是好事——它挡住了一个
        # 会被 CLIP 静默截断的条件），因此这里把服装片段收紧到 36 字符并做词边界截断。
        limit = 36
        trimmed = style if len(style) <= limit else style[:limit].rsplit(" ", 1)[0]
        words = trimmed.split()
        while words and words[-1].lower() in {"a", "an", "the", "and", "with", "of", "in", "on"}:
            words.pop()
        if words:
            phrase.append("outfit " + " ".join(words))
    return phrase


# ---------------------------------------------------------------------------
# 图片扫描
# ---------------------------------------------------------------------------
def captions_path(folder) -> str:
    return os.path.join(config.DATASET_DIR, folder, "captions.txt")


def load_character_captions(folder) -> dict:
    """读取该角色的自动标注（缺文件时返回空 dict，由调用方决定是否报错）。

    训练标签必须逐图不同：如果所有样本共用一个提示词，触发词就无法与"角色的
    具体特征"建立关联，模型只能整体记忆训练集（实测症状：推理时角色化极弱）。
    """
    return load_captions(captions_path(folder))


def images_dir(folder) -> str:
    return os.path.join(config.DATASET_DIR, folder, "images")


def scan_images(folder) -> list:
    """扫描角色 images/ 下的图片，按自然顺序返回路径；目录缺失或为空即报错。"""
    root = images_dir(folder)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"[dataset] 角色图片目录不存在: {root}\n"
            f"请把图片放到 dataset/{folder}/images/ 后再训练。")
    names = sorted((n for n in os.listdir(root)
                    if n.lower().endswith(IMAGE_EXTS) and not n.startswith("_mark_")),
                   key=_natural_key)
    paths = [os.path.join(root, n) for n in names]
    if not paths:
        raise RuntimeError(f"[dataset] 角色图片目录为空: {root}")
    return paths


def validate_dataset(folder, min_images=None) -> dict:
    """训练前的数据校验：有效原始图片数必须达标（方案 §4）。

    返回 {ok, count, files, duplicates, problems}。局部裁剪与合成样本不计入这个数量。
    """
    min_images = config.MIN_DATASET_IMAGES if min_images is None else min_images
    from data_registry import file_hash
    problems, seen, duplicates, entries = [], {}, [], []
    for path in scan_images(folder):
        try:
            image = image_utils.load_image(path)
        except Exception as exc:
            problems.append(f"{os.path.basename(path)}: 无法读取（{exc}）")
            continue
        digest = file_hash(path)
        if digest in seen:
            duplicates.append({"file": os.path.basename(path), "same_as": seen[digest]})
            continue
        seen[digest] = os.path.basename(path)
        entries.append({"path": path, "rel_path": os.path.basename(path),
                        "file_hash": digest, "width": image.size[0], "height": image.size[1]})
    ok = len(entries) >= min_images
    if not ok:
        problems.append(f"有效原始图片仅 {len(entries)} 张，低于要求的 {min_images} 张；"
                        "局部裁剪、增强副本与合成样本不计入这个数量。")
    return {"ok": ok, "count": len(entries), "entries": entries,
            "duplicates": duplicates, "problems": problems}


# ---------------------------------------------------------------------------
# 裁剪策略
# ---------------------------------------------------------------------------
class MixedContentCrop:
    """整图样本的裁剪策略：按原图宽高比吸附到预算桶，尽量不裁剪。

    早期"短边对齐 + 中心裁剪"会把全身图裁成腰部中带（脸没了），因此整图样本
    一律保留完整画面，只做等比缩放与补边。
    """

    def __init__(self, size: int = None, max_long: int = None, max_area: int = None):
        self.size = int(config.RESOLUTION if size is None else size)
        self.max_long = int(config.TRAIN_MAX_LONG if max_long is None else max_long)
        self.max_area = int(config.TRAIN_MAX_AREA if max_area is None else max_area)
        self.buckets = list(config.TRAIN_BUCKETS)

    def pick_bucket(self, width, height):
        best, best_score = None, float("inf")
        for bw, bh in self.buckets:
            if bw > self.max_long or bh > self.max_long or bw * bh > self.max_area:
                continue
            score = abs((width / height) - (bw / bh))
            if score < best_score:
                best, best_score = (bw, bh), score
        return best or (self.size, self.size)

    def __call__(self, image):
        bw, bh = self.pick_bucket(*image.size)
        return ImageOps.pad(image, (bw, bh), method=Image.Resampling.LANCZOS,
                            color=(255, 255, 255))


class DetailCrop:
    """局部细节样本：在**给定的真实区域框**内裁一块，放大到工作分辨率。

    尺寸判据**只有一处**（证据层的 `vision.mark_evidence._crop_quality`）：
    它按"原始像素下限 + 放大倍数上限"决定哪条裁剪可以进入监督，
    通过之后裁剪文件会落盘，数据集直接用那个框重裁。

    这里**不再重复判一遍**：实测踩过两次——两道门用了不同的常量
    （证据层 `MARK_MAX_UPSCALE=12.0`、这里却用别的阈值），
    于是"证据层认为合格"的裁剪在训练第一次取样本时抛
    `ValueError: 细节裁剪需要放大 13.5 倍，超过上限`，训练直接崩。
    同一件事只保留一处判据，这里只保一个"图不能小到没法处理"的兜底。

    没有区域框时**不再回退到"中心偏上的一块"**：那个位置与要学的标志无关，
    等于给模型一个错误的位置监督（规格 §2 第 6 行：无有效证据框不进入标志监督，
    保留原图训练资格，不编造细节位置）。
    """

    def __init__(self, size: int = None, min_side: int = None, box=None):
        self.size = int(config.RESOLUTION if size is None else size)
        self.min_side = int(config.DETAIL_CROP_MIN_SIDE if min_side is None else min_side)
        self.box = box        # (left, top, right, bottom) 归一化坐标；必须来自真实证据

    def pick_box(self, image):
        if not self.box:
            raise ValueError("局部裁剪样本缺少真实区域框：不编造位置")
        width, height = image.size
        left, top, right, bottom = self.box
        return (int(left * width), int(top * height),
                int(right * width), int(bottom * height))

    def __call__(self, image):
        box = self.pick_box(image)
        cropped = image.crop(box)
        if min(cropped.size) < 4:
            raise ValueError(
                f"细节裁剪区域仅 {cropped.size[0]}×{cropped.size[1]} 像素，无法处理")
        if min(cropped.size) < self.min_side:
            # 小图放大到工作分辨率，让纹样在潜空间里有足够像素。
            # 放大只帮助阅读**已有像素**，不声称恢复本来不存在的纹样（规格 §5.3）；
            # 能不能放大由证据层的判据决定，这里只负责把像素做出来。
            scale = self.size / min(cropped.size)
            cropped = cropped.resize((max(8, round(cropped.size[0] * scale)),
                                      max(8, round(cropped.size[1] * scale))),
                                     Image.LANCZOS)
        return ImageOps.pad(cropped, (self.size, self.size),
                            method=Image.Resampling.LANCZOS, color=(255, 255, 255))


def make_transforms(resolution: int):
    return transforms.Compose([MixedContentCrop(size=resolution), transforms.ToTensor()])


def make_detail_transforms(resolution: int, box=None):
    return transforms.Compose([DetailCrop(size=resolution, box=box), transforms.ToTensor()])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharacterDataset(Dataset):
    """角色训练数据集：整图样本 + 真实局部裁剪样本。

    manifest 提供时，样本集合与标签以清单为准（固定数据版本，训练可复现）；
    未提供时回退到扫描 images/ 目录（仅用于自检与冒烟测试）。

    条件构造（规格 §2 第 2、4、5 行）：
      * `profile=None` 表示**禁用身份条件**，不再被 `profile or load_profile(folder)`
        悄悄替换成"重新加载档案"——概念分支曾因此一直带着身份 clause 训练。
      * 每条样本先按**该图自己的可见证据**取标志条件；没有逐图证据时才退回角色级条件。
        背面图不会写进"胸前的徽记"这类不可见内容。
      * `branch` 决定条件里是否混入身份 clause：概念分支默认不混。
    """

    def __init__(self, folder: str, resolution: int = None, profile=_PROFILE_UNSET,
                 card=None, manifest=None, image_limit: int = 0, branch: str = "identity",
                 marks_enabled=None):
        self.folder = folder
        self.resolution = int(config.RESOLUTION if resolution is None else resolution)
        self.branch = str(branch or "identity")
        # 关键：区分"没传档案"与"显式禁用档案"（规格 §2 第 2 行）。
        # 早期写法 `profile or profile_module.load_profile(folder)` 把两者混为一谈：
        # 概念分支传 profile=None 想禁用身份条件，却每次都被重新读盘并写回身份 clause。
        if profile is _PROFILE_UNSET:
            self.profile = profile_module.load_profile(folder)
        else:
            self.profile = profile          # None / {} 都表示"禁用身份条件"，不再被顶掉
        if self.profile:
            self.card = card or cards_module.build_card(self.profile)
        else:
            self.card = card or {"trigger": config.get_trigger(folder), "clause": ""}
        self.trigger = self.card.get("trigger") or config.get_trigger(folder)
        self.clause = self.card.get("clause") or ""

        self.marks_enabled = (config.MARK_TRAIN if marks_enabled is None else marks_enabled)
        self.mark_clause = marks_module.condition_clause(folder) if self.marks_enabled else ""
        self.mark_library = marks_module.load_library(folder) if self.marks_enabled else None
        self.mark_observations = (marks_module.load_observations(folder)
                                  if self.marks_enabled else [])

        self.captions = load_character_captions(folder)
        if manifest:
            self.records = [r for r in manifest.get("samples", []) if r.get("valid", True)]
            self.root = images_dir(folder)
            self.detail_by_path = {r["rel_path"]: r.get("regions") or {}
                                   for r in self.records}
            # 清单里的标签优先（固定数据版本），缺失的逐图回落到 captions.txt。
            # **裁剪样本不回填**：原图的"背面视角 / 全身"对一块局部裁剪没有意义，
            # 回填等于把整图条件塞进局部样本（规格 §2 第 3 行）。
            for record in self.records:
                if record.get("sample_type") == SAMPLE_CROP:
                    record["tags"] = []
                elif not record.get("tags"):
                    record["tags"] = list(self.captions.get(record["rel_path"]) or [])
        else:
            self.root = images_dir(folder)
            paths = scan_images(folder)
            self.records = [{"rel_path": os.path.basename(p), "tags": None,
                             "sample_type": SAMPLE_ORIGINAL} for p in paths]
            self.detail_by_path = {}
            if image_limit and image_limit > 0:
                self.records = self.records[:image_limit]
            for record in self.records:
                record["tags"] = list(self.captions.get(record["rel_path"]) or [])
        # 缺真实框的裁剪样本一律剔除：宁可少一条样本，也不给错误的位置监督
        dropped = [r["rel_path"] for r in self.records
                   if r.get("sample_type") == SAMPLE_CROP
                   and not ((r.get("regions") or {}).get("box"))]
        if dropped:
            self.records = [r for r in self.records
                            if not (r.get("sample_type") == SAMPLE_CROP
                                    and not ((r.get("regions") or {}).get("box")))]
            print(f"[dataset] 丢弃 {len(dropped)} 条没有真实区域框的裁剪样本（不编造位置）")
        if not self.records:
            raise RuntimeError(f"[dataset] 角色 {folder} 没有可用训练样本")

        missing = [r["rel_path"] for r in self.records
                   if not r.get("tags") and r.get("sample_type") == SAMPLE_ORIGINAL]
        if missing:
            print(f"[dataset] 警告: {len(missing)} 个整图样本没有标签，触发词将承担全部语义: "
                  f"{missing[:5]}")

        self.full_transform = make_transforms(self.resolution)
        self.detail_transform = make_detail_transforms(self.resolution)
        self._budget_reports = set()      # 预算裁剪的报告去重（同一样本每 epoch 都会被取）
        types = {}
        for record in self.records:
            types[record.get("sample_type", SAMPLE_ORIGINAL)] = \
                types.get(record.get("sample_type", SAMPLE_ORIGINAL), 0) + 1
        print(f"[dataset] 角色={folder} 分支={self.branch} 样本={len(self.records)} "
              f"类型={types} 分辨率={self.resolution} "
              f"标志条件={'有' if self.mark_clause else '无'}")

    def __len__(self):
        return len(self.records)

    def _tags(self, record):
        tags = record.get("tags")
        if tags is None:
            tags = self.captions.get(record["rel_path"], [])
        elif isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        return tags or []

    def record_branch(self, record) -> str:
        """这条样本属于哪个分支（记录在 regions.branch 上，缺省跟数据集分支）。"""
        regions = record.get("regions") if isinstance(record.get("regions"), dict) else {}
        return str(regions.get("branch") or self.branch)

    def mark_condition(self, record) -> str:
        """该样本的标志条件：优先**该图自己的可见证据**，没有逐图证据才退回角色级条件。

        **每条标志条件必须以它自己的专属触发词开头**（`marks.mark_token`）。
        实测根因：概念分支原先只写"共享的角色触发词 + 这条标志的描述"，
        于是 LoRA 学到的是"`chayaka` -> 这块裁片"，
        **没有任何标志与自己的描述绑定**；而推理时提示词里既没有 `chayaka`、
        又多了一串训练时没见过的词，概念权重因此根本不激活
        （物体表面只得到一层很淡的配色渐变）。
        加上专属 token 之后，"点名某个标志"才成为一个被训练过的条件。
        """
        if not self.marks_enabled or not self.mark_clause:
            return ""
        trigger = config.get_trigger(self.folder)
        if record.get("sample_type") == SAMPLE_CROP:
            # 裁剪样本的条件**只有**它自己的标志描述：既不能沿用原图的视角/构图，
            # 也不能退回角色级条件——那会把这块裁剪没显示的东西一起写进去。
            regions = record.get("regions") or {}
            description = str(regions.get("mark_description") or "").strip()
            mark_id = str(regions.get("mark_id") or "").strip()
            token = ""
            if mark_id:
                for mark in (self.mark_library or {}).get("marks") or []:
                    if mark.get("mark_id") == mark_id:
                        token = marks_module.mark_token(mark, trigger)
                        break
            if not token and mark_id:
                token = marks_module.mark_token({"mark_id": mark_id}, trigger)
            if token and description:
                return f"{token}, {description}"
            return description or token
        own = marks_module.by_image_condition(self.folder, record["rel_path"])
        if own:
            return own
        # **这张图有逐图证据、但一条都不"可见/部分可见"时，标志条件必须留空**（实测缺陷）。
        #
        # 原写法 `return own or self.mark_clause`：`own` 为空就退回角色级条件
        # （"凡是某张图里看得见的标志"），于是**只看到正脸的正面特写**也会被写上
        # "胸口的图案 + 颈部的挂饰"这类在这张图上根本不可见的内容——
        # 给模型的是一对错误的图像-文本配对。实测在真实数据上复现：
        # 019.png（正面 + 脸部特写）的逐图证据为空，拿到的却是整条角色级标志串。
        #
        # 只有在"这张图完全没有逐图证据"时才退回角色级条件（例如 VLM 未启用、
        # 整批证据都是 unknown 的情形），那时退回是唯一可用的信息。
        if marks_module.has_image_evidence(self.folder, record["rel_path"]):
            return ""
        return self.mark_clause

    def get_prompt(self, idx: int) -> str:
        """训练条件：`触发词 + 分支条件 + 质量词 + 该图标注 [+ 该图标志条件]`。

        **这里是预算的最后一道关**：`train.py` 会逐条检查并**直接抛错**，
        因此任何超预算的组装都必须在这里被解决，而不是指望上游各自算准——
        实测（2026-10-05）上游三处各算各的，结果 21 条标注到 176/77 token，
        训练入口一启动就报错。

        裁剪顺序（从最可丢的开始丢）：
          1. 标志条件 → 2. 该图标注 → 3. 质量词 → 4. 分支条件 → 5. 触发词。
        裁剪是**按逗号条目**做的，不是把字符串砍成半截；
        丢掉了什么会打印出来（带样本名），不静默截断。
        """
        record = self.records[idx]
        branch = self.record_branch(record)
        # 概念分支默认不带身份 clause：它学的是"标志长什么样"，
        # 混进"这个角色是谁"会把身份与局部纹样绑死（规格 §2 第 2 行）。
        clause = "" if branch == "concept" else self.clause
        quality = str(config.TRAIN_QUALITY_TAGS or "")
        tags = [str(t) for t in self._tags(record)]
        mark_text = self.mark_condition(record)
        budget = int(config.LABEL_TOKEN_BUDGET)

        def assemble(tag_list, mark_parts, quality_text, clause_text):
            return build_text(self.trigger, clause_text,
                              list(tag_list) + list(mark_parts), quality=quality_text)

        def pieces(text):
            return [part.strip() for part in str(text or "").split(",") if part.strip()]

        mark_parts = pieces(mark_text)
        dropped = []
        prompt = assemble(tags, mark_parts, quality, clause)
        while text_budget.count(prompt) > budget:
            if mark_parts:
                dropped.append(f"标志条目 {mark_parts.pop()!r}")
            elif tags:
                dropped.append(f"标注 {tags.pop()!r}")
            elif quality:
                dropped.append(f"质量词 {quality!r}")
                quality = ""
            elif clause:
                dropped.append(f"分支条件 {clause!r}")
                clause = ""
            else:
                # 只剩触发词还超预算：这是配置问题（触发词本身过长），如实报错
                raise ValueError(
                    f"[dataset] 触发词本身超出 CLIP 预算（{text_budget.count(prompt)} > "
                    f"{budget}）: {self.trigger!r}")
            prompt = assemble(tags, mark_parts, quality, clause)
        if dropped:
            # 同一个样本每个 epoch 都会被取一次，逐次打印会把日志刷满
            # （实测 600 步里同一条警告重复 8 次以上），因此同一组合只报一次。
            signature = (record["rel_path"], tuple(dropped))
            if signature not in self._budget_reports:
                self._budget_reports.add(signature)
                print(f"[dataset] {record['rel_path']} 的训练条件超出 {budget} token，"
                      f"已按优先级丢弃: {dropped[:4]}")
        return prompt

    def evaluation_indices(self):
        """固定诊断集：最先出现的若干张整图样本（一经确定不因效果差而替换）。"""
        indices = [i for i, r in enumerate(self.records)
                   if r.get("sample_type", SAMPLE_ORIGINAL) == SAMPLE_ORIGINAL]
        if not indices:
            indices = list(range(len(self.records)))
        return indices[:max(1, int(config.EVAL_SAMPLE_COUNT))]

    def get_item(self, idx: int, deterministic=False) -> dict:
        record = self.records[idx]
        path = os.path.join(self.root, record["rel_path"])
        image = image_utils.load_image(path)
        regions = record.get("regions") or {}
        if not isinstance(regions, dict):
            regions = {}
        # 唯一裁剪判断：样本类型是 crop 就裁剪，且必须带真实框。
        # 早期实现写 concept、判断只认 detail，概念样本静默按整图训练（规格 §2 第 1 行）。
        if record.get("sample_type") == SAMPLE_CROP:
            box = regions.get("box")
            if not box:
                raise ValueError(
                    f"裁剪样本缺少真实区域框，拒绝按整图训练: {record['rel_path']}")
            pixel_values = make_detail_transforms(self.resolution, box=box)(image)
        else:
            pixel_values = self.full_transform(image)
        item = {"pixel_values": pixel_values, "prompt": self.get_prompt(idx),
                "sample_type": record.get("sample_type", SAMPLE_ORIGINAL),
                "branch": self.record_branch(record),
                "rel_path": record["rel_path"]}
        weights = regions.get("loss_weights")
        if weights:
            import torch
            item["loss_weights"] = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)
        return item

    def __getitem__(self, idx: int) -> dict:
        return self.get_item(idx)


# ---------------------------------------------------------------------------
# 清单构建
# ---------------------------------------------------------------------------
def region_weight_map(record, profile, tags):
    """为一条样本生成"区域加权"图（0~1 的乘性权重，None 表示不启用）。

    动机：角色的发色/服装颜色在整图 loss 里只占很小一块像素，模型容易把它们
    当成次要信息而不学。这里让被点名区域（发区、服装区）的像素在 loss 里权重更高，
    相当于把学习容量导向真正需要还原的地方。

    框来自两个确定性来源，不做臆测：
      1. 档案里 visual_features 的 region 证据（衣服/配饰等，由 VLM 观察得到）；
      2. 自动标注的景别（face close-up / upper body / full body）推出发区大致位置——
         发色是角色最醒目的特征，而档案里没有专门的"头发"特征项。
    """
    import numpy as np
    weights = np.ones((8, 8), dtype=np.float32)      # 8x8 网格与潜空间的粗粒度对齐
    touched = False
    lowered = {str(t).lower() for t in tags}

    # 1) 发区：按景别取画面上部（近景占比更大，因为头发占画面的比例更高）
    if "face close-up" in lowered:
        top_ratio = 0.75
    elif "upper body" in lowered or "full body shot" in lowered:
        top_ratio = 0.40
    elif "back view" in lowered:
        top_ratio = 0.45
    else:
        top_ratio = 0.0
    if top_ratio > 0:
        rows = max(1, int(round(weights.shape[0] * top_ratio)))
        weights[:rows, :] = config.REGION_LOSS_WEIGHT
        touched = True

    # 2) 档案里的区域证据：把对应区域的行带再次加权（服装、配饰等）
    regions = {str(f.get("region") or "").lower()
               for f in (profile or {}).get("visual_features") or []}
    if regions & {"legs", "feet"}:
        weights[int(weights.shape[0] * 0.62):, :] = config.REGION_LOSS_WEIGHT
        touched = True
    if regions & {"waist", "torso", "chest"}:
        start = int(weights.shape[0] * 0.35)
        end = int(weights.shape[0] * 0.70)
        weights[start:end, :] = config.REGION_LOSS_WEIGHT
        touched = True
    if not touched:
        return None
    import torch
    return torch.from_numpy(weights)


def crops_from_marks(folder, image_limit=None, branch="concept") -> list:
    """由**真实标志证据**构造裁剪条目（规格 §6：局部证据学标志形状与内部结构）。

    与旧的"随机挑几张图做中心裁剪"相比，这里每一条裁剪都绑定：
      * 它来自哪张原图的哪个真实区域框（VLM 定位 + 位置可信度检查后的框）；
      * 它学的是哪个标志（写进 regions.mark_id / mark_description）；
      * 它在这张图里是 visible 还是 partial（partial 只监督实际可见的部分）。
    没有证据就不产出条目——保留原图训练资格，不编造细节位置。
    """
    from vision import mark_evidence
    entries = []
    for crop in mark_evidence.training_crops(folder):
        rel_path = crop.get("rel_path")
        box = crop.get("box")
        if not rel_path or not box:
            continue
        source = image_entry(folder, rel_path)
        if source is None:
            continue
        entries.append({
            "rel_path": rel_path, "original_name": rel_path,
            "file_hash": source["file_hash"], "width": source["width"],
            "height": source["height"],
            "tags": [],          # 裁剪样本的条件只描述这块内容，不沿用原图的视角/构图
            "valid": True, "sample_type": SAMPLE_CROP, "source_sample": rel_path,
            "regions": {"box": list(box), "branch": branch,
                        "mark_id": crop.get("mark_id"),
                        "kind": crop.get("mark_class") or "detail",
                        "name": crop.get("mark_name") or "",
                        "mark_description": mark_condition_text(folder, crop.get("mark_id")),
                        "visibility": crop.get("visibility"),
                        "evidence": crop.get("crop_path") or ""},
        })
    # 这里**不再**按原图像素复核尺寸：证据层已经用 `_crop_quality` 判过一次，
    # 判据是"放大到工作尺寸后是否清晰"（`MARK_MIN_SOURCE_PIXELS` + `MARK_MAX_UPSCALE`）。
    # 早期这里又套了一遍 `DETAIL_CROP_MIN_SIDE`（256 个**原图**像素），
    # 于是 56 条已经通过证据层判定的裁剪里有 52 条被二次否决
    # （腰间结绳这类真实标志在原图上只有 50~150 像素，是正常的）。
    # 两道不同量纲的门槛叠在一起，等于把标定好的证据层判据悄悄推翻。
    if image_limit and image_limit > 0:
        keep = {os.path.basename(p) for p in scan_images(folder)[:image_limit]}
        entries = [e for e in entries if e["rel_path"] in keep]
    return entries


def mark_condition_text(folder, mark_id) -> str:
    """某个标志的条件文本（只复述已结构化的内容，取不到就返回空串）。"""
    library = marks_module.load_library(folder)
    if not library:
        return ""
    for mark in library.get("marks") or []:
        if mark.get("mark_id") == mark_id:
            return marks_module.mark_condition(mark)
    return ""


def image_entry(folder, rel_path):
    """按相对路径取该图的尺寸与哈希（用于给裁剪条目补齐元数据）。"""
    from data_registry import file_hash
    path = os.path.join(images_dir(folder), rel_path)
    if not os.path.isfile(path):
        return None
    try:
        image = image_utils.load_image(path)
    except Exception as exc:
        print(f"[dataset] {rel_path}: 无法读取（{type(exc).__name__}: {exc}）")
        return None
    return {"rel_path": rel_path, "file_hash": file_hash(path),
            "width": image.size[0], "height": image.size[1],
            "tags": list(load_character_captions(folder).get(rel_path) or [])}


# 构图标签 -> 脸部在"人物外接框"里的纵向位置与尺寸（比例）。
# 依据很直接：`face close-up` 整幅就是脸；`upper body` 到胸口，头占上部较大一块；
# `full body shot` 头只占最上面一小块。取值偏保守（宁可框大一点带上脖子，
# 也不要把脸切掉一半——切一半会让 LoRA 学到错误的解剖结构）。
_FACE_FRAME = {
    "face close-up": (0.00, 1.00, 1.00, 0.00),
    "upper body": (0.00, 0.42, 0.80, 0.06),
    "full body shot": (0.00, 0.20, 0.62, 0.04),
}
_FACE_FRAME_DEFAULT = (0.00, 0.35, 0.75, 0.05)


def face_box_from_subject(subject_box, tags):
    """由"人物外接框 + 该图构图标签"推出**脸部框**（归一化，已收进画面内）。

    为什么这样推（而不是再问一次模型）：角色档案里 `visual_features` 只有
    region 名、**没有任何框**，而跑一遍 `locate_regions` 要 VLM+SAM 十几分钟。
    而"脸在画面哪个位置"完全由构图上就能定：`face close-up` 整幅是脸、
    `upper body` 取人物框上 42%、`full body shot` 取上 20%。
    人物框本身由主体掩版给出（已经有），因此这一步是确定性的、零额外模型调用。

    实测依据（本轮）：22 张训练图里只有 **1 张**是 `face close-up`，
    `upper body` 下脸只占画面高度 15%~25%，512 分辨率下脸部约 80~130 像素——
    身份权重几乎没学到脸。补脸部裁剪样本就是为了给脸足够的像素预算。
    """
    if not subject_box:
        return None
    left, top, right, bottom = [float(v) for v in subject_box]
    lowered = {str(t).lower() for t in (tags or [])}
    for key, frame in _FACE_FRAME.items():
        if key in lowered:
            return _apply_face_frame((left, top, right, bottom), frame)
    return _apply_face_frame((left, top, right, bottom), _FACE_FRAME_DEFAULT)


def _apply_face_frame(box, frame):
    left, top, right, bottom = box
    rel_top, rel_bottom, width_ratio, margin = frame
    span_y = bottom - top
    span_x = right - left
    face_top = top + span_y * rel_top - margin
    face_bottom = top + span_y * rel_bottom + margin
    centre_x = (left + right) / 2.0
    half_w = span_x * width_ratio / 2.0
    face_left = centre_x - half_w
    face_right = centre_x + half_w
    face_left = min(max(0.0, face_left), 1.0)
    face_top = min(max(0.0, face_top), 1.0)
    face_right = min(max(face_left + 0.02, face_right), 1.0)
    face_bottom = min(max(face_top + 0.02, face_bottom), 1.0)
    return [round(face_left, 6), round(face_top, 6),
            round(face_right, 6), round(face_bottom, 6)]


def face_condition_text(profile, max_chars=90) -> str:
    """脸部裁剪样本的训练条件：**只写脸部的可辨特征**（发色 / 瞳色 / 发型）。

    为什么不复用整图身份条件：那条条件里有服装、配色、身份叙述，
    对一块"脸部裁剪"来说大半是无关信息，而且在 77 token 预算里
    会把真正与脸相关的部分挤掉。这里只取 `appearance` 里的三项。
    没有可用特征时返回空串——**不编造**，交给触发词承担。
    """
    appearance = (profile or {}).get("appearance") or {}
    pieces = []
    hair = str(appearance.get("hair") or "").strip()
    if hair:
        pieces.append(f"hair color {hair}")
    eyes = str(appearance.get("eyes") or "").strip()
    if eyes:
        pieces.append(f"eye color {eyes}")
    style = str(appearance.get("hair_style") or "").strip()
    if style:
        pieces.append(f"hairstyle {style}")
    text = ", ".join(pieces)
    return text[:max_chars] if len(text) > max_chars else text


def build_face_entries(folder, detections, profile=None, manifest=None) -> list:
    """由"人物外接框 + 构图标签"构建**脸部裁剪训练样本**（规格 §2 第 1 行的同类处理）。

    与 `build_concept_entries` 同一形态（`sample_type="crop"` + `regions.box`），
    因此复用同一条 `DetailCrop` 训练链路，不引入第二套裁剪机制。

    条件（`regions.mark_description`）**刻意保持简短**：这里是"角色的脸"，
    不是某个配饰；写太长会挤占 77 token 预算，也会把"脸"和"配饰"混在一起。
    可用的描述来自档案的外观字段（发色/瞳色/发型），它们正是脸部可辨特征。
    """
    entries = []
    missing_meta = []
    for rel_path, items in (detections or {}).items():
        for item in items or []:
            if not isinstance(item, dict) or item.get("kind") != "face":
                continue
            box = item.get("box")
            if not box:
                continue
            width = item.get("width")
            height = item.get("height")
            if not width or not height:
                missing_meta.append(rel_path)
                continue
            entries.append({
                "rel_path": rel_path, "original_name": rel_path,
                "file_hash": item.get("file_hash") or "",
                "width": width, "height": height,
                "tags": [],          # 裁剪样本不继承整图的视角/构图（否则等于把整图条件塞进去）
                "valid": True, "sample_type": SAMPLE_CROP,
                "source_sample": rel_path,
                "regions": {
                    "box": list(box), "branch": "identity",
                    "name": str(item.get("name") or "face"),
                    "kind": "face", "mark_id": None,
                    "mark_description": str(item.get("condition") or ""),
                    "visibility": "visible",
                },
            })
    if missing_meta:
        print(f"[dataset] 警告: {len(set(missing_meta))} 张图的检测项缺少尺寸元数据，"
              f"脸部样本被跳过: {sorted(set(missing_meta))[:5]}")
    if entries:
        print(f"[dataset] 产出 {len(entries)} 条脸部裁剪样本"
              f"（来自 {len({e['rel_path'] for e in entries})} 张原图；"
              f"定位由视觉模型给出并**逐张复核**通过）")
    return entries


def build_manifest_entries(folder, profile, validation, detail_ratio=None, seed=None):
    """把校验通过的图片整理成训练清单条目（整图 + 真实局部裁剪）。

    裁剪来源有两条，按优先级取：
      1. 统一标志证据库里**有真实框**的逐图证据（首选：位置来自实际定位）；
      2. 档案里的特征区域（`visual_features.source_images`）——但它只有"区域名"，
         没有像素级框，因此**只登记为无框的观察记录，不生成裁剪样本**
         （规格 §2 第 6 行：无有效证据框不进入标志监督）。
    """
    del detail_ratio, seed          # 旧参数：裁剪数量现在由证据数量决定，不再按比例随机凑
    tag_map = load_character_captions(folder)
    entries = []
    for item in validation["entries"]:
        tags = list(tag_map.get(item["rel_path"]) or [])
        weights = region_weight_map(item, profile, tags) if config.REGION_WEIGHTED_LOSS else None
        base = {"rel_path": item["rel_path"], "original_name": item["rel_path"],
                "file_hash": item["file_hash"], "width": item["width"],
                "height": item["height"], "tags": tags,
                "valid": True, "sample_type": SAMPLE_ORIGINAL, "source_sample": None,
                "regions": ({"loss_weights": weights.tolist() if hasattr(weights, "tolist")
                             else list(weights)}
                            if weights is not None else {})}
        entries.append(dict(base))
    if config.MARK_TRAIN:
        crops = crops_from_marks(folder, branch="identity")
        if crops:
            entries.extend(crops)
            print(f"[dataset] 由统一标志证据产出 {len(crops)} 条真实裁剪样本")
    # 脸部裁剪样本：由"主体外接框 + 构图标签"推导（见 train.prepare_face_entries）。
    # 实测依据：22 张训练图里只有 1 张 `face close-up`，512 分辨率下脸部
    # 只有约 80~130 像素，身份权重几乎没学到脸——补脸部样本就是给脸足够的像素预算。
    if getattr(config, "FACE_SAMPLE_ENABLED", True):
        try:
            import train as train_module
            face_entries = train_module.face_entries(folder, validation)
            if face_entries:
                entries.extend(face_entries)
                print(f"[dataset] 补充 {len(face_entries)} 条脸部裁剪样本")
        except Exception as exc:      # noqa: BLE001
            # 脸部样本是**增益**而不是必需项：拿不到就如实报告，不让整条流程失败
            print(f"[dataset] 脸部样本生成失败（不影响训练）: {type(exc).__name__}: {exc}")
    return entries


def build_concept_entries(folder, detections, profile, manifest, min_side=None):
    """用"自动观察到的细节区域"生成概念样本清单（真实裁剪，不合成）。

    detections: {rel_path: [{"box": (l,t,r,b) 归一化, "kind": str, "name": str,
                             "width": int, "height": int, "file_hash": str}, ...]}
      —— 由逐标志证据（`vision/mark_evidence`）产出，见 `train.prepare_concept_entries`。

    **尺寸与哈希必须来自 detections 自身**（实测踩过的静默退化）：
    早期版本只在 `manifest["samples"]` 里查源图元数据，而这个清单通常只装了**整图**
    样本（调用方传的是 `validation["entries"]`），于是每条检测都命中不到源图、
    被 `continue` 全部丢掉——训练日志显示
    `概念裁剪样本: 4 条（来自 19 张原图）`（那 4 条是参考图，不需要源图元数据），
    而证据层其实有 60 条可用裁剪，概念分支因此被"样本不足"跳过。
    现在优先用检测项自带的尺寸，manifest 只作为兜底，并在两者都缺失时**明确报告**。

    所有裁剪样本统一写成 `sample_type="crop"`（与整图样本的**唯一**区分方式），
    分支用 `regions.branch` 记录。早期实现写 "concept" 而裁剪判断只认 "detail"，
    概念样本静默按整图训练（规格 §2 第 1 行）。

    `min_side` 参数保留只为向后兼容（历史调用与测试会传它），**不再用于否决**：
    尺寸判据统一在证据层（`vision.mark_evidence._crop_quality`）。
    """
    del min_side
    by_path = {entry["rel_path"]: entry for entry in manifest.get("samples", [])}
    entries = []
    missing_meta = []
    for rel_path, items in detections.items():
        source = by_path.get(rel_path) or {}
        for index, item in enumerate(items):
            box = item.get("box")
            if not box:
                continue          # 无有效证据框 -> 不进入标志监督（不编造位置）
            width = item.get("width") or source.get("width")
            height = item.get("height") or source.get("height")
            if not width or not height:
                missing_meta.append(rel_path)
                continue
            left, top, right, bottom = box
            # 这里**不再**按原图像素复核尺寸（`DETAIL_CROP_MIN_SIDE` 只在下方的
            # `DetailCrop` 以外不再使用）：证据层已经用 `_crop_quality`
            # （原始像素下限 + 放大倍数上限）判定过一次，那是唯一权威判据。
            # 早期在这里又套了一遍 256 **原图**像素的门槛，于是 60 条已经通过
            # 证据层判定的裁剪里有 56 条被二次否决——概念分支因此报"样本不足"被跳过。
            # 实测数据（001.png 1242×1656）：腰间结绳 96×54、祥云纹 184×59、
            # 外袖 420×124——这些就是角色图上真实标志的尺寸，本来就不该按整图量纲卡。
            entries.append({
                "rel_path": rel_path, "original_name": rel_path,
                "file_hash": item.get("file_hash") or source.get("file_hash") or "",
                "width": width, "height": height,
                # 条件只描述这块裁剪里实际可见的内容：标志自身的描述优先，
                # 原图的视角/构图标签只在**裁剪确实覆盖整图时**才有意义，因此不再沿用。
                "tags": [],
                "valid": True, "sample_type": SAMPLE_CROP,
                "source_sample": rel_path,
                "regions": {"box": list(box), "branch": "concept",
                            "name": item.get("name") or "",
                            "kind": item.get("kind") or "detail",
                            "mark_id": item.get("mark_id"),
                            "mark_description": item.get("mark_description") or "",
                            "visibility": item.get("visibility") or "visible"},
            })
    if missing_meta:
        # 不能静默丢：尺寸缺失会让整批裁剪消失，而日志里只看到"样本不足"
        print(f"[dataset] 警告: {len(missing_meta)} 张原图的检测项缺少尺寸元数据，"
              f"已跳过其裁剪样本: {sorted(set(missing_meta))[:5]}")
    return entries


def dataset_from_manifest(folder, manifest, resolution=None, profile=_PROFILE_UNSET,
                          branch="identity") -> CharacterDataset:
    return CharacterDataset(folder=folder, resolution=resolution, profile=profile,
                            manifest=manifest, branch=branch)


def get_train_dataloader(dataset: Dataset, batch_size: int = None, shuffle: bool = True,
                         num_workers: int = None) -> DataLoader:
    """构建训练 DataLoader；shuffle 用固定种子，保证多次运行顺序一致。"""
    batch_size = config.TRAIN_BATCH_SIZE if batch_size is None else batch_size
    num_workers = config.DATALOADER_NUM_WORKERS if num_workers is None else num_workers
    generator = None
    if shuffle:
        import torch
        generator = torch.Generator().manual_seed(config.SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False, generator=generator)


if __name__ == "__main__":
    print("=" * 60)
    print("dataset.py 自检（只读，不加载模型）")
    print("=" * 60)
    folder = config.CHARACTER_ID
    report = validate_dataset(folder)
    print(f"[1] 数据校验: ok={report['ok']} 有效图片={report['count']} "
          f"重复={len(report['duplicates'])}")
    for problem in report["problems"]:
        print(f"    ! {problem}")
    profile = profile_module.load_profile(folder, allow_stale=True)
    entries = build_manifest_entries(folder, profile, report)
    print(f"[2] 清单条目: {len(entries)}（其中真实裁剪样本 "
          f"{sum(1 for e in entries if e['sample_type'] == SAMPLE_CROP)}）")
    dataset = CharacterDataset(folder, profile=profile, image_limit=2)
    item = dataset[0]
    print(f"[3] 样本: shape={tuple(item['pixel_values'].shape)} type={item['sample_type']}")
    print(f"    prompt={item['prompt'][:200]}")
    crops = [i for i, r in enumerate(dataset.records)
             if r.get("sample_type") == SAMPLE_CROP]
    if crops:
        cropped = dataset.get_item(crops[0])
        print(f"[4] 裁剪样本自检: shape={tuple(cropped['pixel_values'].shape)} "
              f"box={dataset.records[crops[0]]['regions'].get('box')}")
        print(f"    prompt={cropped['prompt'][:200]}")
    print("=" * 60)
