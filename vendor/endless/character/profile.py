# -*- coding: utf-8 -*-
"""character/profile.py —— 角色视觉语义档案（单一实现）

职责：
1. 读取/生成 dataset/<角色>/ 的角色语义档案（发型、配色、服装结构、配饰、图案等）。
2. 只做"档案"这一件事：不生成提示词条件、不决定生成参数（那是 subject_transfer 的职责）。
3. 档案可以来自已缓存的 VLM 观察，也可以来自 config.CHARACTER_PROFILES 的人工覆盖；
   两者格式相同，程序不区分来源，也不为某个角色写专用分支。

档案 JSON 结构（与历史 output/semantic/<角色>_character_memory.json 兼容）：
    {
      "folder": "ayaka",
      "identity": "一句话身份描述",
      "appearance": {"hair": ..., "hair_style": ..., "eyes": ..., "face_features": ...},
      "color_palette": {"main": ..., "secondary": ..., "accent": ...},
      "clothing": {"style": ...},
      "accessories": [{"name","shape","color","description","transfer_role",...}],
      "signature_accessories": [...],
      "visual_features": [{"name","description","design_cue","region","kind","importance"}],
      "visual_style": "Anime illustration",
      "source": "vlm" | "manual" | "offline",
      "dataset_fingerprint": "<sha256>"
    }
"""
import hashlib
import json
import os
from pathlib import Path

import config

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# 视觉特征只在这些区域里有意义；越界的 observation 视为噪声（规格：不臆测未观察到的特征）
VALID_REGIONS = {"hair", "face", "eyes", "head", "torso", "chest", "arms", "hands",
                 "legs", "feet", "waist", "surface", "body", "whole", "neck", "back"}
# 观察器的区域词表与本模块的白名单不完全一致，这里显式对齐而不是放宽白名单
_REGION_ALIASES = {"upper body": "torso", "sleeves": "arms", "arm": "arms",
                   "leg": "legs", "foot": "feet", "hands": "arms",
                   "right hand": "arms", "left hand": "arms"}


def profile_path(folder):
    return os.path.join(config.SEMANTIC_DIR, f"{folder}_character_profile.json")


def legacy_profile_path(folder):
    """历史文件名，只读兼容；新档案一律写 profile_path()。"""
    return os.path.join(config.SEMANTIC_DIR, f"{folder}_character_memory.json")


def dataset_files(folder):
    """返回该角色 images/ 下参与训练的图片（自然排序），不递归。"""
    root = Path(config.DATASET_DIR) / folder / "images"
    if not root.is_dir():
        return []
    names = [n for n in os.listdir(root) if n.lower().endswith(IMAGE_EXTS)]
    return sorted((root / n for n in names), key=lambda p: _natural_key(p.name))


def _natural_key(name):
    import re
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def dataset_fingerprint(folder):
    """数据集内容指纹：文件名 + 文件大小 + mtime，用于判断档案是否过期。"""
    digest = hashlib.sha256()
    for path in dataset_files(folder):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8"))
    return digest.hexdigest()


def fingerprint_files(folder, include_captions=None):
    """参与缓存键的**输入文件**：原图 + 两份标志描述文件（+ 可选：captions.txt）。

    规格 §2 结尾要求把用户输入都纳入缓存键：只比较图片指纹时，
    用户改了 marks.txt / mark_refs.txt 却不会让旧档案与旧标志库失效。

    **captions.txt 默认不参与**（实测踩过，代价很大）：
    它由 `dataset.enrich_captions` 每轮训练自动重写，而档案生成又发生在它之前，
    于是"档案写好 → captions 被重写 → 档案立刻过期"成了固定循环——
    表现为**每次训练/推理都重建一次角色档案**（多花约 8 分钟 VLM 时间），
    更糟的是 `test.py` 会在启动时因"语义档案缺失"直接报错退出。
    档案的内容来自**图像观察**，不依赖 captions；captions 只是训练条件的标签，
    因此不该由它决定档案是否过期。需要重新标注判定时用 `include_captions=True`。
    """
    files = list(dataset_files(folder))
    folder_dir = Path(config.DATASET_DIR) / folder
    names = ["marks.txt", "mark_refs.txt"]
    if include_captions is None:
        include_captions = bool(getattr(config, "PROFILE_FINGERPRINT_INCLUDES_CAPTIONS", False))
    if include_captions:
        names.append("captions.txt")
    for name in names:
        path = folder_dir / name
        if path.is_file():
            files.append(path)
    return files


def cache_fingerprint(folder, include_captions=None):
    """完整缓存键：输入文件 + 模型版本 + 标注规则版本 + 提示词版本。"""
    digest = hashlib.sha256()
    for path in fingerprint_files(folder, include_captions=include_captions):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8"))
    digest.update(str(getattr(config, "VLM_PATH", "")).encode("utf-8"))
    digest.update(str(getattr(config, "MARK_PROMPT_VERSION", "")).encode("utf-8"))
    digest.update(str(getattr(config, "LABEL_RULE_VERSION", "")).encode("utf-8"))
    return digest.hexdigest()


def normalize_profile(folder, raw):
    """把任意来源的原始观察整理成统一档案结构；缺失字段留空而不是编造。

    兼容两种输入形态（都来自本项目自己的观察器，不做猜测性解析）：
      A. 逐图观察（character_parser.normalize_character_result 的产物）：
         character_identity / hair{color,style} / face{eye_color,face_features} /
         clothing{description} / color_palette[list]
      B. 已合并档案（本模块输出，或历史 *_character_memory.json）：
         identity / appearance{...} / color_palette{main,secondary,accent} / clothing{style}

    缓存过期的判定在 `load_profile` 里**比较落盘值**，本函数只负责生成当前指纹
    （规格 §2 结尾：归一化会重写指纹，直接拿归一化结果与当前值比较就永远相等，
    过期检测会失去意义）。
    """
    raw = raw if isinstance(raw, dict) else {}
    appearance = raw.get("appearance") if isinstance(raw.get("appearance"), dict) else {}
    hair = raw.get("hair") if isinstance(raw.get("hair"), dict) else {}
    face = raw.get("face") if isinstance(raw.get("face"), dict) else {}
    palette = raw.get("color_palette")
    if isinstance(palette, list):
        # 形态 A：扁平颜色名列表 -> 主/次/点缀
        palette = {"main": palette[0] if len(palette) > 0 else "",
                   "secondary": palette[1] if len(palette) > 1 else "",
                   "accent": palette[2] if len(palette) > 2 else ""}
    elif not isinstance(palette, dict):
        palette = {}
    clothing = raw.get("clothing") if isinstance(raw.get("clothing"), dict) else {}
    accessories = [a for a in (raw.get("accessories") or []) if isinstance(a, dict) and a.get("name")]
    signature = [a for a in (raw.get("signature_accessories") or [])
                 if isinstance(a, dict) and a.get("name")]
    if not accessories:
        # 观察器只产出 signature_accessories 时，用同一份清单，避免"两套并存"
        accessories = [dict(a) for a in signature]
    features = []
    warnings = []
    for item in raw.get("visual_features") or []:
        if not isinstance(item, dict) or not item.get("name") or not item.get("description"):
            warnings.append(f"visual_features: 缺 name/description，已忽略: {item!r}"[:200])
            continue
        region = str(item.get("region") or "").strip().lower()
        region = _REGION_ALIASES.get(region, region)
        if region and region not in VALID_REGIONS:
            warnings.append(f"visual_features[{item.get('name')}]: 区域 {region!r} 不在白名单，按 whole 处理")
            region = "whole"
        features.append({
            "name": item["name"],
            "description": item["description"],
            "design_cue": item.get("design_cue") or "",
            "region": region or "whole",
            "kind": item.get("kind") or "detail",
            "importance": item.get("importance") or "medium",
            "source_images": [str(n) for n in (item.get("source_images") or [])],
        })
    return {
        "folder": folder,
        "identity": raw.get("identity") or raw.get("character_identity") or "",
        "appearance": {
            "hair": appearance.get("hair") or hair.get("color") or "",
            "hair_style": appearance.get("hair_style") or hair.get("style") or "",
            "eyes": appearance.get("eyes") or face.get("eye_color") or "",
            "face_features": appearance.get("face_features") or face.get("face_features") or "",
        },
        "color_palette": {k: palette.get(k) or "" for k in ("main", "secondary", "accent")},
        "clothing": {"style": clothing.get("style") or clothing.get("description") or ""},
        "accessories": accessories,
        "signature_accessories": signature,
        "visual_features": features,
        "visual_style": raw.get("visual_style") or "",
        "source": raw.get("source") or "offline",
        "warnings": warnings,
        # 逐图观察记录（全部有效原图，不只代表图）：规格 §2 第 7 行。
        # 这里保留原始结构，供"某张图能看到什么"的查询使用；空字典表示这次没有做。
        "image_observations": (raw.get("image_observations")
                               if isinstance(raw.get("image_observations"), dict) else {}),
        "dataset_fingerprint": dataset_fingerprint(folder),
        "cache_fingerprint": cache_fingerprint(folder),
    }


def load_profile(folder, allow_stale=False):
    """读取已落盘的档案；不存在或过期返回 None（不触发生成，生成由 build_profile 负责）。

    过期判定用**落盘时写入的指纹**与当前实际指纹比较。落盘档案里可能同时有：
      * dataset_fingerprint    历史字段（只有图片参与），仍然认它以便旧档案继续可用；
      * cache_fingerprint      新字段（图片 + marks.txt + mark_refs.txt + 参考图 +
                               模型版本 + 提示词版本），优先用它。
    """
    for path in (profile_path(folder), legacy_profile_path(folder)):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as stream:
                raw = json.load(stream)
        except (OSError, ValueError) as exc:
            print(f"[profile] 档案损坏，忽略: {path} ({exc})")
            continue
        if allow_stale:
            return normalize_profile(folder, raw)
        recorded_cache = raw.get("cache_fingerprint")
        recorded_dataset = raw.get("dataset_fingerprint")
        if recorded_cache:
            current = cache_fingerprint(folder)
            if recorded_cache != current:
                print(f"[profile] 档案已过期（图片或标志描述文件变化），需要重建: {path}")
                return None
            return normalize_profile(folder, raw)
        if recorded_dataset and recorded_dataset != dataset_fingerprint(folder):
            print(f"[profile] 档案已过期（数据集变化），需要重建: {path}")
            return None
        return normalize_profile(folder, raw)
    return None


def save_profile(folder, profile):
    os.makedirs(config.SEMANTIC_DIR, exist_ok=True)
    path = profile_path(folder)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(profile, stream, ensure_ascii=False, indent=2)
    os.replace(temp, path)
    print(f"[profile] 已保存角色档案: {path}")
    return path


def build_profile(folder, client=None):
    """生成角色档案：人工覆盖优先，其次 VLM 观察；不可用时明确报错，不编造。"""
    override = (config.CHARACTER_PROFILES or {}).get(folder)
    if override:
        profile = normalize_profile(folder, dict(override, source="manual"))
        return save_profile(folder, profile), profile

    observed = _observe_with_vlm(folder, client)
    if observed is None:
        raise RuntimeError(
            f"无法为角色 {folder} 生成语义档案：没有人工覆盖（config.CHARACTER_PROFILES）"
            "且 VLM 不可用。角色档案是自动标注与条件构造的输入，缺失时拒绝臆测。")
    profile = normalize_profile(folder, dict(observed, source="vlm"))
    save_profile(folder, profile)
    return profile_path(folder), profile


def _observe_with_vlm(folder, client):
    """调用既有观察器（character_parser）产出一份观察；VLM 不可用时返回 None。

    不在这里重复实现逐图观察与合并：character_parser.parse_character_with_vlm 已经是
    项目中唯一一份"代表图挑选 + 逐图观察 + 融合"的实现（规格：实现代码只保留一份）。
    """
    if client is None or not getattr(client, "available", False):
        return None
    from dataset import load_captions
    from character.character_parser import parse_character_with_vlm
    folder_dir = os.path.join(config.DATASET_DIR, folder)
    files = dataset_files(folder)
    if not files:
        raise RuntimeError(f"角色 {folder} 的 dataset/{folder}/images/ 下没有图片")
    captions = load_captions(os.path.join(folder_dir, "captions.txt"))
    observed = parse_character_with_vlm(client, folder_dir, captions)
    return observed if isinstance(observed, dict) else None


# ---------------------------------------------------------------------------
# 供条件构造读取的派生视图（不做提示词渲染，那是 subject_transfer 的职责）
# ---------------------------------------------------------------------------
def palette_colors(profile):
    """返回档案里可解析出的 (名称, RGB) 列表；无法解析的颜色被跳过而不是猜。"""
    import re
    named = {
        "white": (245, 245, 245), "black": (25, 25, 28), "grey": (128, 128, 132),
        "gray": (128, 128, 132), "red": (200, 45, 50), "crimson": (180, 30, 55),
        "pink": (240, 150, 180), "orange": (235, 140, 60), "yellow": (235, 205, 90),
        "gold": (208, 170, 80), "green": (80, 160, 90), "teal": (60, 150, 150),
        "cyan": (90, 195, 210), "blue": (70, 110, 200), "navy": (40, 55, 110),
        "purple": (140, 90, 190), "violet": (150, 100, 200), "brown": (130, 90, 60),
        "beige": (225, 210, 185), "silver": (192, 192, 200),
    }
    found = []
    for slot in ("main", "secondary", "accent"):
        text = str((profile.get("color_palette") or {}).get(slot) or "").lower()
        for token in re.split(r"[^a-z]+", text):
            if token in named and (token, named[token]) not in found:
                found.append((token, named[token]))
        for match in re.finditer(r"#([0-9a-f]{6})", text):
            rgb = tuple(int(match.group(1)[i:i + 2], 16) for i in (0, 2, 4))
            found.append((f"#{match.group(1)}", rgb))
    return found


def palette_summary(profile, limit=4) -> str:
    """把档案配色写成提示词片段（例如 "blue, white, pink"）；无颜色返回空串。

    只做"已观察到的颜色"的复述，不新增未观察到的颜色，也不对任何角色写死。
    """
    names = []
    for name, _rgb in palette_colors(profile):
        clean = name.lstrip("#")
        if clean and clean not in names:
            names.append(clean)
    hair = hair_color(profile)
    if hair is not None:
        for name, rgb in _NAMED_COLORS.items():
            if list(rgb) == list(hair) and name not in names:
                names.insert(0, name)
                break
    if not names:
        return ""
    return "colour scheme: " + ", ".join(names[:limit])


def hair_color(profile):
    """从档案的发色描述里解析颜色；解析不出返回 None（调用方应跳过着色，而不是套用别的角色）。

    解析顺序很重要：**先按颜色名解析发色描述本身**，再去档案配色里找。
    反过来的话，配色里排在前面的无关颜色（例如"white"）会先命中，
    把发色错判成那个颜色（实测缺陷：发色描述 "light blue" 被判成 white）。
    """
    import re
    text = str((profile.get("appearance") or {}).get("hair") or "").lower().strip()
    if not text:
        return None
    match = re.search(r"#([0-9a-f]{6})", text)
    if match:
        return [int(match.group(1)[i:i + 2], 16) for i in (0, 2, 4)]

    def score(name):
        """颜色名在发色描述里的匹配强度：整词 > 词内包含；越长的名字越具体。"""
        if re.search(rf"\b{name}\b", text):
            return 2 + len(name)
        if name in text:
            return 1 + len(name)
        return 0

    named = max(_NAMED_COLORS.items(), key=lambda item: score(item[0]))
    if score(named[0]) > 0:
        return list(named[1])
    # 发色描述里没有颜色名时，才退回到档案配色的第一个可解析颜色
    palette = palette_colors(profile)
    return list(palette[0][1]) if palette else None


_NAMED_COLORS = {
    "white": (245, 245, 245), "black": (25, 25, 28), "grey": (128, 128, 132),
    "gray": (128, 128, 132), "red": (200, 45, 50), "pink": (240, 150, 180),
    "orange": (235, 140, 60), "yellow": (235, 205, 90), "gold": (208, 170, 80),
    "green": (80, 160, 90), "teal": (60, 150, 150), "cyan": (90, 195, 210),
    "blue": (70, 110, 200), "navy": (40, 55, 110), "purple": (140, 90, 190),
    "violet": (150, 100, 200), "brown": (130, 90, 60), "beige": (225, 210, 185),
    "silver": (192, 192, 200), "aqua": (120, 210, 215), "turquoise": (70, 190, 180),
}


def describe(profile, limit=10):
    """人类可读摘要，用于日志与验收记录（不是提示词）。"""
    lines = [f"角色档案: {profile.get('folder')}  来源={profile.get('source')}"]
    if profile.get("identity"):
        lines.append(f"  身份: {profile['identity']}")
    palette = profile.get("color_palette") or {}
    lines.append("  配色: " + ", ".join(f"{k}={v or '—'}" for k, v in palette.items()))
    hair = (profile.get("appearance") or {}).get("hair")
    if hair:
        lines.append(f"  发色: {hair}")
    for key in ("visual_features", "signature_accessories", "accessories"):
        items = profile.get(key) or []
        for item in items[:limit]:
            lines.append(f"  [{key}] {item.get('name')}: "
                         f"{item.get('description') or item.get('color') or ''}"[:160])
    for warning in profile.get("warnings") or []:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)
