# -*- coding: utf-8 -*-
"""
character/character_parser.py —— 角色图片语义解析

- pick_representative_images : 整体和脸部建立身份，再覆盖所有配饰特写
  （无可用标签时取 001/中间/最后）
- parse_character_with_vlm    : 代表图逐张过 VLM，合并为规格 §6.2 结构
- parse_character_offline     : 无 VLM 时的兜底结构（字段齐全、值保守）

依赖：config / dataset.load_captions, scan_images / image_utils / vlm.vlm_utils。
只读数据集，不修改原始数据（规格 §5.2 / 禁止 9）。
"""

import os
import sys

# 保证从任意位置直接运行本文件时都能 import 到项目根目录的模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from dataset import load_captions, scan_images  # noqa: E402
import image_utils  # noqa: E402
from PIL import Image  # noqa: E402  用于小尺寸特写放大（四叶发饰等，见 parse_character_with_vlm）
from vlm.vlm_utils import CHARACTER_PROMPT  # noqa: E402


def pick_representative_images(image_files: list, captions: dict) -> list:
    """整体图优先建立身份，其次脸部，再覆盖所有标志/配饰参考。

    规格 §2 第 7 行：代表图**只用于初始摘要**（身份、画风这类全局字段），
    不再承担"全部观察记录"的职责——逐图观察由 `observe_all_images` 负责。

    不限制最多三张，避免遗漏第二个配饰；独立配饰不作为人物身份基底。
    captions 为空 / 没有可用标签时：取第 1 张 / 中间 / 最后 1 张，保证代表性。
    结果去重、按挑选优先级返回。
    """
    if not image_files:
        return []

    def _has_tag(path: str, keyword: str) -> bool:
        tags = [t.lower() for t in captions.get(os.path.basename(path), [])]
        return any(keyword in t for t in tags)

    face_files = [p for p in image_files if _has_tag(p, "face close-up")]
    # 标志配饰：发饰/配饰特写优先（"四叶/leaf"是历史遗留关键词，只在标签里真出现时才命中）
    sig_acc = [p for p in image_files
               if (_has_tag(p, "hair") or _has_tag(p, "four")
                   or _has_tag(p, "leaf") or _has_tag(p, "signature"))]
    acc_files = [p for p in image_files if _has_tag(p, "accessory")]
    feature_files = [p for p in image_files if any(_has_tag(p, tag) for tag in
                     ("clothing close-up", "pattern close-up", "feet close-up"))]
    for tag in ("upper body", "back view"):
        match = next((p for p in image_files if _has_tag(p, tag)), None)
        if match and match not in feature_files:
            feature_files.append(match)

    picks = []
    # 整体/人脸提供身份；所有配饰特写逐一解析，不能只取首张而遗漏后面的配饰。
    whole = [p for p in image_files if _has_tag(p, "full body")]
    if whole:
        picks.append(whole[0])
    # 2. 人脸特写
    if face_files and face_files[0] not in picks:
        picks.append(face_files[0])
    for p in sig_acc + acc_files + feature_files:
        if p not in picks:
            picks.append(p)
    # 3. 无标注时补足整体参考
    rest = [p for p in image_files if p not in picks]
    if rest and len(picks) < 3:
        picks.append(rest[0])
    # 若标签缺失（含 captions 为空）导致不足 3 张，补 001/中间/最后
    if len(picks) < 3:
        for idx in (0, len(image_files) // 2, len(image_files) - 1):
            if 0 <= idx < len(image_files) and image_files[idx] not in picks:
                picks.append(image_files[idx])
            if len(picks) >= 3:
                break

    # 全文覆盖：身份摘要之外**每一张有效原图**都建立一条观察记录（规格 §2 第 7 行）。
    # 代表图排在前面（它们承担身份/画风字段），其余图只补充"这张图上看到了什么"，
    # 因此不会因为代表图选得不好而丢掉整张图的证据。
    observed = list(picks)
    for path in image_files:
        if path not in observed:
            observed.append(path)
    print(f"[character_parser] 代表图挑选: {[os.path.basename(p) for p in picks]}；"
          f"逐图观察覆盖 {len(observed)}/{len(image_files)} 张")
    return observed


def _merge_character(base: dict, extra: dict) -> None:
    """把 extra 的 §6.2 字段合并进 base（首张为主，extra 只做补充）。

    - 标量/字符串字段：base 为空时由 extra 填充
    - 列表字段（color_palette / dominant_colors）：按值去重追加
    - signature_accessories：按 name 去重追加（补充首张缺失的配饰）
    """
    if not isinstance(extra, dict):
        return
    for key, val in extra.items():
        if isinstance(val, dict):
            bd = base.setdefault(key, {})
            if not isinstance(bd, dict):
                bd = {}
                base[key] = bd
            for k2, v2 in val.items():
                if (not bd.get(k2) or bd.get(k2) == "unknown") and v2:
                    bd[k2] = v2
        elif isinstance(val, list):
            if key in {"signature_accessories", "visual_features"}:
                # 按 name 去重，只追加 base 中不存在的配饰
                names = {a.get("name") for a in base.get(key, []) if isinstance(a, dict)}
                for a in val:
                    if isinstance(a, dict) and a.get("name") and a["name"] not in names:
                        base.setdefault(key, []).append(a)
                        names.add(a["name"])
                    elif isinstance(a, dict) and a.get("name"):
                        previous = next(x for x in base[key] if x.get("name") == a["name"])
                        if a.get("isolated_reference"):
                            previous["isolated_reference"] = True
                            previous["source_images"] = list(dict.fromkeys(
                                a.get("source_images", []) + previous.get("source_images", [])))
                        for image in a.get("source_images", []):
                            if image not in previous.setdefault("source_images", []):
                                previous["source_images"].append(image)
            else:
                base_list = base.setdefault(key, [])
                for item in val:
                    if item not in base_list:
                        base_list.append(item)
        else:
            if (not base.get(key) or base.get(key) == "unknown") and val:
                base[key] = val


def normalize_character_result(raw):
    """Validate model output before merging; schema validity is not visual accuracy."""
    warnings = []
    if not isinstance(raw, dict):
        return {}, ["response must be a JSON object"]

    def text(value):
        if not isinstance(value, str):
            return ""
        value = value.strip()
        if (not value or "|" in value or value.lower() in {
                "...", "unknown", "none", "n/a", "not visible", "short name", "short object name", "visible geometry",
                "visible colors", "brief visible design", "visible appearance",
                "non-anatomical colors, shapes, textures or panel arrangement"}
                or value.lower().startswith(("4 to 10 words", "short colors, pattern"))):
            return ""
        return value

    result = {}
    for key in ("character_identity", "visual_style"):
        if text(raw.get(key)):
            result[key] = text(raw[key])
    for key, fields in (("hair", ("color", "style")),
                        ("face", ("eye_color", "face_features")),
                        ("clothing", ("description",))):
        value = raw.get(key)
        if value is not None and not isinstance(value, dict):
            warnings.append(f"{key}: expected object")
        if isinstance(value, dict):
            result[key] = {k: text(value[k]) for k in fields if text(value.get(k))}
    colors = raw.get("color_palette", [])
    if not isinstance(colors, list):
        warnings.append("color_palette: expected list")
        colors = []
    result["color_palette"] = list(dict.fromkeys(text(c) for c in colors if text(c)))
    aliases = {"upper body": "torso", "sleeves": "arms", "arm": "arms",
               "leg": "legs", "foot": "feet", "hands": "arms", "right hand": "arms",
               "left hand": "arms"}
    regions = {"head", "chest", "arms", "waist", "legs", "feet", "torso"}
    for key in ("visual_features", "signature_accessories"):
        result[key] = []
        values = raw.get(key, [])
        if not isinstance(values, list):
            warnings.append(f"{key}: expected list")
            continue
        for index, value in enumerate(values):
            label = f"{key}[{index}]"
            if not isinstance(value, dict) or not text(value.get("name")):
                warnings.append(f"{label}: missing or placeholder name")
                continue
            item = {k: text(value[k]) for k in (
                "name", "description", "shape", "color", "location", "design_cue")
                    if text(value.get(k))}
            if key == "visual_features":
                region = text(value.get("region")).lower()
                region = aliases.get(region, region)
                if region not in regions or not item.get("description"):
                    warnings.append(f"{label}: invalid region or description")
                    continue
                item["region"] = region
                kind = text(value.get("kind")).lower()
                if kind in {"pattern", "garment", "accessory", "material"}:
                    item["kind"] = kind
                else:
                    warnings.append(f"{label}: ignored invalid kind")
                item["importance"] = (value.get("importance") if value.get("importance")
                                      in ("high", "medium", "low") else "medium")
            else:
                role = value.get("transfer_role")
                item["transfer_role"] = role if role in ("emblem", "wearable", "prop") else "unknown"
                item["semantic_importance"] = (value.get("semantic_importance")
                    if value.get("semantic_importance") in ("high", "medium", "low") else "medium")
            result[key].append(item)
    if not any(result.values()):
        return {}, warnings + ["no usable semantic fields"]
    return result, warnings


def parse_character_with_vlm(client, folder_dir: str, captions: dict) -> dict:
    """对代表图逐张调用 client.analyze_image(img, CHARACTER_PROMPT)，合并返回 §6.2 结构。

    - 代表图由 pick_representative_images 挑选（包含全部配饰特写）
    - 单张图片分析失败只打印警告并跳过（不中断整体流程）
    - 全部失败 / client 不可用：回退 parse_character_offline 兜底结构
    """
    if client is None or not getattr(client, "available", False):
        print("[character_parser] 告警: VLM 不可用，回退 offline 解析")
        return parse_character_offline(captions)

    image_files = scan_images(folder_dir)  # 目录缺失/为空会抛异常（fail fast）
    picks = pick_representative_images(image_files, captions)

    results = []
    face_results = []
    analysis_warnings = []
    parsed_sources = []
    for path in picks:
        tags = captions.get(os.path.basename(path), [])
        r, warnings = _observe_one(client, path, tags)
        analysis_warnings.extend(
            {"file": os.path.basename(path), "reason": w} for w in warnings)
        if not r:
            print(f"[character_parser] 告警: {os.path.basename(path)} 未返回有效语义，跳过")
            continue
        r = _postprocess(r, tags)
        if _is_face_closeup(tags):
            face_results.append(r)
        for key in ("signature_accessories", "visual_features"):
            for feature in r.get(key) or []:
                if isinstance(feature, dict):
                    feature["source_images"] = [os.path.basename(path)]
                    feature["isolated_reference"] = _is_isolated_detail(tags)
        results.append(r)
        parsed_sources.append(os.path.basename(path))
        print(f"[character_parser] Parsed {os.path.basename(path)}", flush=True)

    if not results:
        print("[character_parser] 告警: 所有代表图分析均失败，返回 offline 兜底结构")
        fallback = parse_character_offline(captions)
        fallback["analysis_warnings"] = analysis_warnings
        return fallback

    # 首张为主，其余只补充缺失字段/配饰
    base = dict(results[0])
    for extra in results[1:]:
        _merge_character(base, extra)
    # A face close-up carries more evidence for eyes/hair than a tiny full-body face.
    for parsed in face_results:
        for key in ("hair", "face"):
            for field, value in (parsed.get(key) or {}).items():
                if isinstance(value, str) and value.strip().lower() not in {"", "unknown"}:
                    base.setdefault(key, {})[field] = value
    base["source_images"] = parsed_sources
    base["analysis_warnings"] = analysis_warnings
    base["reference_images"] = {}
    for slot, keywords in (("portrait", ("face close-up",)),
                           ("costume", ("upper body", "full body"))):
        name = next((os.path.basename(p) for p in image_files
                     if any(k in " ".join(captions.get(os.path.basename(p), [])).lower()
                            for k in keywords)), None)
        if name:
            base["reference_images"][slot] = {"file": name}
            if slot == "costume":
                details = [os.path.basename(p) for p in image_files if any(
                    k in " ".join(captions.get(os.path.basename(p), [])).lower()
                    for k in ("clothing close-up", "feet close-up", "pattern close-up"))]
                if details:
                    base["reference_images"][slot]["files"] = [name] + details[:3]

    # ---- 逐图观察记录：**全部有效原图**，不只代表图（规格 §2 第 7 行）----
    # 代表图已经有完整观察结果，这里直接复用，不重复问模型（省显存时间）。
    observed = {}
    for path, parsed in zip(parsed_sources, results):
        observed[path] = {"source": "vlm_representative", "tags": [
            str(t) for t in (captions.get(path) or [])], "features": _feature_names(parsed),
            "accessories": _accessory_names(parsed)}
    for path in image_files:
        name = os.path.basename(path)
        if name in observed:
            continue
        tags = captions.get(name, [])
        # 代表图之外的图只做**轻量观察**（一个短提示词），失败就记缺口：
        # 它们的职责是"这张图上看到了什么"，不是重建身份。
        record = {"source": "lightweight_vlm", "tags": [str(t) for t in tags],
                  "features": [], "accessories": []}
        try:
            img = image_utils.load_image(path)
            img.thumbnail((768, 768))
            r = client.analyze_image(img, LIGHT_OBSERVE_PROMPT, temperature=0,
                                     max_new_tokens=256)
            if isinstance(r, dict):
                record["note"] = str(r.get("note") or "").strip()
                positions = r.get("mark_positions")
                if isinstance(positions, list):
                    record["mark_positions"] = [str(p) for p in positions if str(p).strip()]
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            analysis_warnings.append({"file": name, "reason": record["error"]})
        observed[name] = record
        print(f"[character_parser] 逐图观察 {name}: {record.get('note') or record.get('error') or '—'}",
              flush=True)
    base["image_observations"] = observed
    print(f"[character_parser] VLM 角色解析完成（融合 {len(results)} 张代表图；"
          f"逐图观察 {len(observed)}/{len(image_files)} 张）")
    return base


# 轻量逐图观察：只问"这张图上有什么标志、在哪"，不重建身份。
LIGHT_OBSERVE_PROMPT = (
    "Look at this image of one character design. Return ONLY JSON:\n"
    '{"note":"one short sentence about what character detail this image actually shows",'
    '"mark_positions":["a character mark and where it sits in THIS image"]}\n'
    "Report only marks you can actually see here. Use an empty list when you see none. "
    "Do not describe the background, the pose, or anything outside the character design."
)


def _is_isolated_detail(tags) -> bool:
    return any("accessory close-up" in str(t).lower() or "pattern close-up" in str(t).lower()
               for t in tags or [])


def _is_face_closeup(tags) -> bool:
    return any("face close-up" in str(t).lower() for t in tags or [])


def _feature_names(parsed) -> list:
    return [str(f.get("name")) for f in (parsed or {}).get("visual_features") or []
            if isinstance(f, dict) and f.get("name")]


def _accessory_names(parsed) -> list:
    return [str(a.get("name")) for a in (parsed or {}).get("signature_accessories") or []
            if isinstance(a, dict) and a.get("name")]


def _observe_one(client, path, tags):
    """对单张图跑一次观察，返回 (规范化结果, 警告)。异常只记警告不中断。"""
    warnings = []
    try:
        garment_detail = any(str(t).lower() in {"feet close-up", "clothing close-up"}
                             for t in tags)
        img = image_utils.load_image(path)
        limit = 640 if garment_detail else 1024
        img.thumbnail((limit, limit))
        # 特写图可能很小（如发饰 114x116），直接送 VLM 细节不可辨，
        # 放大到短边 512 再分析（仅内存副本，不落盘、不改数据集，规格 §禁止9）
        if max(img.size) < 512:
            scale = 512 / max(img.size)
            img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)),
                             Image.LANCZOS)
        detail = _is_isolated_detail(tags)
        prompt = CHARACTER_PROMPT
        if garment_detail:
            prompt = ('Look closely at the visible body part and fabric. Return ONLY JSON with string fields '
                      'body_part and description. Describe visible colors, shapes and garment details in one short sentence. '
                      'Do not describe anything outside the image.')
        if detail:
            prompt = ('Describe this isolated object or pattern, using only visible evidence. '
                      'Return ONLY JSON {"name":"short object name", "shape":"visible geometry", '
                      '"color":"visible colors", "description":"brief visible design", '
                      '"transfer_role":"emblem or wearable or prop"}. '
                      'A hand-held object is a prop. A flat decorative graphic is an emblem. '
                      'Do not invent a wearer, clothing, body parts, or hidden placement.')
        r = client.analyze_image(img, prompt, temperature=0,
                                 max_new_tokens=160 if garment_detail else 1024)
        if not r:
            return {}, warnings
        if garment_detail and isinstance(r.get("body_part"), str):
            part = r["body_part"].strip().lower()
            r = {"visual_features": [{"name": part + " detail", "region": part,
                  "kind": "garment", "description": r.get("description"),
                  "importance": "high"}]}
        if detail:
            if not isinstance(r.get("name"), str) or not r["name"].strip():
                return {}, ["Isolated detail did not return a visible object description"]
            r = {"signature_accessories": [dict(r, location="unknown",
                                               semantic_importance="high")],
                 "visual_features": []}
        r, warnings = normalize_character_result(r)
        return r, warnings
    except Exception as exc:
        print(f"[character_parser] 告警: {path} 分析失败，跳过: {type(exc).__name__}: {exc}")
        return {}, [f"{type(exc).__name__}: {exc}"]


def _postprocess(r, tags):
    """按这张图的景别裁掉它**不可能**提供的字段（局部图不能给身份）。"""
    if _is_face_closeup(tags):
        r.pop("clothing", None)
        r["visual_features"] = [f for f in r.get("visual_features", [])
                                if isinstance(f, dict) and f.get("region") == "head"]
    elif any(str(t).lower() in {"clothing close-up", "feet close-up"} for t in tags):
        # 局部衣物图不能提供发色/瞳色身份，也不能提供头部配饰
        r.pop("hair", None)
        r.pop("face", None)
        r.pop("character_identity", None)
        r.pop("color_palette", None)
        r["signature_accessories"] = []
        if any(str(t).lower() == "feet close-up" for t in tags):
            r["visual_features"] = [f for f in r.get("visual_features", [])
                                    if isinstance(f, dict)
                                    and f.get("region") in {"feet", "legs"}]
    return r


def parse_character_offline(captions: dict) -> dict:
    """无 VLM 时的 §6.2 兜底结构：字段齐全、值保守（unknown / 空列表）。"""
    return {
        "character_identity": "unknown",
        "visual_style": "unknown",
        "color_palette": [],
        "hair": {"color": "unknown", "style": "unknown"},
        "face": {"eye_color": "unknown", "face_features": "unknown"},
        "clothing": {"description": "unknown", "dominant_colors": []},
        "signature_accessories": [],
        "note": "offline fallback (no VLM)",
    }


if __name__ == "__main__":
    """自检块（PyCharm 直接运行）：不加载 VLM，仅验证代表图挑选与 offline 兜底结构。"""
    print("=" * 60)
    print("character/character_parser.py 自检")
    print("=" * 60)

    folder_dir = os.path.join(config.DATASET_DIR, config.CHARACTER_ID)
    cap_file = os.path.join(folder_dir, "captions.txt")

    # 1. 代表图挑选（VLM 路径的前置步骤）
    imgs = scan_images(folder_dir)
    caps = load_captions(cap_file)
    print(f"[1] 图片共 {len(imgs)} 张，captions 共 {len(caps)} 条")
    picks = pick_representative_images(imgs, caps)
    print(f"    挑选结果: {[os.path.basename(p) for p in picks]}")

    # 2. 空 captions 场景（应回退 001/中间/最后）
    picks_no_cap = pick_representative_images(imgs, {})
    print(f"[2] 空 captions 挑选: {[os.path.basename(p) for p in picks_no_cap]}")

    # 3. offline 兜底结构
    offline = parse_character_offline(caps)
    print(f"[3] offline 兜底结构字段: {sorted(offline.keys())}")
    print(f"    note = {offline['note']!r}")

    print("=" * 60)
