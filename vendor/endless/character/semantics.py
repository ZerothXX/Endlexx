# -*- coding: utf-8 -*-
"""character/semantics.py —— 角色语义的派生与角色包（唯一实现）

分工（避免"同一件事两处实现"）：
    character/profile.py   观察与档案本身（读/写/校验）
    character/semantics.py 训练与推理使用的派生条件 + 角色包（manifest / 权重登记）
    subject_transfer.py    把这些条件落成生成参数（提示词、结构条件、掩版）

语义只是条件，不替代权重：manifest 里记录的权重文件才是角色学习的产物（方案 §2.6）。
"""
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import config
import model_utils
from character import profile as profile_module


def _trim_text(text, limit):
    """按词边界截断长描述，避免提示词里出现半个单词。"""
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut[:cut.rfind(" ")]
    return cut.rstrip(" ,;:")


def identity_clause(profile, max_chars=140, outfit_chars=34):
    """主（身份）LoRA 的角色条件：角色本身——面部、发色、配色、**服装结构**。

    这是"这个角色是谁"的条件，训练与推理都用它。
    **不包含**配饰、图案、花纹等细粒度装饰：那是概念分支的学习目标（见 concept_clause），
    混进主条件会把"角色身份"和"局部纹样"绑死，既拖慢身份收敛，也让它学不准图案。

    预算与**装配顺序**（实测踩过一次严重回归，务必看完）：
    SDXL 的 CLIP 只有 77 token，推理提示词里这条之后还要放"换成角色的脸"、
    源图媒介（保住原图画风是用户的硬要求）、构图、姿态、质量词，最后才是标志条件，
    因此不能把整条身份段落无脑塞进去。

    但**顺序**比长度更要紧：早期实现按 `parts` 的先后依次塞到 `max_chars` 为止，
    于是把上限从 120 收到 96 时，**第一个被挤掉的正好是 `outfit`（服装结构）**——
    而"服装结构"是角色身份里最显眼的一半。实测后果非常明确：
    用这样训练出来的权重出图，人物变成了"深色头发、看不出是哪个角色"，
    而此前（标注里带着 outfit）的版本能给出正确的蓝发与角色轮廓。

    因此现在的规则是：
      1. **先给服装结构留出额度**（`outfit_chars`），它绝不会被别的条目挤掉；
      2. 配色 / 瞳色 / 发色 / 发型 在剩余额度里按优先级加入
         （发型的文字最长、对"是谁"的贡献最小，因此排在最后、最可能被略去）；
      3. 身份一句话（如 "Samurai girl with silver hair"）只在还有余量时加入，
         且它是**优先级最低**的一项（它提供的信息与 1、2 大量重复）。
    """
    palette = profile.get("color_palette") or {}
    colors = [str(v).strip() for v in palette.values() if str(v).strip()]
    # 优先级：配色 > 瞳色 > 发色 > 发型（越靠后越"可省"）
    appearance = profile.get("appearance") or {}
    ranked = []
    if colors:
        ranked.append(("palette", "color scheme: "
                       + ", ".join(list(dict.fromkeys(colors))[:3])))
    if appearance.get("eyes"):
        ranked.append(("eyes", f"eye color {appearance['eyes']}"))
    if appearance.get("hair"):
        ranked.append(("hair", f"hair color {appearance['hair']}"))
    if appearance.get("hair_style"):
        ranked.append(("hair_style", f"hairstyle {appearance['hair_style']}"))

    # 第一步：为服装结构预留额度（这是实测回归的根因，必须先占）
    outfit = str((profile.get("clothing") or {}).get("style") or "").strip()
    reserved = ""
    if outfit:
        allowed = min(outfit_chars, max(0, max_chars - len("outfit ") - 2))
        if allowed > 0:
            reserved = f"outfit {_trim_text(outfit, allowed)}"

    clause = ""
    for _key, part in ranked:
        candidate = f"{clause}, {part}" if clause else part
        merged = f"{candidate}, {reserved}" if reserved else candidate
        if len(merged) > max_chars:
            continue          # 装不下就跳过这一项，继续试后面的（不终止）
        clause = candidate
    if reserved:
        clause = f"{clause}, {reserved}" if clause else reserved

    identity = str(profile.get("identity") or "").strip()
    if identity:
        part = _trim_text(identity, 40)
        candidate = f"{clause}, {part}" if clause else part
        if len(candidate) <= max_chars:
            clause = candidate
    return clause


def concept_clause(profile, max_chars=220):
    """概念分支的条件：标志性配饰、图案、花纹、边饰等细粒度视觉概念。

    该分支解决的是"主 LoRA 学不细"的部分（用户重申：这些是角色之外还需要学的视觉概念）。
    目前用于：角色包 semantics.json 的登记、后续概念 LoRA 训练与推理时的补充条件。
    """
    parts = []
    for item in (profile.get("signature_accessories") or [])[:4]:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        colour = str(item.get("color") or "").strip()
        parts.append(f"{name} ({colour})" if colour else name)
    rank = {"high": 0, "medium": 1, "low": 2}
    features = sorted(profile.get("visual_features") or [],
                      key=lambda f: rank.get(str(f.get("importance")), 1))
    for feature in features:
        if len(parts) >= 8:
            break
        name = str(feature.get("name") or "").strip()
        if not name:
            continue
        cue = str(feature.get("design_cue") or feature.get("description") or "").strip()
        parts.append(f"{name}: {_trim_text(cue, 50)}" if cue else name)

    clause = ""
    for part in parts:
        candidate = f"{clause}, {part}" if clause else part
        if len(candidate) > max_chars:
            break
        clause = candidate
    return clause


def prompt_clause(profile, max_chars=120):
    """兼容入口：主条件 = 身份条件（历史上本函数混装了配饰与图案，已拆分）。
    默认预算与 identity_clause 保持一致，避免两条入口给出不同长度的条件。"""
    return identity_clause(profile, max_chars=max_chars)


def region_map(profile):
    """{区域: [特征名...]}，供局部修复与评价定位使用（不用于训练标签）。"""
    regions = {}
    for feature in profile.get("visual_features") or []:
        name = feature.get("name")
        if not name:
            continue
        regions.setdefault(feature.get("region") or "whole", []).append(name)
    return regions


def coverage(profile):
    """证据覆盖报告：哪些必要信息缺失（方案 §4.6：证据不足要报告缺口，不臆测）。"""
    gaps = []
    if not (profile.get("appearance") or {}).get("hair"):
        gaps.append("未观察到发色")
    if not any(str(v).strip() for v in (profile.get("color_palette") or {}).values()):
        gaps.append("未观察到角色配色")
    if not (profile.get("clothing") or {}).get("style"):
        gaps.append("未观察到服装结构")
    if not profile.get("visual_features"):
        gaps.append("未观察到图案/线条/边饰等视觉特征")
    if not profile.get("signature_accessories"):
        gaps.append("未观察到独立配饰（无独立配饰的角色应转到服装区域/线条/图案组合特征）")
    return gaps


# ---------------------------------------------------------------------------
# 角色包
# ---------------------------------------------------------------------------
def character_dir(folder):
    return os.path.join(config.CHARACTERS_DIR, folder)


def manifest_path(folder):
    return os.path.join(character_dir(folder), "manifest.json")


def semantics_path(folder):
    return os.path.join(character_dir(folder), "semantics.json")


def load_manifest(folder):
    """读取角色包清单；不存在返回 None（调用方必须显式处理，不静默回退）。"""
    path = manifest_path(folder)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("character") != folder:
        raise ValueError(f"角色包错配：{path} 记录的是 {manifest.get('character')!r}")
    return manifest


def weight_path(folder, role="identity") -> str:
    """该角色指定分支的权重路径（不存在时返回路径本身，由调用方决定是否报错）。"""
    manifest = load_manifest(folder)
    if manifest:
        for entry in manifest.get("weights") or []:
            if entry.get("role") == role:
                return str(_checked_weight_path(folder, entry))
    return os.path.join(character_dir(folder), f"{folder}_{role}.safetensors")


def _checked_weight_path(folder, entry):
    """清单中的权重必须位于本角色包内，不能指向其他角色或外部文件。"""
    relative = entry.get("file") if isinstance(entry, dict) else None
    if not isinstance(relative, str) or not relative:
        raise ValueError("角色包权重缺少文件路径")
    path = (Path(config.CHARACTERS_DIR) / relative).resolve()
    if (path.parent != Path(character_dir(folder)).resolve()
            or path.suffix.lower() != ".safetensors"):
        raise ValueError(f"角色包权重路径不属于角色 {folder}: {relative}")
    return path


def load_semantics(folder):
    path = semantics_path(folder)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def published_profile(folder):
    """推理使用角色包发布时的档案快照，不受后续 dataset 编辑影响。"""
    payload = load_semantics(folder)
    profile = payload.get("profile") if isinstance(payload, dict) else None
    if isinstance(profile, dict) and profile.get("folder") == folder:
        return profile
    return None


def publish(folder, weights, profile, dataset=None, training=None, notes=None) -> dict:
    """把训练产物发布为角色包：权重 + manifest + semantics（方案 §6）。

    weights: [(role, file_path, rank, alpha, target_modules, steps), ...]
    """
    weights = list(weights)
    if not any(role == "identity" for role, *_ in weights):
        raise ValueError("发布角色包必须包含身份权重")
    if not isinstance(profile, dict) or profile.get("folder") != folder:
        raise ValueError("发布角色包必须包含对应角色的语义档案")
    root = Path(character_dir(folder))
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{folder}.stage-", dir=root.parent))
    backup = root.parent / f".{folder}.backup-{uuid.uuid4().hex}"
    entries = []
    try:
        if root.exists():
            shutil.copytree(root, stage, dirs_exist_ok=True)
        for role in ("identity", "concept"):
            (stage / f"{folder}_{role}.safetensors").unlink(missing_ok=True)
        for role, path, rank, alpha, target_modules, steps in weights:
            local = stage / f"{folder}_{role}.safetensors"
            shutil.copy2(path, local)
            entry = model_utils.adapter_manifest(
                role=role, entry=str(local), rank=rank, alpha=alpha,
                target_modules=target_modules, steps=steps,
                dataset_id=(dataset or {}).get("dataset_id"),
                notes=notes)
            entry["file"] = f"{folder}/{local.name}"
            entries.append(entry)
        manifest = {
            "schema": 1,
            "character": folder,
            "trigger": config.get_trigger(folder),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_model": os.path.relpath(config.BASE_MODEL_PATH, config.PROJECT_ROOT).replace("\\", "/"),
            "base_model_sha256_head": None,
            "weights": entries,
            "dataset": dataset or {},
            "training": training or {},
            "compatibility": {
                "base_model": "sdxl",
                "requires_same_base": True,
                "note": "所有权重必须与 manifest.base_model 同一底模训练，跨底模不可加载",
            },
            "notes": notes or "",
        }
        with (stage / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        semantics = {
            "character": folder,
            "trigger": config.get_trigger(folder),
            "profile": profile,
            # 两条条件分属两个分支，互不混装（用户重申）：
            #   identity -> 主 LoRA 学"角色本身"（面部/发色/配色/服装）
            #   concept  -> 概念 LoRA 学"配饰/图案/花纹/边饰"等细粒度视觉概念
            "identity_clause": identity_clause(profile),
            "concept_clause": concept_clause(profile),
            "regions": region_map(profile),
            "coverage_gaps": coverage(profile),
            "note": "语义用于条件构造与检查，不替代权重（方案 §6）",
        }
        with (stage / "semantics.json").open("w", encoding="utf-8") as stream:
            json.dump(semantics, stream, ensure_ascii=False, indent=2)
        had_previous = root.exists()
        if had_previous:
            os.replace(root, backup)
        try:
            os.replace(stage, root)
        except Exception:
            if had_previous:
                try:
                    os.replace(backup, root)
                except Exception as rollback_error:
                    raise RuntimeError(f"发布失败且回滚失败，原角色包保留在 {backup}") from rollback_error
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    print(f"[pack] 角色包已发布: {root}")
    for entry in entries:
        print(f"[pack]   权重 {entry['role']}: {entry['file']} "
              f"({entry['bytes'] / 1e6:.1f}MB, rank={entry['rank']}, {entry['trained_steps']} 步)")
    return manifest


def select_references(folder, limit=None):
    """从该角色自己的图片里挑选参考增强用图（按自动标注的景别与视角 + 统一标志证据）。

    规则（不看角色编号、不用人工卡片）：
      * 脸部槽：优先"面部特写"，其次"上半身 / 正面视角"；
      * 服装槽：优先"全身 / 上半身"，避开各种 close-up（局部图会把贴图感带进来）；
      * 单独标志槽：用户给的**标志参考图裁剪**（`output/marks/<角色>/ref_*.png`）——
        它是"这个标志长什么样"最直接的证据，规格 §3 明确要求保留真实参考图在推理时使用；
      * 缺标注的图片不参与挑选——没有依据就不选，而不是随便拿一张。

    返回 [{"slot","file","rel_path","tags","path"}]，可能少于 limit 项。
    """
    import dataset as dataset_module
    limit = config.REFERENCE_MAX if limit is None else limit
    captions = dataset_module.load_character_captions(folder)
    rows = []
    for path in dataset_module.scan_images(folder):
        name = os.path.basename(path)
        tags = [str(t).lower() for t in captions.get(name, [])]
        if not tags:
            continue
        rows.append({"file": name, "rel_path": name, "tags": tags, "text": ", ".join(tags)})

    def score_face(row):
        text = row["text"]
        points = 0
        points += 100 if "face close-up" in text else 0
        points += 40 if "upper body" in text else 0
        points += 20 if "front view" in text else 0
        points += 10 if "three-quarter view" in text else 0
        points -= 60 if "back view" in text else 0
        points -= 40 if "pattern close-up" in text or "accessory close-up" in text else 0
        return -points

    def score_costume(row):
        text = row["text"]
        points = 0
        points += 80 if "full body" in text else 0
        points += 50 if "upper body" in text else 0
        points += 20 if "front view" in text else 0
        points -= 80 if "close-up" in text else 0
        points -= 40 if "back view" in text else 0
        return -points

    chosen = []
    for slot, scorer in (("face", score_face), ("costume", score_costume)):
        ranked = sorted(rows, key=scorer)
        pick = next((row for row in ranked
                     if row["file"] not in {c["file"] for c in chosen}
                     and scorer(row) < 0), None)
        if pick is not None:
            chosen.append({"slot": slot, "file": pick["file"],
                           "rel_path": pick["rel_path"], "tags": pick["tags"]})
        if len(chosen) >= limit:
            break

    # 标志参考槽：优先用统一标志证据层落盘的真实裁剪
    marks_module_local = None
    try:
        from character import marks as marks_module_local
    except Exception:
        marks_module_local = None
    if marks_module_local is not None and len(chosen) < limit:
        for mark in (marks_module_local.load_library(folder) or {}).get("marks") or []:
            if len(chosen) >= limit:
                break
            for evidence in mark.get("evidence") or []:
                if evidence.get("kind") != "reference_image" or not evidence.get("crop_path"):
                    continue
                absolute = None
                try:
                    from vision import mark_evidence
                    absolute = mark_evidence.resolve_output_path(evidence["crop_path"])
                except Exception:
                    absolute = None
                if absolute is None:
                    candidate = os.path.join(config.PROJECT_ROOT, evidence["crop_path"])
                    absolute = candidate if os.path.isfile(candidate) else None
                if absolute is None:
                    continue
                chosen.append({
                    "slot": "mark",
                    "file": os.path.basename(evidence["crop_path"]),
                    "rel_path": evidence["crop_path"],
                    "tags": [str(mark.get("mark_class") or "mark"), str(mark.get("name") or "")],
                    "path": absolute,
                    "mark_id": mark.get("mark_id"),
                    "note": "单独标志参考图：用户提供的证据配对",
                })
                break
    return chosen[:limit]


def audit_switch(folder) -> dict:
    """角色切换审计：返回本次运行实际会使用的权重与语义来源（方案 §11.7）。"""
    manifest = load_manifest(folder)
    if manifest is None:
        return {"character": folder, "weights": [], "semantics": None,
                "ready": False, "reason": "角色包不存在（尚未训练）"}
    semantics = load_semantics(folder)
    entries = manifest.get("weights")
    if not isinstance(entries, list):
        entries = []
    missing = []
    if not any(isinstance(e, dict) and e.get("role") == "identity" for e in entries):
        missing.append("identity")
    weights = []
    seen_roles = set()
    for entry in entries:
        try:
            path = _checked_weight_path(folder, entry)
            if entry["role"] in seen_roles:
                raise ValueError(f"重复的权重角色: {entry['role']}")
            seen_roles.add(entry["role"])
            expected_bytes = entry.get("bytes")
            if (not path.is_file() or path.stat().st_size == 0
                    or (isinstance(expected_bytes, int)
                        and path.stat().st_size != expected_bytes)):
                missing.append(str(entry.get("file")))
            weights.append({"role": entry["role"], "file": entry["file"],
                            "scale_role": entry["role"]})
        except (KeyError, ValueError, TypeError) as exc:
            missing.append(str(exc))
    valid_semantics = (isinstance(semantics, dict)
                       and semantics.get("character") == folder
                       and isinstance(semantics.get("profile"), dict)
                       and semantics["profile"].get("folder") == folder)
    if not valid_semantics:
        missing.append("semantics.json")
    return {
        "character": folder,
        "trigger": manifest.get("trigger"),
        "base_model": manifest.get("base_model"),
        "weights": weights,
        "semantics": semantics_path(folder) if valid_semantics else None,
        "ready": not missing,
        "missing": missing,
        "reason": "" if not missing else "角色包不完整或权重文件异常",
    }
