# -*- coding: utf-8 -*-
"""subject_transfer.py —— 角色视觉语义迁移流程（唯一实现）

把"源图 + 角色包"变成一次受控生成所需的全部条件：

    保留什么（源图）  形态、比例、动作、朝向、构图、穿戴关系、表现媒介（画风）
    迁移什么（角色）  配色、发色、服装结构、面部身份、标志性配饰、图案与线条

用户目标为"极致 cosplay"：以还原角色为准，因此**允许并追求换脸**；
源图保留的是形态与画风，不是脸。

本模块只做条件构造与决策，不加载模型、不保存图片；实际生成由 test.py 调用。
"""
import math
import os

import config
import image_utils
import text_budget
from character import cards as cards_module
from character import marks as marks_module
from character import profile as profile_module
from character import semantics
from character.transfer_constraints import composition_constraints, fit_prompt, region_visible

# 源图表现媒介 -> 必须保住的特征（方案 §7：像素图保留像素组织，线稿保留线条与疏密）
MEDIUM_TERMS = {
    "pixel art": "pixel art, pixelated, limited palette",
    "line art": "line art, black outlines, no shading",
    "sketch": "pencil sketch, visible strokes",
    "watercolor": "watercolor, soft pigment bleed",
    "oil painting": "oil painting, visible brush strokes",
    "painting": "painting, visible brush strokes",
    "3d render": "3d render, cgi",
    "photograph": "photograph, photorealistic",
    "anime": "anime illustration, cel shading",
    "illustration": "illustration",
}
# 与源媒介冲突、必须抑制的词（只有检测到该媒介时才加入）
MEDIUM_NEGATIVES = {
    "pixel art": "smooth gradients, anti-aliasing, photorealistic, 3d render",
    "line art": "shading, greyscale gradient, painting, 3d render, photograph",
    "sketch": "full color render, photograph, 3d render",
    "watercolor": "hard cel shading, 3d render, photograph",
    "oil painting": "anime, cel shading, 3d render, photograph",
    "photograph": "anime, cartoon, cel shading, illustration, line art",
    "3d render": "anime, cel shading, 2d flat illustration, sketch",
    "anime": "photograph, 3d render",
}


def medium_terms(style: str) -> str:
    """把观察到的媒介描述映射为提示词片段；无法识别时返回空串（不硬加画风）。"""
    text = str(style or "").lower()
    for key in ("pixel", "line art", "sketch", "watercolor", "oil", "3d", "render",
                "photo", "realistic", "anime", "illustration"):
        if key == "oil" and "oil" in text:
            return MEDIUM_TERMS["oil painting"]
        if key in text:
            mapped = {"pixel": "pixel art", "3d": "3d render", "render": "3d render",
                      "photo": "photograph", "realistic": "photograph"}.get(key, key)
            if mapped in MEDIUM_TERMS:
                return MEDIUM_TERMS[mapped]
    return ""


def medium_key(style: str) -> str:
    text = str(style or "").lower()
    for key in ("pixel", "line art", "sketch", "watercolor", "oil", "3d", "render",
                "photo", "realistic", "anime", "illustration"):
        if key in text:
            return {"pixel": "pixel art", "3d": "3d render", "render": "3d render",
                    "photo": "photograph", "realistic": "photograph"}.get(key, key)
    return ""


def generation_settings(subject_type: str, source=None, slot: str = "") -> dict:
    """按主体类型给出重绘强度 / 结构权重 / 权重强度；数值全部来自 config。

    source 给定时，若判定为线条密集型输入（线稿/草图/像素画），
    改用一套更保守的参数：强度更低、结构条件更高，保住"内容就是线条"的输入。

    ## `lora_scale` 到底调的是哪个适配器（这是一个**修过的静默 bug**）

    `lora_scale` 历史上是按主体类型从 `HUMAN/ANIMAL/OBJECT_LORA_SCALE` 取的，
    而调用方一律把它写给**身份**适配器（`adapter_name` 缺省 = "default"）。
    但 config 里那几个值的注释写的是"概念权重强度"（`OBJECT_SURFACE_LORA_SCALE=1.45`、
    `ATTACH_LORA_SCALE=1.60`），非人物分支还专门把身份权重设为 0
    （`NON_HUMAN_IDENTITY_SCALE=0.0`，"与概念权重叠加会糊成一团"）。
    于是实际效果是：**身份权重被 0.85 打开、概念权重恒为 0.55**，与设计意图正好相反。

    现在把"调哪个适配器、调多大"写成显式字段：
      * `scale_target`：这一遍渲染由哪个适配器承担（人物 `default`，非人物 `concept`）；
      * `lora_scale`  ：`scale_target` 那个适配器的强度；
      * `identity_scale`：身份适配器的强度（非人物为 0）；
      * `concept_scale` ：概念适配器的强度。
    调用方据此一次提交两个适配器，不再让"调了哪个"取决于缺省参数。
    """
    prefix = {"human": "HUMAN", "object": "OBJECT", "animal": "ANIMAL"}.get(subject_type)
    if prefix:
        strength = getattr(config, f"{prefix}_IMG2IMG_STRENGTH")
        control = getattr(config, f"{prefix}_CONTROLNET_SCALE")
        end = getattr(config, f"{prefix}_CONTROL_END")
    else:
        strength = config.IMG2IMG_STRENGTH
        control = config.CONTROLNET_SCALE
        end = config.CONTROL_GUIDANCE_END
    lineart = bool(source is not None and is_lineart_like(source))
    if lineart:
        strength = min(strength, config.LINEART_IMG2IMG_STRENGTH)
        control = max(control, config.LINEART_CONTROLNET_SCALE)

    identity_scale = float(adapter_scales_for(subject_type)["identity"])
    concept_scale = float(adapter_scales_for(subject_type)["concept"])
    if subject_type in ("animal", "object"):
        # 非人物的"加法"由概念权重承担；不同场合的强度是不同的配置项：
        #   * 表面/纹样承印走 OBJECT_SURFACE_LORA_SCALE；
        #   * 附着配饰（颈圈/胸饰）走 ATTACH_LORA_SCALE（局部加，可以更高）。
        target = "concept"
        if slot == "surface":
            scale = float(getattr(config, "OBJECT_SURFACE_LORA_SCALE", concept_scale))
        elif slot == "attachment":
            scale = float(getattr(config, "ATTACH_LORA_SCALE", concept_scale))
        else:
            scale = concept_scale
        concept_scale = scale
    else:
        # **人物**（以及主体类型未知的保守分支）：这一遍由身份权重承担，
        # 概念权重按配置（人物默认为 0）。
        target = "default"
        scale = identity_scale
        if subject_type not in ("human",):
            # 主体类型未知：强度取两侧的较小值，概念权重完全不参与
            # （宁可少做加法，也不拿没验证过的概念强度去改未知主体）。
            scale = min(float(config.LORA_SCALE), float(config.ANIMAL_LORA_SCALE))
            concept_scale = 0.0
    if (not all(math.isfinite(v) for v in (strength, control, end, scale, identity_scale))
            or not 0 < strength <= 1 or not 0 < end <= 1 or control < 0
            or scale < 0 or identity_scale < 0):
        raise ValueError(f"生成参数非法: strength={strength} control={control} "
                         f"end={end} scale={scale} identity={identity_scale}")
    if target == "default":
        identity_scale = scale
    return {"strength": strength, "control_scale": control, "control_end": end,
            "lora_scale": scale, "lineart_like": lineart,
            "scale_target": target,
            "identity_scale": identity_scale,
            "concept_scale": concept_scale}


def adapter_scales_for(subject_type) -> dict:
    """按主体类型给出 (身份强度, 概念强度) 的**唯一**决定处。

    为什么集中在一处：这两个值此前分散在 `test.py` 的多处调用点上，
    实测出现过"先设概念 0.55、再把身份设 0、随后渲染又把身份设回 0.85"
    这种互相覆盖的顺序问题（`_render` 的缺省适配器名正是身份那一支）。
    现在只在这里决定，调用方照抄，避免再次错位。
    """
    kind = str(subject_type or "unknown")
    if kind in ("animal", "object"):
        return {"identity": float(getattr(config, "NON_HUMAN_IDENTITY_SCALE", 0.0)),
                "concept": float(getattr(config, "CONCEPT_NON_HUMAN_SCALE", 0.0))}
    if kind == "human":
        return {"identity": float(getattr(config, "HUMAN_LORA_SCALE", 0.0)),
                "concept": float(getattr(config, "CONCEPT_HUMAN_SCALE", 0.0))}
    return {"identity": min(float(getattr(config, "LORA_SCALE", 0.0)),
                            float(getattr(config, "ANIMAL_LORA_SCALE", 0.0))),
            "concept": 0.0}




def _negative_pieces(text):
    """把一条负向词串拆成可按优先级逐个取舍的小条目。

    为什么必须拆（实测踩过）：`ANIMAL_NEGATIVE_PROMPT` 是一条以逗号分隔的长串
    （"anthropomorphic, humanoid body, human clothing, ..."）。整串当一个原子块时，
    预算只要装不下它就**整条丢掉**，于是最关键的 "anthropomorphic" 也没了——
    而那正是非人物分支最不能丢的一条（"不许画成拟人"）。
    拆开之后按顺序逐个加入，能装几个装几个。
    """
    return [piece.strip() for piece in str(text or "").split(",") if piece.strip()]


def negative_prompt(subject) -> str:
    """按主体类型组合负向词；非人物必须显式抑制拟人化。

    **顺序即优先级**（`negative_prompt_with_style` 会按这个顺序裁到预算内）：
      1. 非人物：不许变成人（`NON_HUMAN` / `ANIMAL` / `OBJECT`）——这是硬约束；
      2. 通用"不许画坏"（`NEGATIVE_PROMPT`）；
      3. 构图覆盖类负向词。
    实测踩过：早期把通用质量词排在最前，预算被压缩时**先丢掉的竟是
    "anthropomorphic"**（"不许画成拟人"），而它恰恰是非人物分支最不能丢的一条。
    """
    subject_type = subject.get("subject_type", "unknown")
    groups = []
    if subject_type == "human":
        groups.append(config.HUMAN_NEGATIVE_PROMPT)
    else:
        groups.append(config.NON_HUMAN_NEGATIVE_PROMPT)
        if subject_type == "animal":
            groups.append(config.ANIMAL_NEGATIVE_PROMPT)
        elif subject_type == "object":
            groups.append(config.OBJECT_NEGATIVE_PROMPT)
    groups.append(config.NEGATIVE_PROMPT)
    _, coverage_negative = composition_constraints(subject)
    groups.extend(coverage_negative)
    pieces = []
    for group in groups:
        pieces.extend(_negative_pieces(group))
    return ", ".join(dict.fromkeys(pieces))


def _token_count(text) -> int:
    """text 的 CLIP token 数（与训练入口共用同一份实现，见 `text_budget`）。"""
    return text_budget.count(text)


def _clip_tokenizers():
    """底模的两个 CLIP tokenizer（`fit_prompt` 需要它们才能准确计数）。

    实现已集中到 `text_budget`：本项目曾经在三个地方各写一份预算逻辑，
    结果三处都漏（详见 `text_budget` 的模块注释）。
    """
    return text_budget.tokenizers()


def positive_prompt(folder, profile, subject, trigger=None) -> str:
    """构造提示词：触发词 + 角色条件 +（人物：面部还原）+ 源图几何/媒介保留。

    条件超出 CLIP 预算时按"角色条件 -> 源图保留条件"的顺序保留必需项；
    仍然超预算的装饰性词由 fit_prompt 记录丢弃，不静默截断。
    """
    trigger = trigger or config.get_trigger(folder)
    subject_type = subject.get("subject_type", "unknown")
    card = cards_module.build_card(profile)
    required = [trigger]
    if card.get("clause"):
        required.append(card["clause"])

    if subject_type == "human":
        # 极致 cosplay：明确要求面部身份成为角色，而不是"原主体穿着角色服装"
        required.append("the person's face becomes the character's face")
    elif subject_type in {"animal", "object"}:
        # 非人物：只做表面与附件层面的角色化，禁止添加人体结构（方案 §7）
        required.append("character colour scheme and pattern applied to the surface, no human face")
        # 非人物没有"换脸"这一条，required 反而更长；实测动物提示词会超过 CLIP 的
        # 77 token 上限（日志出现 "88 > 77"），于是末尾的质量词被截断。
        # 处理：非人物丢掉最长的那段服装描述（权重在视觉上已经表达），
        # 保住"角色配色 + 发色 + 瞳色"这些身份锚点与质量词。
        required[1] = _trim_clause(required[1], max_parts=4, max_chars=28)

    preserve = []
    source_medium = medium_terms(subject.get("style"))
    if source_medium:
        preserve.append(source_medium)
    positive_geometry, _ = composition_constraints(subject)
    preserve.extend(positive_geometry)
    framing = str(subject.get("framing") or "").lower()
    if framing in {"upper body", "bust", "portrait", "close-up"}:
        preserve.append("upper body")
    pose = str(subject.get("pose") or "").strip()
    if pose and pose.lower() not in {"unknown", "none"}:
        preserve.append(f"pose: {pose}")

    # 标志条件属于**可选**项，刻意排在"保住源图"的约束之后：
    # 实测（2026-10-05）结构化后的标志描述很长（单条可到 272 字符），
    # 早期把它塞进 required 后，`fit_prompt` 为了容纳它把 preserve 里的
    # 源图媒介（pixel art 等）整个丢掉——用户明确要求"保留原图画风"，
    # 拿画风换标志是本末倒置。因此：画风与构图优先，标志能写多少写多少。
    marks_clause = ""
    if subject_type == "human" and config.MARK_IN_PROMPT:
        # 按"条"取标志条件、并用**短点名**：描述自身含逗号，先连成整串再按逗号裁
        # 会把描述切碎；而完整描述一条就要 20+ token，整图提示词只够点名。
        # 训练与推理都点名同一批（短）标志，条件分布保持一致——这一点很关键：
        # 早期"训练写全、推理写半截"会让模型学到的条件与使用时对不上。
        marks_items = marks_module.condition_items(
            folder, max_chars=config.MARK_PROMPT_MAX_CHARS, brief=True)
        marks_clause = _fit_mark_items(
            marks_items, budget=config.MARK_PROMPT_TOKEN_BUDGET, joiner=", ")
    quality = str(config.TRAIN_QUALITY_TAGS or "").strip()
    optional = ([marks_clause] if marks_clause else []) + preserve + \
        ([quality] if quality else [])
    try:
        prompt, dropped = fit_prompt(required, optional, tokenizers=_clip_tokenizers(), max_tokens=75)
    except ValueError:
        # 必需项本身超预算：先压缩角色条件（保留配色与配饰，丢掉最长的描述），
        # 而不是丢弃"保留源图"的约束——那会让结果偏离需求。
        trimmed = _trim_clause(card.get("clause") or "", max_parts=4)
        required = [trigger] + ([trimmed] if trimmed else []) + required[2:]
        prompt, dropped = fit_prompt(required, optional, tokenizers=_clip_tokenizers(), max_tokens=75)
    if dropped:
        print(f"[transfer] 提示词预算已满，未加入: {dropped}")
    return _enforce_token_budget(prompt, 75)


def _enforce_token_budget(prompt, max_tokens=75):
    """最后一道硬闸：交给 CLIP 的正向词一定在预算内。

    为什么需要：`fit_prompt` 会用真 tokenizer 逐条取舍，理论上已经够了；
    但**中文的 token 权重随上下文变化**——把若干条目拼在一起之后，
    整串的 token 数可能大于各条目之和（实测 73 → 90 出现过）。
    因此这里对最终字符串再核一次，超了就按逗号从**末尾**丢条目
    （末尾是质量词、构图词这类最可丢的），直到进得去。
    丢掉的条目会打印，不静默截断。
    """
    if _token_count(prompt) <= max_tokens:
        return prompt
    parts = [part.strip() for part in str(prompt).split(",") if part.strip()]
    dropped = []
    while parts and _token_count(", ".join(parts)) > max_tokens:
        dropped.append(parts.pop())
    if dropped:
        print(f"[transfer] 最终正向词超预算，已丢弃末尾条目: {dropped}")
    return ", ".join(parts)


def _shrink_to_budget(text, budget):
    """把单条文本按 **token** 逐字收缩到预算内（二分找最长可容纳前缀）。

    中文的字符数与 token 数差得很远（实测 21 个汉字 ≈ 21 token，
    按字符截到 32 字符仍可能超预算），所以只能按 token 二分。
    """
    text = str(text or "").strip()
    if not text or _token_count(text) <= budget:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if _token_count(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    if low <= 0:
        return ""
    return text[:low].rstrip(" ,，、:：")


def _fit_mark_items(items, budget, joiner=", "):
    """把**多条标志条件**按预算塞进去，多余的整条丢弃（**不切断单条**）。

    为什么要按"条"而不是按逗号（实测踩过，是"非人物加法完全看不到"的根因之一）：
    `carry_clause` 产出的是 `this real repeating pattern: <描述>` 这样的整句，
    而描述自身**含逗号**。早期的 `_fit_mark_clause` 按逗号无条件切分，
    于是 `light purple three concentric circles with two intersecting lines
    on a white background` 被切成若干碎片，每片单看都没有意义的图案信息；
    更糟的是第一条碎片就占满 30 token 的预算，后面的标志**一条都进不去**——
    而在 `surface_prompt` 里这条是被 `if carry:` 判断后才追加的，
    结果是"有承载计划、却没有任何标志进入提示词"，日志上只看到预算已满。

    **装不下第一条时不再直接放弃整条预算**：按长度从短到长试，
    让短条目先占位（实测：点名预算 14 token 时，第一条 11 token 的长名
    会把后面 4~6 token 的有效短名全部挡掉，最后只写进去一个标志）。
    顺序约束只在"同一优先级内"放松——重要性排序仍由调用方给出，
    这里只解决"长条目霸占预算"这一个问题。
    """
    pieces = [str(piece).strip() for piece in (items or []) if str(piece).strip()]
    if not pieces:
        return ""
    kept = [pieces[0]] if _token_count(pieces[0]) <= budget else []
    for piece in pieces[1:]:
        candidate = joiner.join(kept + [piece])
        if _token_count(candidate) <= budget:
            kept.append(piece)
    if not kept:
        # 第一条就超预算：先试更短的条目，全都装不下才收缩第一条
        for piece in sorted(pieces[1:], key=_token_count):
            if _token_count(piece) <= budget:
                kept = [piece]
                break
    if kept:
        return joiner.join(kept)
    return _shrink_to_budget(pieces[0], budget)


def _fit_mark_clause(clause, budget=22):
    """把标志条件按**逗号条目**裁到 token 预算内（从前往后保留）。

    只用于"逗号切分不会破坏语义"的条件（例如 `condition_clause` 的英文条列）。
    标志描述自身含逗号时请改用 `_fit_mark_items`，否则描述会被切成碎片。
    """
    pieces = [piece.strip() for piece in str(clause or "").split(",") if piece.strip()]
    kept = []
    for piece in pieces:
        candidate = ", ".join(kept + [piece])
        if _token_count(candidate) > budget:
            continue
        kept.append(piece)
    if kept or not pieces:
        return ", ".join(kept)
    return _shrink_to_budget(pieces[0], budget)


def _trim_clause(clause, max_parts=4, max_chars=40):
    """按逗号切分条件文本，只保留前若干项并截断过长的单项。"""
    parts = [p.strip() for p in str(clause).split(",") if p.strip()]
    trimmed = [p if len(p) <= max_chars else p[:max_chars].rstrip() for p in parts[:max_parts]]
    return ", ".join(trimmed)


def negative_prompt_with_style(subject) -> str:
    """在主体负向词之上补上"与源媒介冲突"的负向词，防止画风被换成角色参考的画风。

    媒介冲突词必须和主体负向词**一起**做预算：实测单独给主体负向词做预算之后，
    动物的合成结果仍有 80/77——超出的正是这里追加的媒介词。

    优先级（实测调整过）：主体核心（不许变成人 / 不许画坏）
    -> **媒介冲突词** -> 构图覆盖类负向词。
    媒介词曾经排在覆盖词之后，于是"线稿输入禁止 photograph"被挤掉——
    而保住源图媒介正是用户需求里明确的一条（保留原图画风）；
    构图覆盖词只是"别画全身"这类容忍度更高的约束，被丢弃的应该是它。
    """
    subject_type = subject.get("subject_type", "unknown")
    groups = []
    if subject_type == "human":
        groups.append(config.HUMAN_NEGATIVE_PROMPT)
    else:
        groups.append(config.NON_HUMAN_NEGATIVE_PROMPT)
        if subject_type == "animal":
            groups.append(config.ANIMAL_NEGATIVE_PROMPT)
        elif subject_type == "object":
            groups.append(config.OBJECT_NEGATIVE_PROMPT)
    groups.append(config.NEGATIVE_PROMPT)
    _, coverage_negative = composition_constraints(subject)
    key = medium_key(subject.get("style"))
    extra = MEDIUM_NEGATIVES.get(key, "")

    budget = int(config.NEGATIVE_TOKEN_BUDGET)
    kept, dropped = [], []

    def try_add(part):
        candidate = ", ".join(kept + [part])
        if _token_count(candidate) <= budget:
            kept.append(part)
            return True
        dropped.append(part)
        return False

    # 顺序即优先级；每一组内部再按逗号拆成小条目逐个加入，
    # 这样"装不下整组"不会把组里最关键的那一条也一起丢掉。
    for group in groups:
        for piece in _negative_pieces(group):
            if piece not in kept:
                try_add(piece)
    # 媒介冲突词排在核心约束之后；整条放不下时按条目裁剪，
    # "线稿不许变成照片"这类最关键的一条会先被保住。
    for piece in _negative_pieces(extra):
        if piece not in kept:
            try_add(piece)
    for piece in _negative_pieces(", ".join(coverage_negative)):
        if piece not in kept:
            try_add(piece)
    if dropped:
        print(f"[transfer] 负向词预算已满（{budget} token），未加入: {dropped}")
    return ", ".join(kept)


def character_reference_prompt(card) -> str:
    """文生图条件下"把角色画出来"的提示词（用于取颜色参照与底模能力探针）。

    只用身份条件与质量词，不加源图几何与媒介条件：目的是让模型自由地画出它认为
    这个角色应该有的颜色，而不是被源图的构图约束。
    """
    parts = [card.get("trigger") or "", card.get("clause") or "",
             config.TRAIN_QUALITY_TAGS or "", "upper body, looking at viewer, simple background"]
    return ", ".join(p for p in parts if p)


def style_clause(profile, subject) -> str:
    """说明本次运行"学的是角色、保的是源图"，写进生成记录供复核。"""
    return (f"角色语义来自 {profile.get('folder')}（source={profile.get('source')}）；"
            f"源图媒介={subject.get('style') or '未识别'}；"
            f"迁移目标={config.TRANSFER_GOAL}")


# ---------------------------------------------------------------------------
# 条件准备与校验
# ---------------------------------------------------------------------------
def prepare(folder, subject, image, profile=None, card=None) -> dict:
    """构造一次生成的完整条件包；任何必需条件缺失都在这里显式失败。"""
    profile = profile or semantics.published_profile(folder)
    if profile is None:
        raise RuntimeError(
            f"角色 {folder} 没有语义档案，无法构造条件；请先运行 python train.py --character {folder}")
    card = card or cards_module.build_card(profile)
    semantics_audit = semantics.audit_switch(folder)
    if not semantics_audit["ready"]:
        raise FileNotFoundError(
            f"角色包不完整（{semantics_audit['reason']}）：{semantics_audit.get('missing')}")
    weights = [entry for entry in semantics_audit["weights"] if entry["role"] == "identity"]
    if not weights and not config.MULTI_ADAPTER_ALLOWED:
        raise FileNotFoundError(f"角色 {folder} 的角色包内没有 identity 权重")

    boundary = hair_boundary(subject, profile)
    package = {
        "character": folder,
        "trigger": card.get("trigger"),
        "card": card,
        "profile": profile,
        "weights": semantics_audit["weights"],
        "subject": subject,
        "settings": generation_settings(subject.get("subject_type", "unknown"), source=image),
        "positive": positive_prompt(folder, profile, subject, trigger=card.get("trigger")),
        "negative": negative_prompt_with_style(subject),
        "structure": structure_condition(image),
        "hair_boundary": boundary,
        "style_clause": style_clause(profile, subject),
        "coverage_gaps": semantics.coverage(profile),
    }
    for gap in package["coverage_gaps"]:
        print(f"[transfer] 证据缺口（不会臆测）: {gap}")
    return package


def region_refine_settings(subject_type: str, source=None) -> dict:
    """局部强化重绘的参数：强度更高、结构条件更弱、角色权重更强。

    source 给定时，线条密集型输入改用更保守的一套（强度更低、结构更高），
    避免把线稿的线条在区域重绘里抹掉；`config.LINEART_REGION_REFINE=False` 时
    线条类输入**整条跳过**区域强化（实测那一遍会把线条抹掉、只剩一块发白色块）。
    """
    if not getattr(config, "REGION_REFINE", False):
        return {}
    if subject_type not in {"human", "animal", "object", "unknown"}:
        return {}
    strength = float(config.REGION_REFINE_STRENGTH)
    control = float(config.REGION_REFINE_CONTROL_SCALE)
    if source is not None and is_lineart_like(source):
        if not getattr(config, "LINEART_REGION_REFINE", True):
            return {}
        strength = min(strength, config.LINEART_REGION_STRENGTH)
        control = max(control, config.LINEART_REGION_CONTROL_SCALE)
    return {
        "strength": strength,
        "control_scale": control,
        "lora_scale": float(config.REGION_REFINE_LORA_SCALE),
        "targets": list(config.REGION_REFINE_TARGETS),
        # 局部强化这一遍是**用角色权重把该区域重画**：人物走身份支，
        # 概念支按主体类型（人物默认为 0）。写明 `scale_target` 之后，
        # `_render` 不会再默认把强度写到身份支上去（见 generation_settings 的说明）。
        "scale_target": "default",
        "identity_scale": float(config.REGION_REFINE_LORA_SCALE),
        "concept_scale": float(adapter_scales_for(subject_type)["concept"]),
        "lineart_like": False,
    }


def region_mask(image, box, pad=None, feather=None):
    """按归一化框生成羽化掩版（L 模式，255=区域内部）。

    框外扩 pad 比例，避免切口正好压在轮廓线上；羽化让局部结果与整图结果平滑过渡。
    """
    from PIL import Image, ImageDraw, ImageFilter
    pad = config.REGION_REFINE_PAD if pad is None else pad
    feather = config.REGION_REFINE_FEATHER if feather is None else feather
    width, height = image.size
    left, top, right, bottom = box
    box_w, box_h = right - left, bottom - top
    left = max(0.0, left - box_w * pad)
    top = max(0.0, top - box_h * pad)
    right = min(1.0, right + box_w * pad)
    bottom = min(1.0, bottom + box_h * pad)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle(
        [left * width, top * height, right * width, bottom * height], fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather / 2.0))
        # 羽化会把整幅都抬离 0，重新拉回纯黑背景，保证区域外逐像素等于整图结果
        mask = mask.point(lambda value: 0 if value < 8 else min(255, int(value * 1.35)))
    return mask


def attachment_settings(region) -> dict:
    """非人物"加法"支路的生成参数：更强概念权重 + 更弱结构条件，让配饰真正成形。

    `ATTACH_LORA_SCALE`（1.60）按 config 的注释是**概念权重强度**，
    因此这里显式写 `scale_target="concept"`、身份支置 0。
    实测踩过：不写明时这个 1.60 会被写到**身份**权重上，而"非人物只用概念权重"
    的设定被静默覆盖，于是"配饰渲染不出可辨认形状"这条老结论其实是在错误的权重下得到的。
    """
    if not config.ATTACH_NON_HUMAN:
        return {}
    return {
        "strength": config.ATTACH_REFINE_STRENGTH,
        "lora_scale": config.ATTACH_LORA_SCALE,
        "control_scale": config.ATTACH_CONTROL_SCALE,
        "control_end": config.CONTROL_GUIDANCE_END,
        "region": region,
        "scale_target": "concept",
        "identity_scale": float(getattr(config, "NON_HUMAN_IDENTITY_SCALE", 0.0)),
        "concept_scale": float(config.ATTACH_LORA_SCALE),
        "lineart_like": False,
    }


def attachment_prompt(folder, profile, subject, region) -> str:
    """附着区域的提示词：明确"添加"而不是"替换"，并禁止人体结构。

    标志描述来自该角色**有真实证据**的标志，并按承载计划写出"把这个真实的标志
    做成颈圈饰片 / 胸饰"——而不是泛泛的"角色图案"。

    为什么必须这么写（config.py 里记录的附着区域四轮实测）：概念权重学的是
    "角色裁片长什么样"，没有"这是一条戴在脖子上的项圈"这种位置/语义概念，
    单靠"加个配饰"实测只会在猫颈部得到一层看不出是什么的暖色。
    写明"用这个真实标志做成可佩戴件"才有可能真正成形。

    预算（实测）：这条提示词曾经到 **85/77 token**——因为它同时拼了承载计划、
    配色摘要、档案条件与一长串保留约束，而其中"配色摘要"与档案条件里的
    colour scheme 完全重复。这里按优先级裁剪，并由 `fit_prompt` 兜底。
    """
    trigger = config.get_trigger(folder)
    card = cards_module.build_card(profile)
    palette = profile_module.palette_summary(profile)
    subject_type = subject.get("subject_type", "animal")
    carry = marks_module.carry_clause(folder, subject_type, limit=2)
    if region == "collar":
        subject_phrase = ("a collar band in the character's colours, shaped as a real "
                          "wearable ornament")
    elif region == "chest":
        subject_phrase = ("a decorated chest piece in the character's colours, shaped as a "
                          "real ornament")
    else:
        subject_phrase = ("a small three-dimensional ornament in the character's colours "
                          "worn on top of the head, clearly shaped, not a flat colour band")
    required = [trigger, subject_phrase]
    if carry:
        required.append(carry)
    elif palette:
        # 没有标志证据时才用配色摘要；有承载计划就不再重复写颜色
        required.append(palette)
    # 档案条件只取前 3 段：它包含 colour scheme，与配色摘要重复，全量拼会直接爆预算
    clause = _trim_clause(card.get("clause") or "", max_parts=3, max_chars=30)
    if clause and not carry:
        required.append(clause)
    # 必须保留主体本身：非人物不换皮、不拟人化。按优先级排序，被丢弃的是靠后的装饰性条目。
    preserve = ["keep the original animal unchanged", "no human face", "no human body",
                "the surface keeps its own pattern", "clearly shaped accessory",
                "no flat colour band"]
    # 质量词排最后（它最可丢）；"必须保留主体"这类语义约束优先。
    quality = str(config.TRAIN_QUALITY_TAGS or "").strip()
    if quality:
        preserve.append(quality)
    try:
        prompt, dropped = fit_prompt(required, preserve,
                                     tokenizers=_clip_tokenizers(), max_tokens=75)
    except ValueError:
        prompt = ", ".join(required[:2] + preserve[:2])
        dropped = ["承载计划与配色（必需项本身超预算）"]
    if dropped:
        print(f"[transfer] 附着区域提示词预算已满，未加入: {dropped}")
    return prompt


def surface_settings() -> dict:
    """物体表面图案支路的生成参数（概念权重画图案，结构条件保住形状与光影）。

    `OBJECT_SURFACE_LORA_SCALE`（1.45）按 config 的注释是**概念权重强度**，
    因此显式写 `scale_target="concept"`、身份支置 0。
    """
    if not config.OBJECT_SURFACE:
        return {}
    return {
        "strength": config.OBJECT_SURFACE_STRENGTH,
        "lora_scale": config.OBJECT_SURFACE_LORA_SCALE,
        "control_scale": config.OBJECT_SURFACE_CONTROL_SCALE,
        "control_end": config.CONTROL_GUIDANCE_END,
        "scale_target": "concept",
        "identity_scale": float(getattr(config, "NON_HUMAN_IDENTITY_SCALE", 0.0)),
        "concept_scale": float(config.OBJECT_SURFACE_LORA_SCALE),
        "lineart_like": False,
    }


def surface_prompt(folder, profile, subject) -> str:
    """物体表面图案的提示词：明确是"在物体表面画**这个角色真实的**图案"。

    规格 §2 第 8 行的两条修正都在这里：
      * **禁止**给没有花卉证据的角色补 `floral pattern`——那是凭空加内容；
      * 不再追加身份 clause（"这个角色是谁"与"物体表面画什么图案"是两件事，
        身份条件在非人物上强度本来就是 0，写进去只挤占 CLIP 预算）。
    图案描述改为取自承载计划里**有图像证据**的标志；一条都没有时如实留空，
    并在条件里说明"没有可用的标志证据"，而不是编一个通用的花纹。

    预算（实测）：这条提示词曾经到 **261/77 token**——承载计划里写进了
    结构化后很长的标志描述，而它是直接拼接、没有任何裁剪。现在按优先级
    用 `fit_prompt` 裁到 75 token 以内，被丢弃的条目会打印出来。
    """
    trigger = config.get_trigger(folder)
    carries = marks_module.carry_items(folder, "object", limit=3)
    required = [trigger, "the character's real marks on the object surface"]
    optional = []
    # 配色**不再单独追加**：档案里的配色维度比标志级条件宽（实测这里是
    # `blue, pink, gold` 三个颜色名，约 6 token），而 77 token 的预算里
    # 每一份额度都要留给"看得出是这个角色的标志"。颜色本身会随标志描述一起进来
    # （标志描述里带具体颜色），因此这里省下的额度直接给标志与语义约束。
    if carries:
        # 按"条"塞入，不按逗号切分：标志描述自身含逗号，切碎后就没有图案信息了
        # （实测：这是"非人物加法完全看不到"的根因之一）。
        # 预算 30 实测只装得下 1 条（单条描述约 22 token），
        # 而用户要的是"一眼看得出是这个角色的标志"——只给一条图案显然不够，
        # 因此按 config 给到能装 2~3 条的额度。
        optional.append(_fit_mark_items(
            carries, budget=config.SURFACE_MARK_TOKEN_BUDGET))
    else:
        # 没有标志证据时只做配色，不编造花纹（规格 §2 第 8 行）
        palette = profile_module.palette_summary(profile)
        if palette:
            optional.append(palette)
        optional.append("colour scheme only, no invented floral or geometric pattern")
        print("[transfer] 表面图案：该角色没有可用的标志证据，只做配色，不补任何花纹")
    optional.extend(["keep the object's shape, material and lighting unchanged",
                     "no human face", "no human body",
                     "no character standing in the scene"])
    # 质量词排在**最后**：它是最可丢的装饰项。实测 75 token 预算下它常常挤掉
    # "no character standing in the scene" 这类**语义约束**——而用户要的是
    # "一眼看得出是这个角色的标志"，不是"画质词齐全"。语义约束优先于画质词。
    quality = str(config.TRAIN_QUALITY_TAGS or "").strip()
    if quality:
        optional.append(quality)
    try:
        prompt, dropped = fit_prompt(required, optional,
                                     tokenizers=_clip_tokenizers(),
                                     max_tokens=75)
    except ValueError:
        prompt = ", ".join(required[:2])
        dropped = ["配色与承载计划（必需项本身超预算）"]
    if dropped:
        print(f"[transfer] 表面图案提示词预算已满，未加入: {dropped}")
    return prompt


def prepare_surface(folder, subject, image, box, profile=None, card=None) -> dict:
    """物体表面图案的条件包（框内做图案，框外由合成阶段逐像素保留源图）。"""
    profile = profile or semantics.published_profile(folder)
    card = card or cards_module.build_card(profile)
    settings = surface_settings()
    if not settings:
        return {}
    return {
        "character": folder,
        "box": box,
        "positive": surface_prompt(folder, profile, subject),
        "negative": negative_prompt_with_style(subject),
        "settings": settings,
    }


def prepare_attachment(folder, subject, image, box, region, profile=None, card=None) -> dict:
    """附着区域的条件包（非人物加法支路）。"""
    profile = profile or semantics.published_profile(folder)
    card = card or cards_module.build_card(profile)
    settings = attachment_settings(region)
    if not settings:
        return {}
    return {
        "character": folder,
        "box": box,
        "region": region,
        "positive": attachment_prompt(folder, profile, subject, region),
        "negative": negative_prompt_with_style(subject),
        "settings": settings,
    }


def prepare_region(folder, subject, image, box, profile=None, card=None) -> dict:
    """局部强化重绘用的条件包：条件文本与整图一致，强度与权重单独取自 config。"""
    profile = profile or semantics.published_profile(folder)
    card = card or cards_module.build_card(profile)
    settings = region_refine_settings(subject.get("subject_type", "unknown"), source=image)
    if not settings:
        return {}
    return {
        "character": folder,
        "box": box,
        "positive": positive_prompt(folder, profile, subject, trigger=card.get("trigger")),
        "negative": negative_prompt_with_style(subject),
        "settings": settings,
    }


def feather_alpha(mask, feather=None, interior=False):
    """把硬边掩版变成"内部实心 + 边界渐变"的 alpha 图（全项目唯一一份实现）。

    为什么不能直接高斯模糊整张掩版：模糊会把边界内侧的整圈也拉低成半透明，
    于是"区域内部"出现一圈中间值的硬台阶——实测症状就是轮廓周围那条
    "过度分层"的带子。正确做法是：内部保持实心，只让边界外侧过渡到 0，
    即 [实心区] -> [渐变带] -> [外部 0]。

    interior=True 时，过渡带被移到掩版**内部**（先向里腐蚀 feather 像素再模糊）：
    此时掩版外逐像素保持 0，合成结果在掩版外严格等于底图。需要"掩版外一定是源图"
    的场合（配色锚定、只保留区域合成）必须用这个模式，否则 2~3 像素的过渡值
    会渗出掩版外，在轮廓周围留下一条可见的亮/色边。
    """
    from PIL import Image, ImageFilter
    layer = mask.convert("L")
    binary = layer.point(lambda v: 255 if v >= 128 else 0)
    feather = config.REGION_REFINE_FEATHER if feather is None else feather
    if feather <= 0 and not config.EDGE_CLEAN:
        return binary
    kernel = max(3, int(feather) | 1) if feather > 0 else 3
    if interior:
        inner = binary.filter(ImageFilter.MinFilter(kernel))
        alpha = inner.filter(ImageFilter.GaussianBlur(max(0.5, feather / 2.0)))
    else:
        inner = binary.filter(ImageFilter.MinFilter(kernel))
        blurred = binary.filter(ImageFilter.GaussianBlur(max(0.5, feather / 2.0)))
        # 内核内保持 255；内核外只保留被模糊抬起来的过渡值；再往外为 0
        alpha = Image.composite(Image.new("L", binary.size, 255), blurred, inner)
    if config.EDGE_CLEAN:
        import numpy as np
        array = np.asarray(alpha).astype(np.float32)
        low = 255.0 * config.EDGE_CLEAN_LOW
        high = 255.0 * config.EDGE_CLEAN_HIGH
        array = np.clip((array - low) / max(1.0, high - low), 0.0, 1.0) * 255.0
        alpha = Image.fromarray(array.astype(np.uint8), mode="L")
    return alpha


def merge_region(base, overlay, mask, feather=None):
    """把局部结果按掩版合成回整图结果（掩版外**逐像素**等于整图结果）。"""
    from PIL import Image
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.NEAREST)
    if overlay.size != base.size:
        overlay = overlay.resize(base.size, Image.LANCZOS)
    alpha = feather_alpha(mask, feather)
    return Image.composite(overlay.convert("RGB"), base.convert("RGB"), alpha)


def preserve_source_outside(source, result, regions, edited, max_distance=None):
    """把"不需要改的地方"逐像素还原成源图，只保留真正发生了角色化的像素。

    实测依据：整图 img2img 会把源图高频细节几乎抹平（照片样例的拉普拉斯方差
    119.1 -> 5.5，线稿 1054 -> 1.9），这就是"涂抹感"的主因——不是某一档强度太高，
    而是整幅重画本身。保源图细节的做法：

    * regions：本次真正要改的像素区域（发区/服装区/附件），这些地方一律用生成结果；
    * 其余像素：只有在"源图与结果差异足够大"时才采用结果（说明那里确实被角色化了），
      否则还原源图。这样皮肤、背景、线稿线条等与角色无关的细节全部保住。

    edited 是本次真正参与改写的区域名集合（用于审计），max_distance 控制判定阈值。
    """
    import numpy as np
    from PIL import Image
    if not config.PRESERVE_SOURCE_OUTSIDE_REGIONS:
        return result, {"applied": False, "reason": "配置关闭"}
    max_distance = (config.PRESERVE_MAX_COLOR_DISTANCE if max_distance is None
                    else max_distance)
    base = np.asarray(source.convert("RGB")).astype(np.int16)
    rendered = np.asarray(result.convert("RGB")).astype(np.int16)
    if base.shape != rendered.shape:
        return result, {"applied": False, "reason": "尺寸不一致"}

    keep = np.zeros(base.shape[:2], dtype=bool)          # True=采用生成结果
    for mask in (regions or {}).values():
        if mask is None:
            continue
        layer = mask.convert("L")
        if layer.size != source.size:
            layer = layer.resize(source.size, Image.NEAREST)
        keep |= np.asarray(layer) > 128
    # 区域外：差异大才采用（那里的确变了），否则还原源图细节
    distance = np.abs(base - rendered).sum(axis=2)
    keep |= distance > max_distance

    merged = np.where(keep[:, :, None], rendered, base).astype(np.uint8)
    changed = float(keep.mean())
    return Image.fromarray(merged, mode="RGB"), {
        "applied": True, "kept_ratio": round(changed, 4),
        "max_color_distance": max_distance, "edited_regions": sorted(edited or []),
    }


def match_region_luminance(source, result, regions, tolerance=None, skip_regions=()):
    """把每个改写区域的**平均明度**对回源图，抑制"整图重绘顺手提亮/压暗"。

    实测症状：整图 img2img 会把某些区域整体提亮（猫额头同一区域均值 169.9 → 185.4），
    看起来像给主体加了一条淡带。这不是任何附着区域造成的，而是整幅重绘的副作用。
    做法：在每个区域内比较源图与结果的均值，差得超过容差就按差值整体平移回来
    （平移而不是乘系数，保留区域内原有的明暗层次）。

    `skip_regions`：**不要对回源图的区域**。做配色锚定（`vision/color_hint`）时，
    锚定会**有意**改变该区域的平均明度（把深色头发搬到浅色目标色），
    而本函数的 `source` 是**锚定之后**的图，于是"对回源图"会把刚搬过去的明度又搬回来——
    实测 human2：头部被平移 **-43.8**，等于把发色锚定整个抵消掉，
    出图头发依旧是源图的深棕色。因此锚定过的区域必须显式排除。
    """
    import numpy as np
    from PIL import Image
    if tolerance is None:
        tolerance = config.LUMINANCE_MATCH_TOLERANCE
    if not regions:
        return result, {"applied": False, "reason": "没有区域"}
    skip = {str(name) for name in (skip_regions or ())}
    base = np.asarray(source.convert("RGB")).astype(np.float32)
    rendered = np.asarray(result.convert("RGB")).copy()
    entries, skipped = [], []
    for name, mask in regions.items():
        if mask is None:
            continue
        if str(name) in skip:
            skipped.append(str(name))
            continue
        layer = mask.convert("L")
        if layer.size != source.size:
            layer = layer.resize(source.size, Image.NEAREST)
        inside = np.asarray(layer) > 128
        if inside.sum() < 64:
            continue
        delta = float(base[inside].mean()) - float(rendered[inside].mean())
        if abs(delta) <= tolerance:
            continue
        rendered[inside] = np.clip(rendered[inside].astype(np.float32) + delta, 0, 255)
        entries.append({"region": name, "mean_shift": round(delta, 2),
                        "pixels": int(inside.sum())})
    if skipped:
        print(f"[model] 区域明度平移已跳过（这些区域做过配色锚定，对回源图会抵消锚定）: {skipped}")
    if not entries:
        return result, {"applied": False, "skipped_regions": skipped,
                        "reason": ("各区域明度已在容差内" if not skipped
                                   else "锚定区域已跳过，其余区域明度在容差内")}
    print("[model] 区域明度已对回源图: "
          + ", ".join(f"{e['region']}{e['mean_shift']:+.1f}" for e in entries))
    return Image.fromarray(rendered.astype(np.uint8), mode="RGB"), {
        "applied": True, "tolerance": tolerance, "regions": entries,
        "skipped_regions": skipped}


def high_frequency_ratio(source):
    """源图的高频占比（0~1）：线条/纹理密集的输入（线稿、像素画）值高，照片值低。

    用途：线稿这类输入的"内容"就是线条本身，被扩散抹平后基本什么都不剩；
    因此需要对它们**收紧改写范围**（少动、只在必要处动），而不是照照片的参数一视同仁。
    判据只用源图自身的统计量，不含任何角色/题材假设。
    """
    import numpy as np
    if source is None or not hasattr(source, "convert"):
        return 0.0          # 拿不到图像时不判定，按普通输入处理
    grey = np.asarray(source.convert("L")).astype(np.float32)
    if grey.size < 64:
        return 0.0
    lap = np.abs(grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
                 - 4 * grey[1:-1, 1:-1])
    # 用固定的绝对阈值判断"边缘像素"：线稿的黑白跳变远高于照片的自然纹理
    return float((lap > config.LINEART_EDGE_THRESHOLD).mean())


def is_lineart_like(source):
    """是否属于"线条即内容"的输入（线稿、草图、像素画）。"""
    return high_frequency_ratio(source) >= config.LINEART_EDGE_RATIO


def sharpen_region(source, rendered, mask, amount=None, radius=None):
    """只对被改写的区域做反锐化，补回扩散造成的软化。

    为什么需要：扩散重绘的输出天然比源图软，直接合成回源图会在边界处显出
    "糊—锐"突变，看起来就是"这一块被涂抹过、那一块很清楚"。这里把区域结果
    先按源图的锐度补回来，再合成，两边的清晰度才接得上。
    做法只在掩版内生效：掩版外逐像素不动，因此不会给未改写区域引入人工痕迹。
    """
    import numpy as np
    from PIL import Image, ImageFilter
    if not config.REGION_SHARPEN:
        return rendered, {"applied": False, "reason": "配置关闭"}
    amount = config.REGION_SHARPEN_AMOUNT if amount is None else amount
    radius = config.REGION_SHARPEN_RADIUS if radius is None else radius
    layer = mask.convert("L")
    if layer.size != source.size:
        layer = layer.resize(source.size, Image.NEAREST)
    inside = np.asarray(layer) > 128
    if inside.sum() < 64:
        return rendered, {"applied": False, "reason": "区域过小"}
    base = rendered.convert("RGB")
    blurred = base.filter(ImageFilter.GaussianBlur(radius))
    sharp = np.clip(np.asarray(base).astype(np.float32)
                    + amount * (np.asarray(base).astype(np.float32)
                                - np.asarray(blurred).astype(np.float32)), 0, 255)
    out = np.asarray(rendered.convert("RGB")).copy()
    out[inside] = sharp[inside].astype(np.uint8)
    return Image.fromarray(out, mode="RGB"), {"applied": True, "amount": amount,
                                              "radius": radius,
                                              "pixels": int(inside.sum())}


_UNSET = object()


def limit_change_to_regions(source, rendered, keep_mask, max_distance=_UNSET):
    """只有在"相对源图确有变化"的地方才采用生成结果，其余逐像素还原源图。

    这是"整图重画导致发糊"的根本对策：不被点名的区域**不参与重绘**，
    即使生成结果把它们画得更"好看"，也不采用——因为源图在那里本来就是对的、
    而且是清晰的。判定用颜色距离，阈值越小越保守。

    max_distance 的语义（踩过坑，必须区分两种"没给值"）：
        * 不传（_UNSET）      -> 用 config.LIMIT_CHANGE_MAX_DISTANCE；
        * 显式传 None         -> **只保留区域内的生成结果**（区域外逐像素等于源图），
                                 即"只保留区域改写"模式；边界最干净。
    早期写法把 None 也当成"没给值"，于是显式传 None 会被配置默认值顶掉，
    调用方以为切换了模式、实际没有生效（实测踩过）。
    """
    import numpy as np
    from PIL import Image
    if not config.LIMIT_CHANGE_TO_REGIONS:
        return rendered, {"applied": False, "reason": "配置关闭"}
    region_only = max_distance is None
    if max_distance is _UNSET:
        max_distance = config.LIMIT_CHANGE_MAX_DISTANCE
    base = np.asarray(source.convert("RGB")).astype(np.int16)
    drew = np.asarray(rendered.convert("RGB")).astype(np.int16)
    if base.shape != drew.shape:
        return rendered, {"applied": False, "reason": "尺寸不一致"}
    inside = np.zeros(base.shape[:2], dtype=bool)
    for mask in (keep_mask or {}).values() if isinstance(keep_mask, dict) else [keep_mask]:
        if mask is None:
            continue
        layer = mask.convert("L")
        if layer.size != source.size:
            layer = layer.resize(source.size, Image.NEAREST)
        inside |= np.asarray(layer) > 128
    distance = np.abs(base - drew).sum(axis=2)
    # 只保留区域改写的模式：严格只在掩版内采用生成结果，
    # 区域外逐像素等于源图 —— 分辨率、纹理、光影全部天然保留。
    if region_only:
        keep = inside
    else:
        keep = inside | (distance > max_distance)
    merged = np.where(keep[:, :, None], drew, base).astype(np.uint8)
    return Image.fromarray(merged, mode="RGB"), {
        "applied": True, "max_distance": max_distance, "region_only": bool(region_only),
        "changed_ratio": round(float(keep.mean()), 4)}


def edge_clean(mask, feather=None):
    """把掩版边界收紧到轮廓上，减少"分界不清"。

    羽化让过渡自然，但过宽的羽化会把边界摊成一片模糊带。这里先按阈值把羽化结果
    重新量化，让过渡带尽可能窄（只保留真正需要的几像素），再交给合成。
    """
    import numpy as np
    from PIL import Image
    alpha = feather_alpha(mask, feather=feather)
    array = np.asarray(alpha).astype(np.float32)
    # 把低于一个下限的值压成 0、高于一个上限的抬成 255，中间保留（过渡带因此变窄）
    low = 255 * config.EDGE_CLEAN_LOW
    high = 255 * config.EDGE_CLEAN_HIGH
    cleaned = np.clip((array - low) / max(1.0, (high - low)), 0.0, 1.0) * 255.0
    return Image.fromarray(cleaned.astype(np.uint8), mode="L")


def structure_condition(image):
    """源图结构条件（Canny）：保轮廓与构图，不锁死需要改变的服装内部纹理。"""
    return image_utils.extract_canny(image)


def hair_boundary(subject, profile) -> dict:
    """发型/发色的取舍记录：源图发型轮廓保留，角色发色迁移（明确写进审计）。"""
    observed_hair = bool(subject.get("visible_regions") and
                         any(str(r).lower() in {"head", "face", "hair"}
                             for r in subject.get("visible_regions") or []))
    target = profile_module.hair_color(profile)
    return {
        "keep_hairstyle_from_source": observed_hair,
        "target_hair_color_rgb": target,
        "applied": False,
        "note": ("源图发型轮廓保留；发色按角色档案迁移" if observed_hair
                 else "源图未观察到头部区域，发色迁移按角色档案整体进行"),
    }


def describe(package) -> str:
    """把条件包摘要成人类可读文本，用于运行日志与验收记录。"""
    settings = package["settings"]
    lines = [
        f"[条件] 角色={package['character']} 触发词={package['trigger']} "
        f"主体={package['subject'].get('subject_type')}",
        f"[条件] 权重={[w['file'] for w in package['weights']] or '（角色包内未声明）'}",
        f"[条件] 强度 img2img={settings['strength']} control={settings['control_scale']}"
        f"@{settings['control_end']} lora={settings['lora_scale']}",
        f"[条件] {package['style_clause']}",
    ]
    return "\n".join(lines)
