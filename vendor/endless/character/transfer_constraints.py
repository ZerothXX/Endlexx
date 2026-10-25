"""Source-visible composition and bounded prompt assembly (no model imports)."""
import re


def visible_regions(subject):
    values = subject.get("visible_regions") or []
    if not isinstance(values, list):
        values = [values]
    return {str(v).strip().lower().replace("_", " ") for v in values}


def region_visible(subject, region):
    framing = str(subject.get("framing") or "").lower()
    coverage = subject.get("coverage") or {}
    if region in {"legs", "feet"} and (framing in {"upper body", "bust", "portrait", "close-up"}
            or (isinstance(coverage, dict) and coverage.get("legs") == "out_of_frame")):
        return False
    regions = visible_regions(subject)
    if region in regions:
        return True
    aliases = {
        "head": {"face", "hair"},
        "torso": {"upper body", "chest", "shoulders", "body"},
        "chest": {"upper body", "torso", "bust"},
        "arms": {"upper body", "hands"},
        "waist": {"hips", "hip"},
        "legs": {"leg", "thighs", "knees", "lower legs"},
        "feet": {"foot", "shoes", "boots"},
    }
    # 'full body' permits clothing, but says nothing about exposed skin.
    return "full body" in regions or bool(regions & aliases.get(region, set()))


def geometry_terms(subject):
    """Observed source geometry, independent of the selected character."""
    terms=[]
    if subject.get('subject_type') != 'human':
        return terms
    if subject.get('proportions') == 'chibi':
        terms.append('chibi, large head, small body')
    views = {'front view','three-quarter view','side view','back view'}
    if subject.get('body_view') == subject.get('head_view') and subject.get('body_view') in views:
        terms.append('face and torso '+subject['body_view'])
        return terms
    for key,label in (('body_view','torso'),('head_view','face')):
        if subject.get(key) in views:
            terms.append(label+' '+subject[key])
    return terms


def composition_constraints(subject):
    """Use observed framing/coverage; unknown is never interpreted as bare skin."""
    positive, negative = [], []
    if subject.get("subject_type") != "human":
        return positive, negative
    positive.extend(geometry_terms(subject))
    framing = str(subject.get("framing") or "").lower()
    regions = visible_regions(subject)
    upper = framing in {"upper body", "bust", "portrait", "close-up"}
    upper = upper or (bool(regions) and not region_visible(subject, "legs")
                      and not region_visible(subject, "feet"))
    if upper:
        positive.append("upper body")
        negative.extend(["full body", "bare legs", "bare thighs", "feet"])
    coverage = subject.get("coverage") or {}
    if not isinstance(coverage, dict):
        coverage = {}
    if coverage.get("legs") == "covered":
        positive.append("covered legs")
        negative.extend(["bare legs", "bare thighs", "skirt lift"])
    if coverage.get("chest") == "covered":
        positive.append("covered chest")
        negative.extend(["cleavage", "open shirt"])
    hands = str(subject.get("hands") or "").lower()
    if hands in {"overlapping hands", "folded hands", "crossed hands"}:
        positive.append(hands)
    elif hands == "hidden":
        positive.append("hands out of view")
        negative.append("visible hands")
    return positive, negative


def fit_prompt(required, optional, tokenizers=(), max_tokens=77):
    """Never discard anatomy/symbol constraints to keep decorative quality words.

    Counts both SDXL tokenizers including BOS/EOS. An oversized required block
    fails explicitly instead of silently truncating essential conditions.

    **tokenizers 必须由调用方传入真实的两个 CLIP tokenizer。**
    实测踩过（2026-10-05）：不传时这里退回 `\\w+|[^\\s]` 的字符级估算，
    它对英文大致够用，但**严重低估中文**——标志描述里带着中文短名
    （"外袖短至手肘，内袖长至手背"），估算出 52，真实 tokenizer 数是 90，
    于是 `fit_prompt` 认为装得下、原样返回，最终交给 CLIP 的是一段超预算提示词。
    调用方现在通过 `subject_transfer._clip_tokenizers()` 传入真 tokenizer。
    """
    tokenizers = tuple(t for t in tokenizers if t is not None)
    def count(text):
        if tokenizers:
            return max(len(t(text, truncation=False, verbose=False)["input_ids"]) for t in tokenizers)
        return len(re.findall(r"\w+|[^\w\s]", text)) + 2
    parts = [p.strip() for p in required if p and p.strip()]
    if count(", ".join(parts)) > max_tokens:
        raise ValueError("Required transfer conditions exceed the CLIP token budget; "
                         "shorten the character accessory description or subject conditions.")
    dropped = []
    for phrase in optional:
        if not phrase or phrase in parts:
            continue
        if count(", ".join(parts + [phrase])) <= max_tokens:
            parts.append(phrase)
        else:
            dropped.append(phrase)
    # 兜底：required 之外的项都放完之后仍然超预算（只在估算与真实计数不一致时发生），
    # 从**末尾**开始丢掉可选项目，保证交给 CLIP 的一定在预算内。
    while len(parts) > len([p for p in required if p and p.strip()]) and \
            count(", ".join(parts)) > max_tokens:
        removed = parts.pop()
        dropped.append(removed)
    return ", ".join(parts), dropped
