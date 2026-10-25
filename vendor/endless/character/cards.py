# -*- coding: utf-8 -*-
"""character/cards.py —— 由角色档案派生生成条件（唯一实现）

档案（profile）描述"角色长什么样"；本模块回答"生成时用什么条件表达它"。
两者分开是为了让条件可以单独复核：条件错了不必重新观察角色。

被谁使用：
    subject_transfer.py  正式迁移流程
    web/server.py        本地角色工作台（训练和转化共用正式入口）
"""
import config
from character import profile as profile_module
from character import semantics


def build_card(profile) -> dict:
    """把档案编译成生成条件卡片。缺什么就少什么，不用别的角色的值兜底。

    clause      = 主（身份）条件：角色本身（面部/发色/配色/服装），主 LoRA 与推理使用
    concept     = 概念条件：配饰/图案/花纹/边饰，概念分支使用（不混进主条件）
    """
    trigger = config.get_trigger(profile.get("folder") or config.CHARACTER_ID)
    card = {
        "folder": profile.get("folder"),
        "trigger": trigger,
        "identity": profile.get("identity") or "",
        "clause": semantics.identity_clause(profile),
        "concept": semantics.concept_clause(profile),
        "hair_color": profile_module.hair_color(profile),
        "negative": "",
        "visual_style": profile.get("visual_style") or "",
        "coverage_gaps": semantics.coverage(profile),
    }
    negatives = []
    hair = card["hair_color"]
    if hair:
        # 只在确实观察到发色时才约束"错误的发色"，避免无依据的负向条件
        for name, rgb in profile_module.palette_colors(profile):
            if list(rgb) == list(hair):
                continue
            if name in {"black", "brown", "blonde", "yellow", "green", "purple"}:
                negatives.append(f"{name} hair")
    card["negative"] = ", ".join(dict.fromkeys(negatives[:4]))
    return card


def load_card(folder, allow_stale=True) -> dict:
    """读取该角色的条件卡片；没有档案时返回最小卡片（只含触发词），由调用方决定是否继续。"""
    profile = profile_module.load_profile(folder, allow_stale=allow_stale)
    if profile is None:
        return {"folder": folder, "trigger": config.get_trigger(folder), "clause": "",
                "concept": "", "hair_color": None, "negative": "", "identity": "",
                "missing_profile": True, "coverage_gaps": ["角色语义档案不存在"]}
    return build_card(profile)


def style_token(profile) -> str:
    """画风词：只描述角色参考图的表现媒介，不作为生成目标的画风（目标画风来自源图）。"""
    style = str(profile.get("visual_style") or "").lower()
    for key, token in (("pixel", "pixel art"), ("anime", "anime illustration"),
                       ("illustration", "illustration"), ("photo", "photograph"),
                       ("3d", "3d render"), ("sketch", "sketch"), ("line", "line art")):
        if key in style:
            return token
    return ""


def describe(card) -> str:
    lines = [f"角色条件卡片: {card.get('folder')}  触发词={card.get('trigger')}"]
    if card.get("identity"):
        lines.append(f"  身份: {card['identity']}")
    if card.get("clause"):
        lines.append(f"  主条件(身份): {card['clause'][:300]}")
    if card.get("concept"):
        lines.append(f"  概念条件(配饰/图案): {card['concept'][:300]}")
    if card.get("hair_color"):
        lines.append(f"  发色 RGB: {card['hair_color']}")
    if card.get("negative"):
        lines.append(f"  负向: {card['negative']}")
    for gap in card.get("coverage_gaps") or []:
        lines.append(f"  ! 证据缺口: {gap}")
    return "\n".join(lines)
