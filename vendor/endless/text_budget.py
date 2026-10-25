# -*- coding: utf-8 -*-
"""text_budget.py —— 提示词 token 预算（唯一实现）

为什么必须单独一个模块（2026-10-05 实测的教训）：
提示词预算在项目里散成了三处，各自用不同办法估算，于是**每一处都漏过**：

    * `subject_transfer` 用真 tokenizer 逐条裁剪，但只是"尽量"；
    * `train.py` 用真 tokenizer 检查，超了**直接抛错**——于是训练根本起不来；
    * 没有任何地方检查 `captions.txt` 里那些已经写进去的标注，
      结果 21 条训练标注到 176/77 token（超一倍多）。

把它集中到一个不依赖 torch 的小模块（只依赖 transformers 的 tokenizer，
取不到时退回字符估算并**明确标注不可靠**），三处共用同一份实现。

核心约定：
    * 计数取 SDXL 两个 CLIP tokenizer 的**较大值**（两个都要过）；
    * 中文与英文的 token/字符比差得很远（实测 21 个汉字 ≈ 21 token），
      因此任何"按字符数估算"的裁剪都必须用真 tokenizer 兜底；
    * 裁剪一律**按逗号条目**从后往前丢，丢掉的条目要能打印出来，不静默截断。
"""
import os

# 进程内缓存：`count` 在一次提示词构造里会被调用几十次，
# 每次都重新遍历 HuggingFace 缓存目录会明显变慢（实测每条候选条目一次）。
_TOKENIZER_CACHE = []


def reset_cache():
    """清空 tokenizer 缓存（测试用）。"""
    del _TOKENIZER_CACHE[:]


def tokenizers():
    """底模的两个 CLIP tokenizer；取不到时返回空列表（调用方应据此降级）。"""
    if _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[0]
    found = _scan()
    _TOKENIZER_CACHE.append(found)
    return found


def _scan():
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if not os.path.isdir(hub):
        return []
    try:
        from transformers import CLIPTokenizer, CLIPTokenizerFast
    except Exception:
        return []
    roots = []
    for name in sorted(os.listdir(hub)):
        snapshots = os.path.join(hub, name, "snapshots")
        if os.path.isdir(snapshots):
            roots.extend(os.path.join(snapshots, snap)
                         for snap in sorted(os.listdir(snapshots)))
    found = []
    for root in roots:
        if len(found) >= 2:
            break
        for cls, sub in ((CLIPTokenizer, "tokenizer"), (CLIPTokenizerFast, "tokenizer_2")):
            if any(existing[1] == sub for existing in found):
                continue
            try:
                found.append((cls.from_pretrained(root, subfolder=sub,
                                                  local_files_only=True), sub))
            except Exception:
                continue
    if len(found) < 2:
        return []
    return [item[0] for item in found]


def count(text) -> int:
    """text 的 token 数（两个 tokenizer 取较大值）。

    取不到 tokenizer 时退回 `词数 + 2`：这是**不可靠的**下界
    （中文会被严重低估），因此调用方在关键路径上应当先检查
    `tokenizers()` 是否为空，并把降级事实记进审计。

    `verbose=False` 是必需的：hf 的 tokenizer 在文本超过模型最大长度时会
    打印 "Token indices sequence length is longer than the specified maximum
    sequence length"，而本模块的职责**就是**数那些可能超长的中间串，
    于是每数一次就刷一行告警——实测把训练与推理日志淹掉，
    还让人误以为真的送进了超预算提示词（送进 CLIP 的都是裁剪后的结果）。
    """
    text = str(text or "")
    found = tokenizers()
    if not found:
        return len(text.split()) + 2
    counts = []
    for tokenizer in found:
        try:
            counts.append(len(tokenizer(text, truncation=False, verbose=False)["input_ids"]))
        except TypeError:
            # 少数 tokenizer 版本不支持 verbose 关键字
            counts.append(len(tokenizer(text, truncation=False)["input_ids"]))
        except Exception:
            continue
    return max(counts) if counts else len(text.split()) + 2


def fits(text, budget: int = 77) -> bool:
    return count(text) <= budget


def fit(text, budget: int = 77, keep_first: bool = False) -> tuple:
    """把逗号分隔的提示词裁到预算内；返回 (裁剪结果, 被丢弃的条目)。

    从**末尾**开始丢（末尾通常是质量词、构图词这类最可丢的项）。
    `keep_first=True` 时至少保住第一条：第一条装不下就按 token 逐字收缩
    （中文的字符数与 token 数差得远，按字符硬截会超预算）。
    """
    text = ", ".join(part.strip() for part in str(text or "").split(",") if part.strip())
    dropped = []
    if count(text) <= budget:
        return text, dropped
    parts = [part.strip() for part in text.split(",") if part.strip()]
    while parts and count(", ".join(parts)) > budget:
        dropped.append(parts.pop())
    if not parts and keep_first and text:
        # 连一条都放不下：二分找最长可容纳前缀
        head = text.split(",")[0].strip()
        low, high = 0, len(head)
        while low < high:
            mid = (low + high + 1) // 2
            if count(head[:mid]) <= budget:
                low = mid
            else:
                high = mid - 1
        return (head[:low].rstrip(" ,，、:：") if low > 0 else ""), dropped
    return ", ".join(parts), dropped


def join_within_budget(pieces, budget: int = 77, skip_oversized: bool = True) -> str:
    """按顺序把若干条目拼到预算内。

    `skip_oversized=True`（默认）时，**单条超预算就跳过该条**，继续尝试后面的短条目。
    这是实测踩出来的：早期"第一条超预算就终止整个拼接"会让整类条件悄悄变成空串
    （结构化后的标志描述单条可到 272 字符，而预算是 200，于是所有标志条件都成了空串，
    日志却完全正常）。
    """
    kept = []
    for piece in pieces:
        piece = str(piece or "").strip()
        if not piece:
            continue
        if skip_oversized and count(piece) > budget:
            continue
        candidate = ", ".join(kept + [piece])
        if count(candidate) > budget:
            continue
        kept.append(piece)
    return ", ".join(kept)
