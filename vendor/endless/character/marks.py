# -*- coding: utf-8 -*-
"""character/marks.py —— 统一角色标志库（唯一实现）

规格来源：docs/统一角色标志与自动证据提取_技术调整说明_20261004.md §3、§4。

用户的两份文本文件只是**同一种"角色标志"的两种输入方式**：
    dataset/<角色>/marks.txt       每行一个标志描述，搜索整个图集
    dataset/<角色>/mark_refs.txt   每行 "相对图片路径<TAB>描述"，单独标志参考图

进入本模块后统一成同一种数据结构，不设"通用标志模型"与"单独标志模型"，
也不因输入文件不同而赋予固定生成优先级。

分级语义（用户人工标注约定，本项目按此处理，不重新发明）：
    * 带人物部位指示的行  -> 该部位的观察记录（观察级线索，可能被遮挡或不在视野内）；
    * 不带部位指示的行    -> 角色级通用象征标志，贯穿整个设计，图集中必有。
    两种都是标志，统一进同一个库；区别只体现在 `importance`（前者偏观察记录，
    后者关乎"看到就知道是这个角色"）与 `scope`（whole / 具体部位）。
    少数不带部位指示的行也可能是普通通用标志（不太需要在意），因此重要性
    只是**默认倾向**，最终由证据与用户输入决定。

用户输入是中文大白话。结构化与英文化都交给项目内的 VLM（`vlm/vlm_utils.py`
的 MARK_STRUCTURE_PROMPT）；VLM 不可用时**不臆造**，只保留原文并记录缺口。

产出（用户不必手工编辑）：
    output/semantic/<角色>_marks.json        统一标志库
    output/semantic/<角色>_observations.jsonl 逐图证据（每个标志 × 每张图一行）
"""
import hashlib
import json
import os
import re

import config

# 逐图可见状态（规格 §4）。本项目只声明**确有依据**的状态，其余记 unknown：
#   可见证据 -> visible / partial；模型或规格明确判定 -> occluded / out_of_view / absent；
#   没有证据也没有判定 -> unknown（不回退成"遮挡"，那会把未知当已知）。
VISIBILITY_STATES = ("visible", "partial", "occluded", "out_of_view", "absent", "unknown")

# 标志分类（规格 §6：非人物承载方式按类别决定，不分"通用/单独"）
MARK_CLASSES = ("palette", "emblem", "pattern", "trim", "wearable", "material", "detail")

# 用户在两份文件里都可能写入"必有"这类设计层声明；只影响默认保留倾向，不覆盖图像级事实。
DECLARED_ALWAYS = ("必有", "一定要有", "always", "must have")

# 中文部位词 -> 内部区域名。**顺序即优先级**：
# "胸前"必须命中 chest 而不是 face（"脸"是子串），因此脸部只能由明确的"脸/颜"触发。
_ZH_REGION_RULES = (
    ("chest", ("前胸甲", "胸前", "胸口", "胸甲", "胸饰", "胸")),
    ("neck", ("脖子", "颈部", "颈")),
    ("waist", ("腰前", "腰间", "腰部", "腰带", "腰")),
    ("back", ("背后", "背部", "后背", "背面")),
    ("head", ("发髻", "头发", "发色", "发型", "发饰", "刘海", "头顶", "头")),
    ("face", ("脸颊", "脸部", "脸", "颜")),
    ("eyes", ("瞳色", "瞳孔", "眼睛", "眼")),
    ("arms", ("袖口", "袖子", "外袖", "内袖", "袖", "手臂", "胳膊", "手背", "手腕")),
    ("hands", ("手部", "手指", "手持", "手")),
    ("feet", ("木屐", "鞋", "足", "脚")),
    ("legs", ("长裙", "裙", "腿", "膝")),
    ("torso", ("上衣", "衣身", "躯干", "身")),
    ("surface", ("表面", "材质", "纹样", "花纹", "图案")),
    ("whole", ("整体", "全身", "整幅")),
)

# 配色/色彩构成类描述天然是**角色级**的，不属于任何单一部位。
# 不加这一层的话，"颜色由淡蓝和深蓝组成"会因为含"脸"（"颜色"里没有脸，
# 但描述里出现"粉"等字）之类的子串被误判到某个部位（实测踩过）。
_PALETTE_TOKENS = ("配色", "颜色", "色彩", "色调", "colour scheme", "color scheme",
                   "colour", "color")

# 标志类别关键词。**顺序即优先级**：先"是什么东西"，再"上面有什么纹样"。
# 反过来的话，"中心带图案的黑色前胸甲"会因为含"图案"被判成边饰（实测踩过），
# 而它其实是一件胸前甲（emblem）。
_CLASS_RULES = (
    ("emblem", ("徽记", "徽章", "纹章", "前胸甲", "胸甲", "中心带图案", "emblem", "crest")),
    ("wearable", ("项圈", "项链", "挂饰", "吊坠", "配饰", "蝴蝶结", "发饰",
                  "扇", "木屐", "袜", "鞋")),
    ("material", ("材质", "质感", "缎", "绸", "棉", "丝")),
    ("pattern", ("樱花", "祥云", "云纹", "花纹", "波浪", "波纹", "图案",
                 "pattern", "sakura", "motif")),
    ("trim", ("边饰", "包边", "窄边", "金线", "镶边", "绳", "结", "缘", "纹")),
)

# ---- 标志的两种实体形态（规格 §5：不是所有条目都能"框出来"，但都能"看出来"）----
# locatable : 有边界的实体标志（徽记、配饰、图案、花纹、边饰）——逐图查询位置 + 可见状态，
#             可见的像素才是正向外观监督（规格 §4）。
# describable: 没有边界的外观属性/构成（配色、发色、瞳色、发型剪裁）——逐图查询可见状态
#             与可见描述，但不产生位置框，也不进入"裁剪样本"。
# 用户的分级约定（前几条通用象征标志 / 后面各部位观察）与本字段是两件事：
# 前者说"是不是角色级通用标志"，后者说"能不能被框出来"。两者都保留。
_ENTITY_MODES = {"locatable", "describable"}

# 明确属于"外观属性"而不是"有边界的标志"的关键词
_TRAIT_TOKENS = ("配色", "颜色", "色彩", "色调", "发色", "瞳色", "发髻", "马尾",
                 "发型", "刘海", "发长", "剪", "质", "colour", "color")

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_ALLOWED_TARGET_ROLES = ("emblem", "wearable", "prop", "pattern", "palette", "unknown")


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def dataset_dir(folder) -> str:
    return os.path.join(config.DATASET_DIR, folder)


def marks_text_path(folder) -> str:
    return os.path.join(dataset_dir(folder), "marks.txt")


def mark_refs_path(folder) -> str:
    return os.path.join(dataset_dir(folder), "mark_refs.txt")


def library_path(folder) -> str:
    return os.path.join(config.SEMANTIC_DIR, f"{folder}_marks.json")


def observations_path(folder) -> str:
    return os.path.join(config.SEMANTIC_DIR, f"{folder}_observations.jsonl")


def reference_crop_dir(folder) -> str:
    return os.path.join(config.OUTPUT_DIR, "marks", folder)


# ---------------------------------------------------------------------------
# 解析（容错：用户的文件可能用制表符、空格或全角空格分隔）
# ---------------------------------------------------------------------------
def _strip_comment(line: str) -> str:
    """去掉行内注释与首尾空白；# 开头整行注释；BOM 与全角空格一并清掉。"""
    text = line.replace("\ufeff", "").replace("\u3000", " ").strip()
    if not text or text.startswith("#"):
        return ""
    return text


def parse_mark_lines(folder) -> dict:
    """读取 marks.txt：每行一个标志描述。返回 {marks, warnings}。

    同一文件内的重复行**不在这里复制成两条**：重复代表强调，由
    `merge_into_library` 统一按"强调增量有上限"的规则处理（规格 §4）。
    """
    path = marks_text_path(folder)
    marks, warnings = [], []
    if not os.path.isfile(path):
        return {"marks": marks, "warnings": [f"没有 marks.txt（{path}）：跳过用户声明的标志"],
                "present": False}
    seen = set()
    with open(path, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            text = _strip_comment(line)
            if not text:
                continue
            if text in seen:
                warnings.append(f"marks.txt 第 {line_no} 行与前面重复，按强调处理: {text}")
            seen.add(text)
            marks.append({"description": text, "source": "marks.txt", "line": line_no})
    if not marks:
        warnings.append(f"marks.txt 里没有有效描述行（空文件或全是注释）: {path}")
    return {"marks": marks, "warnings": warnings, "present": True}


def resolve_reference(folder, rel_path: str):
    """解析 mark_refs.txt 里的相对路径；返回绝对路径或 None（解析不到就报缺口，不猜）。

    先按原样解析；解析不到时做一次**保守**的等价重试：同一目录下、文件名主干去掉
    前导零后相同、扩展名相同。用户手写的引用常写成 `images/021.png`，而磁盘上是
    `21.png`——两者指向同一张图，属于同一引用的写法差异，不是"猜另一个文件"。
    """
    text = str(rel_path or "").replace("\\", "/").strip()
    if not text:
        return None
    root = dataset_dir(folder)
    direct = os.path.join(root, text.replace("/", os.sep))
    if os.path.isfile(direct):
        return os.path.abspath(direct)
    # 等价重试：同一目录下、主干数字去前导零后相同、扩展名相同
    folder_part, _, name = text.rpartition("/")
    stem, ext = os.path.splitext(name)
    if not re.fullmatch(r"0*\d+", stem or ""):
        return None
    directory = os.path.join(root, folder_part.replace("/", os.sep)) if folder_part else root
    if not os.path.isdir(directory):
        return None
    wanted = str(int(stem))
    # 先按相同扩展名找（最保守）
    for entry in sorted(os.listdir(directory)):
        other_stem, other_ext = os.path.splitext(entry)
        if other_ext.lower() != ext.lower():
            continue
        if re.fullmatch(r"0*\d+", other_stem) and str(int(other_stem)) == wanted:
            return os.path.abspath(os.path.join(directory, entry))
    # 再按"主干数字相同、扩展名是**同一张图的另一种写法**"找。
    #
    # 为什么需要（实测事故）：用户写的 `mark_refs.txt` 是
    # `images/005.png 扇子`，而 `dataset/ayaka/images` 里实际是 `005.jpg`——
    # **同一张图的扩展名写法不同**（png/jpg/webp 混用），不是"猜另一张图"。
    # 只按相同扩展名匹配时，这四条参考图里有两条解析不到，
    # 连带把整库指纹改掉、标志链整条失效（出图上什么标志都没有）。
    # 图片扩展名集合是封闭的，因此这一步仍然是确定性的、不会误配。
    image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    if ext.lower() in image_exts:
        for entry in sorted(os.listdir(directory)):
            other_stem, other_ext = os.path.splitext(entry)
            if other_ext.lower() not in image_exts:
                continue
            if re.fullmatch(r"0*\d+", other_stem) and str(int(other_stem)) == wanted:
                return os.path.abspath(os.path.join(directory, entry))
    return None


def parse_mark_refs(folder) -> dict:
    """读取 mark_refs.txt：每行 "相对图片路径<TAB>描述"。

    分隔符按制表符、全角空格、连续空格依次尝试（用户的文件用空格分隔），
    描述里本身允许有空格，因此**只在第一个分隔处切开**。
    每行参考图与描述构成一条确定的配对；解析不到图片就如实报告，不静默丢弃。
    """
    path = mark_refs_path(folder)
    refs, warnings, unresolved = [], [], []
    if not os.path.isfile(path):
        return {"refs": refs, "warnings": [], "unresolved": [], "present": False}
    with open(path, "r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            text = _strip_comment(line)
            if not text:
                continue
            if "\t" in text:
                raw_path, _, description = text.partition("\t")
            else:
                match = re.match(r"^(\S+)[ \u3000]+(.*)$", text)
                if match:
                    raw_path, description = match.group(1), match.group(2)
                else:
                    raw_path, description = text, ""
                if not description:
                    warnings.append(f"mark_refs.txt 第 {line_no} 行只有路径没有描述，已记录缺口")
            raw_path, description = raw_path.strip(), description.strip()
            resolved = resolve_reference(folder, raw_path)
            if resolved is None:
                unresolved.append({"line": line_no, "rel_path": raw_path,
                                   "reason": "dataset 目录下找不到这张图（含去前导零的等价写法）"})
                continue
            refs.append({"rel_path": raw_path,
                         "resolved_rel_path": os.path.relpath(resolved, dataset_dir(folder))
                         .replace("\\", "/"),
                         "abs_path": resolved, "description": description, "line": line_no,
                         # 把"实际解析到哪张图"一并带出：跨扩展名解析（`005.png` -> `005.jpg`）
                         # 之后，只留原始 rel_path 的话，下游按它去拼路径仍然找不到文件
                         # （实测：单独标志参考图 4 条继续报"参考图不存在"）。
                         "exists": True,
                         "resolved_ext": os.path.splitext(resolved)[1].lower()})
    return {"refs": refs, "warnings": warnings, "unresolved": unresolved, "present": True}


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------
def normalize_regions(value, description="") -> list:
    """把部位说法规范成内部区域名列表（**第一个是主区域**）。

    一行描述可以同时点到多个部位："淡蓝色上衣，深蓝色宽的长裙至膝盖" 是 torso + legs。
    只取一个区域会丢掉另一半信息，也会让后面的"位置是否可信"检查误杀正确证据。
    """
    combined = f"{value or ''} {description or ''}"
    if any(token in combined.lower() for token in _PALETTE_TOKENS):
        return ["whole"]                 # 配色类描述天然是角色级的
    found = []
    for source in (value, description):
        text = str(source or "").strip()
        if not text:
            continue
        for region, keywords in _ZH_REGION_RULES:
            if region in found:
                continue
            if any(keyword in text for keyword in keywords):
                found.append(region)
        plain = re.sub(r"[^a-z_]", "", text.lower())
        if plain in config.MARK_REGIONS and plain not in found:
            found.append(plain)
    # whole 只是"哪里都可能有"的兜底，不能挤掉已经识别出的具体部位
    specific = [region for region in found if region != "whole"]
    if specific:
        return specific
    return found or ["whole"]


def normalize_region(value, description="") -> str:
    """主区域（`normalize_regions` 的第一个结果）。"""
    return normalize_regions(value, description)[0]


def entity_mode(description: str, region: str) -> str:
    """这条描述是"能被框出来的实体标志"还是"没有边界的外观属性"。"""
    text = str(description or "")
    if any(token in text.lower() for token in _TRAIT_TOKENS):
        return "describable"
    return "locatable"


def classify_mark(description: str, region: str) -> str:
    """默认标志分类：配色类先判，再按已见关键词判断，命中不了就是 detail（不猜）。"""
    text = str(description or "")
    if any(token in text.lower() for token in _PALETTE_TOKENS):
        return "palette"
    for mark_class, keywords in _CLASS_RULES:
        if any(keyword in text for keyword in keywords):
            return mark_class
    return {"feet": "wearable", "head": "wearable",
            "neck": "wearable", "waist": "wearable",
            "chest": "pattern", "surface": "pattern"}.get(region, "detail")


def canonical_name(description: str, limit: int = 18) -> str:
    """规范化短名：取描述的第一个短句并按标点/空格收口（用户原文优先，不改写语义）。"""
    text = re.split(r"[，,。；;（(]", str(description or "").strip())[0]
    text = text.strip()
    return text[:limit] if len(text) > limit else text


def default_importance(region: str, declared: bool) -> str:
    """默认重要性倾向（可被证据与用户输入覆盖）。

    不带人物部位指示的行是角色级通用象征标志，更可能"看到就知道是这个角色"；
    带部位指示的行主要是人物各部位的观察记录。两者都进同一个库、同等对待，
    差别只在这个默认倾向上；用户显式写"必有"则直接提到 high。
    """
    if declared:
        return "high"
    return "high" if region == "whole" else "medium"


def declared_always(description: str) -> bool:
    text = str(description or "").lower()
    return any(token.lower() in text for token in DECLARED_ALWAYS)


def _mark_id(description: str, index: int) -> str:
    """可复现的短 ID：由规范化描述哈希派生，不用行号（重排文件不改变身份）。"""
    digest = hashlib.sha1(str(description).strip().encode("utf-8")).hexdigest()[:8]
    return f"m{index:02d}_{digest}"


def mark_token(mark, trigger=None) -> str:
    """该标志**专属的触发词**，训练与推理必须用同一个（这一条是概念能否被唤起的关键）。

    为什么必须每个标志一个专属词（实测定位，代价很大）：
    概念分支原先所有标志都用**同一个角色触发词**训练
    （`chayaka, masterpiece, best quality, <该标志的描述>`），
    于是 LoRA 学到的是"`chayaka` -> 这块裁片"，**没有任何一个标志与自己的描述绑定**。
    推理时提示词写的是
    `the character's real marks on the object surface, this real repeating
    pattern: light purple three concentric circles…`——
    里面既**没有** `chayaka`，又多了一串训练时没见过的词。
    结果概念权重根本不激活：出图只是一层很淡的配色渐变，
    看起来像"权重没学到"，实际是**条件对不上**。
    更隐蔽的是训练 loss 一路降到 0.0007：图像能被共享触发词解释，
    所以 loss 好看，但"每个标志可被单独点名"这个真正的目标从未达成。

    形式：角色触发词 + 标志序号（如 `chayaka_m02`），
    保证**唯一、稳定、可读**，且不依赖描述文本（描述改了 token 不变）。
    """
    token = str(mark.get("token") or "").strip()
    if token:
        return token
    prefix = str(trigger or config.get_trigger("")).strip() or "mark"
    mark_id = str(mark.get("mark_id") or "").strip()
    match = re.match(r"m(\d+)", mark_id)
    suffix = f"_{int(match.group(1)):02d}" if match else ""
    return f"{prefix}{suffix}" if suffix else prefix


def ensure_tokens(library, trigger=None) -> int:
    """给库里每条标志补上 `token` 字段（缺则补，已有则保留）。返回补了几条。

    落盘保存，保证训练与推理读到的是**同一个** token（而不是各自现算）。
    """
    added = 0
    for mark in (library or {}).get("marks") or []:
        if not str(mark.get("token") or "").strip():
            mark["token"] = mark_token(mark, trigger)
            added += 1
    return added


def _similarity(left: str, right: str) -> float:
    """两段中文描述的字符二元组 Jaccard 相似度（0~1）。"""
    def grams(text):
        text = re.sub(r"[\s，,。；;、（）()]+", "", str(text or ""))
        if len(text) < 2:
            return {text} if text else set()
        return {text[i:i + 2] for i in range(len(text) - 1)}
    a, b = grams(left), grams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


# 颜色词（中英）：颜色不同就是**不同的标志**。
# 规格 §4 明确要求"不能把两个不同的蝴蝶结误合并"——白樱花与黑樱花正是这种情形，
# 它们的描述重合度很高（0.6+），只靠文本相似度必然误合并（实测踩过）。
_COLOR_TOKENS = (
    "白色", "黑色", "金色", "银色", "红色", "蓝色", "淡蓝", "深蓝", "绿色", "黄色",
    "粉色", "紫色", "灰色", "灰白", "棕色", "米色", "橙色", "青色", "白", "黑", "金",
    "银", "红", "蓝", "绿", "黄", "粉", "紫", "灰", "棕",
    "white", "black", "gold", "golden", "silver", "red", "blue", "green", "yellow",
    "pink", "purple", "grey", "gray", "brown", "orange", "cyan", "beige",
)


def _colors(text) -> set:
    lowered = str(text or "").lower()
    return {token for token in _COLOR_TOKENS if token in lowered}


def _has_cjk(text) -> bool:
    """文本里是否含中日韩字符（用于判断"这条描述有没有英文版"）。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _attribute_conflict(left: str, right: str) -> list:
    """两份描述里互斥的属性（目前只判颜色，但它是最常见也最危险的一类）。

    返回冲突说明列表；非空即表示**不能合并**。
    """
    conflicts = []
    left_colors, right_colors = _colors(left), _colors(right)
    if left_colors and right_colors and not (left_colors & right_colors):
        conflicts.append(f"颜色不同：{sorted(left_colors)} vs {sorted(right_colors)}")
    return conflicts


def merge_into_library(folder, entries, references) -> dict:
    """把两种入口的标志合并成统一库。

    规则（规格 §4）：
      * 同一标志在两份文件里重复 -> 合并为一个 mark_id，保留全部来源，并**增加重要性**；
        不复制成两个生成对象，也不靠重复文本无限放大权重（强调增量有上限）。
      * 相似名称**不足以**判定重复：需要部位、形状、描述同时接近，否则保留候选关系。
      * 单独标志图与描述是一对确定的证据配对；它允许归入已有标志，也允许自成一类。
    """
    marks = []
    for entry in entries:
        description = entry["description"]
        regions = normalize_regions("", description)
        region = regions[0]
        duplicate = next((m for m in marks
                          if _similarity(description, m["description"]) >= 0.999), None)
        if duplicate is not None:
            # 同一文件内重复 = 强调（有界增量），不复制成第二个生成对象
            duplicate["sources"].append({"kind": "marks.txt", "line": entry.get("line"),
                                         "text": description})
            duplicate["emphasis_sources"] = min(int(config.MARK_EMPHASIS_MAX_SOURCES),
                                                duplicate["emphasis_sources"] + 1)
            duplicate["user_importance"] = "high"
            continue
        mark = {
            "mark_id": _mark_id(description, len(marks) + 1),
            "name": canonical_name(description),
            "description": description,
            "description_zh": description,
            "aliases": [],
            "region": region,
            "regions": regions,
            "scope": "whole" if region == "whole" else "part",
            "entity_mode": entity_mode(description, region),
            "mark_class": classify_mark(description, region),
            "outline": "", "internal_topology": "", "color_relation": "",
            "repeat_unit": "", "material": "", "relative_scale": "",
            "appearance_source": "unstructured",
            "default_presence": ("always" if region == "whole"
                                 else "unknown"),
            "user_importance": default_importance(region, declared_always(description)),
            "evidence_confidence": "unverified",
            "emphasis_sources": 1,
            "sources": [{"kind": "marks.txt", "line": entry.get("line"),
                         "text": description}],
            "evidence": [],
            "migration": {"origin_region": region, "target_role": "unknown",
                          "allowed_deformation": True,
                          "count_constraint": "", "direction_constraint": ""},
        }
        marks.append(mark)

    warnings, links = [], []
    for reference in references:
        description = reference["description"] or canonical_name(reference["rel_path"])
        best, best_score = None, 0.0
        for mark in marks:
            score = max(_similarity(description, mark["description"]),
                        _similarity(description, mark["name"]))
            if score > best_score:
                best, best_score = mark, score
        # 相似名称/相似描述都不足以判定重复：还要属性不冲突（规格 §4）。
        conflicts = _attribute_conflict(description, best["description"]) if best else []
        if conflicts:
            warnings.append(
                f"标志「{description}」与「{best['name']}」描述相似（{best_score:.2f}）"
                f"但属性冲突（{'；'.join(conflicts)}），按**不同标志**保留，不合并")
        if best is not None and best_score >= config.MARK_MERGE_SIMILARITY and not conflicts:
            target = best
            links.append({"rel_path": reference["rel_path"], "mark_id": target["mark_id"],
                          "similarity": round(best_score, 3), "merged": True})
            target["sources"].append({"kind": "mark_refs.txt", "line": reference["line"],
                                      "text": description,
                                      "rel_path": reference["rel_path"]})
            target["aliases"] = list(dict.fromkeys(target["aliases"] + [description]))
            target["emphasis_sources"] = min(int(config.MARK_EMPHASIS_MAX_SOURCES),
                                             target["emphasis_sources"] + 1)
            target["user_importance"] = "high"
            target["evidence_confidence"] = "reference_image"
            target["appearance_source"] = "reference_image"
        else:
            mark = {
                "mark_id": _mark_id(description, len(marks) + 1),
                "name": canonical_name(description),
                "description": description,
                "description_zh": description,
                "aliases": [],
                "region": "whole",
                "scope": "isolated",
                "entity_mode": entity_mode(description, "whole"),
                "mark_class": classify_mark(description, "whole"),
                "outline": "", "internal_topology": "", "color_relation": "",
                "repeat_unit": "", "material": "", "relative_scale": "",
                "appearance_source": "reference_image",
                "default_presence": "unknown",
                "user_importance": "medium",
                "evidence_confidence": "reference_image",
                "emphasis_sources": 1,
                "sources": [{"kind": "mark_refs.txt", "line": reference["line"],
                             "text": description, "rel_path": reference["rel_path"]}],
                "evidence": [],
                "migration": {"origin_region": "whole", "target_role": "unknown",
                              "allowed_deformation": True,
                              "count_constraint": "", "direction_constraint": ""},
            }
            mark["regions"] = ["whole"]
            marks.append(mark)
            links.append({"rel_path": reference["rel_path"], "mark_id": mark["mark_id"],
                          "similarity": round(best_score, 3), "merged": False,
                          "reason": ("属性冲突，" + "；".join(conflicts)) if conflicts
                          else "与既有标志的描述重合度不足，保留为独立标志"})
            if best is not None and best_score > 0 and not conflicts:
                warnings.append(
                    f"标志「{description}」与「{best['name']}」相似度 {best_score:.2f}，"
                    f"低于阈值 {config.MARK_MERGE_SIMILARITY}，按独立标志保留候选关系")
    return {"marks": marks, "links": links, "warnings": warnings}


def parse_marks(folder) -> dict:
    """解析两份文本文件并合并成统一标志库（不读图、不加载模型）。"""
    text_result = parse_mark_lines(folder)
    refs_result = parse_mark_refs(folder)
    merged = merge_into_library(folder, text_result["marks"], refs_result["refs"])
    warnings = list(text_result["warnings"]) + list(refs_result["warnings"]) + \
        list(merged["warnings"])
    library = {
        "schema": 1,
        "character": folder,
        "mark_model": "vlm" if config.USE_VLM else "offline",
        "label_rule_version": config.LABEL_RULE_VERSION,
        "prompt_version": config.MARK_PROMPT_VERSION,
        "files": {"marks_txt": os.path.isfile(marks_text_path(folder)),
                  "mark_refs_txt": os.path.isfile(mark_refs_path(folder))},
        "marks": merged["marks"],
        "links": merged["links"],
        "references": [{k: v for k, v in ref.items() if k != "abs_path"}
                       for ref in refs_result["refs"]],
        "unresolved_refs": refs_result["unresolved"],
        "warnings": warnings,
        "unresolved_gaps": [],
    }
    # **必须落盘输入指纹**：`load_library` 会拿它判"用户改了 marks.txt 之后旧库是否失效"，
    # 而早期只在读取端比较、写入端从来不写，于是这个检查永远不触发——
    # 改了用户输入却继续用旧库，属于静默错误（实测发现）。
    library["input_fingerprint"] = input_fingerprint(folder)
    return library


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------
def save_library(folder, library) -> str:
    os.makedirs(config.SEMANTIC_DIR, exist_ok=True)
    # 落盘前补齐每条标志的专属触发词：训练与推理必须读到**同一个** token，
    # 而不是各自现算（现算一旦规则改动，旧权重就再也唤不起来）。
    ensure_tokens(library)
    # **指纹必须与实际写入的内容一致**：`load_library` 会拿它判"用户改了
    # marks.txt 之后旧库是否失效"，而早期写入端从来不写这个字段，
    # 于是检查永远不触发（改了用户输入却继续用旧库）；
    # 而只在一个构造路径里写、其他路径不写，又会造成"刚存的库读回来就过期"。
    # 因此统一在这里刷新：内容与指纹一起落盘，任何写入路径都自洽。
    library["input_fingerprint"] = input_fingerprint(folder)
    # 一并保存两份描述文件的哈希：过期时才能说出"是哪个文件的内容变了"，
    # 而不是只给一句布尔值（实测：过期了却打印"与当前输入不一致（0 项）"）。
    library["file_hashes"] = file_hashes(folder)
    path = library_path(folder)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(library, stream, ensure_ascii=False, indent=2)
    os.replace(temp, path)
    print(f"[marks] 统一标志库已保存: {path}（{len(library.get('marks') or [])} 个标志）")
    return path


def fingerprint_difference(folder, library) -> list:
    """逐项指出标志库与当前输入的差异（给人看的原因，不是布尔值）。

    为什么要"逐项"（实测事故）：`input_fingerprint` 是一个整体哈希，
    过期时只能打印一句"输入文件或提示词变化"，**看不出是哪一项**。
    实测就是因为 `mark_refs.txt` 里的 `images/005.png` 在磁盘上其实是 `005.jpg`，
    解析结果一变、参考图哈希跟着变、整体指纹就不同了——
    而这种"输入缺失"与"用户真的改了要求"必须区分开：前者要恢复输入，后者才该重建。
    """
    reasons = []
    for path in (marks_text_path(folder), mark_refs_path(folder)):
        name = os.path.basename(path)
        recorded = (library.get("files") or {}).get(
            "marks_txt" if name == "marks.txt" else "mark_refs_txt")
        if not os.path.isfile(path) and recorded:
            reasons.append(f"{name} 已不存在（建库时存在）")
        elif os.path.isfile(path) and recorded is False:
            reasons.append(f"{name} 新建（建库时不存在）")
    # **逐文件哈希对比**：上面只比"存在与否"，内容被改（例如用户往 marks.txt
    # 里补了一行）时一条差异都给不出来，于是 `stale_reasons` 为空、
    # 调用方看不出到底什么变了（实测：过期了却打印"与当前输入不一致（0 项）"）。
    recorded_hashes = library.get("file_hashes") or {}
    for path in (marks_text_path(folder), mark_refs_path(folder)):
        name = os.path.basename(path)
        if not os.path.isfile(path):
            continue
        current = _file_sha256(path)[:16]
        previous = recorded_hashes.get(name)
        if previous and previous != current:
            reasons.append(f"{name} 内容已变化（{previous} -> {current}）")
        elif not previous:
            reasons.append(f"{name} 没有可比对的历史哈希（库由更早版本写入）")
    if not reasons:
        # 兜底：指纹不同但上面每一项都看不出来，说明差异来自参考图字节或 VLM 路径
        reasons.append("输入文件与参考图均未检出差异，差异可能来自参考图像素内容或 VLM 路径变化")
    parsed = parse_mark_refs(folder)
    if parsed.get("unresolved"):
        for item in parsed["unresolved"]:
            reasons.append(f"mark_refs.txt 第 {item['line']} 行的参考图解析不到: "
                           f"{item['rel_path']}（{item['reason']}）")
    recorded_refs = {str(r.get("resolved_rel_path")) for r in (library.get("references") or [])}
    current_refs = {str(r.get("resolved_rel_path")) for r in (parsed.get("refs") or [])}
    for missing in sorted(recorded_refs - current_refs):
        reasons.append(f"参考图 {missing} 现在解析不到（建库时有）")
    for added in sorted(current_refs - recorded_refs):
        reasons.append(f"参考图 {added} 是新增的（建库时没有）")
    if str(library.get("prompt_version")) != str(config.MARK_PROMPT_VERSION):
        reasons.append(f"定位提示词版本变化: {library.get('prompt_version')} -> "
                       f"{config.MARK_PROMPT_VERSION}")
    return reasons


def load_library(folder, allow_stale=True):
    """读取标志库；不存在时返回 None。

    **过期时默认仍然返回库，并大声报告差异**（`allow_stale=False` 可强制返回 None）。

    为什么改成"降级使用"（实测事故，必须记录）：
    原先过期直接返回 `None`，于是整条标志链**静默失效**——
    实测出图上杯子什么标志都没有，而日志里只有一句
    "标志库已过期（输入文件或提示词变化），需要重建"，
    看不出"承载整条被跳过"。更糟的是这个过期**不是用户改了要求**：
    `mark_refs.txt` 写的是 `images/005.png`，而磁盘上是 `005.jpg`，
    解析结果一变、参考图哈希跟着变、整体指纹就不同了。

    区分两种情况很重要：
      * 输入"缺失/解析不到" -> 库里的 21 条标志描述与 462 条逐图证据**仍然有效**，
        降级使用远好于整条失效；
      * 用户真的改了 `marks.txt` -> `allow_stale=False` 的调用方会拿到 None 去重建。
    无论哪种，都把**具体差异**打印出来，不再只给一句笼统的"已过期"。
    """
    path = library_path(folder)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            library = json.load(stream)
    except (OSError, ValueError) as exc:
        print(f"[marks] 标志库损坏，忽略: {path} ({exc})")
        return None
    if not library.get("marks") and not (library.get("files") or {}).get("marks_txt"):
        return None
    recorded = library.get("input_fingerprint")
    if recorded and recorded != input_fingerprint(folder):
        reasons = fingerprint_difference(folder, library)
        print(f"[marks] 标志库与当前输入不一致（{len(reasons)} 项）:")
        for reason in reasons[:8]:
            print(f"[marks]   - {reason}")
        if not allow_stale:
            print("[marks] 调用方要求必须一致 -> 返回空，需要重建")
            return None
        print(f"[marks] 降级使用已有标志库（{len(library.get('marks') or [])} 条标志）；"
              f"如需完全一致请重建")
        library = dict(library, stale=True, stale_reasons=reasons)
    # **参考图引用始终按当前输入重新解析**（刻意放在"是否过期"分支之外）。
    #
    # 实测事故（两轮才修对）：`mark_refs.txt` 写的是 `images/005.png`，
    # 而磁盘上是 `005.jpg`。跨扩展名解析修好之后**整体指纹也跟着变了**
    # （参考图字节进了哈希），于是 `recorded == current`、代码走"一致"这条路，
    # 原本写在过期分支里的"刷新引用"被整个跳过；库里存的旧值
    # （`images/005.png`、`images/21.png`）继续被下游使用，
    # 表现为"上游修好了、4 条单独标志参考图仍然全部报不存在"。
    # 结论：引用是**当前输入的投影**，每次读取都重新算，不在库里留副本当权威。
    try:
        parsed = parse_mark_refs(folder)
        if parsed.get("refs") or parsed.get("unresolved"):
            refreshed = []
            old_by_rel = {str(r.get("rel_path")): r
                          for r in (library.get("references") or [])}
            for ref in parsed["refs"]:
                previous = dict(old_by_rel.get(str(ref.get("rel_path"))) or {})
                previous.update(ref)        # **当前解析结果优先**
                refreshed.append(previous)
            before = [r.get("resolved_rel_path") for r in (library.get("references") or [])]
            after = [r.get("resolved_rel_path") for r in refreshed]
            if before != after:
                print(f"[marks] 参考图引用已按当前输入刷新: {len(refreshed)} 条"
                      f"（{before} -> {after}）")
            library["references"] = refreshed
            library["unresolved_refs"] = parsed.get("unresolved") or []
    except Exception as exc:      # noqa: BLE001
        print(f"[marks] 参考图引用刷新失败（沿用库内记录）: {type(exc).__name__}: {exc}")
    return library


def _file_sha256(path) -> str:
    """文件的 sha256（读不了就返回空串，不抛）。"""
    try:
        with open(path, "rb") as stream:
            return hashlib.sha256(stream.read()).hexdigest()
    except OSError:
        return ""


def input_fingerprint(folder) -> str:
    """两份输入文件 + 标志参考图 + 提示词版本的指纹（进缓存键，规格 §2 结尾要求）。"""
    digest = hashlib.sha256()
    for path in (marks_text_path(folder), mark_refs_path(folder)):
        digest.update(os.path.basename(path).encode("utf-8"))
        if os.path.isfile(path):
            with open(path, "rb") as stream:
                digest.update(stream.read())
        else:
            digest.update(b"<missing>")
    refs = parse_mark_refs(folder)
    for ref in refs["refs"]:
        digest.update(ref["resolved_rel_path"].encode("utf-8"))
        try:
            with open(ref["abs_path"], "rb") as stream:
                digest.update(stream.read())
        except OSError:
            digest.update(b"<unreadable>")
    digest.update(str(config.MARK_PROMPT_VERSION).encode("utf-8"))
    digest.update(str(config.VLM_PATH).encode("utf-8"))
    return digest.hexdigest()


def file_hashes(folder) -> dict:
    """两份描述文件的 sha256 前 16 位，随库一起保存。

    用途：过期时能说出"是哪个文件的内容变了"，而不是只给一句
    "输入文件或提示词变化"（实测：过期了却打印"与当前输入不一致（0 项）"）。
    """
    result = {}
    for path in (marks_text_path(folder), mark_refs_path(folder)):
        if os.path.isfile(path):
            result[os.path.basename(path)] = _file_sha256(path)[:16]
    return result


def save_observations(folder, observations) -> str:
    os.makedirs(config.SEMANTIC_DIR, exist_ok=True)
    path = observations_path(folder)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as stream:
        for row in observations:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)
    print(f"[marks] 逐图证据已保存: {path}（{len(observations)} 条）")
    return path


def load_observations(folder) -> list:
    path = observations_path(folder)
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def evidence_by_image(observations) -> dict:
    """{rel_path: {mark_id: observation}}，供数据集构造裁剪样本与条件。"""
    table = {}
    for row in observations or []:
        rel_path = row.get("rel_path")
        mark_id = row.get("mark_id")
        if not rel_path or not mark_id:
            continue
        table.setdefault(rel_path, {})[mark_id] = row
    return table


def image_observations(folder) -> dict:
    """{rel_path: {"view":..., "framing":..., "tags":..., "visibility_note":...}}。

    逐图观察覆盖**全部有效原图**（不是只挑代表图）；没有 VLM 时退化为
    captions.txt 的既有标签，并如实记录来源，不编造视角。
    """
    rows = {}
    captions = {}
    try:
        import dataset as dataset_module
        captions = dataset_module.load_character_captions(folder)
        paths = dataset_module.scan_images(folder)
    except Exception as exc:                      # 目录缺失等：如实返回空表
        print(f"[marks] 逐图观察不可用（{type(exc).__name__}: {exc}）")
        return rows
    stored = {}
    for row in load_observations(folder):
        stored.setdefault(row.get("rel_path"), {})[row.get("mark_id")] = row
    for path in paths:
        name = os.path.basename(path)
        tags = [str(t) for t in (captions.get(name) or [])]
        view = next((t for t in tags if t.endswith("view")), "")
        framing = next((t for t in tags if t in (
            "full body shot", "upper body", "face close-up", "clothing close-up",
            "feet close-up", "accessory close-up", "pattern close-up")), "")
        rows[name] = {"rel_path": name, "view": view, "framing": framing, "tags": tags,
                      "source": "captions.txt" if tags else "unlabelled",
                      "marks": sorted(stored.get(name, {}).keys())}
    return rows


# ---------------------------------------------------------------------------
# 条件文本（供训练与推理使用；只复述已知内容，不新增未观察到的描述）
# ---------------------------------------------------------------------------
def _evidence_text(mark) -> str:
    """标志的结构化外观文本：优先用 VLM 结构化结果，其次原文，绝不臆造。

    结构化结果本来就按"颜色优先、几何其次"写成一句可直接当 caption 的英文
    （`description_en`），因此**不再重复拼接**结构字段：实测把
    outline / inner structure / colour relation / material / scale 全部串起来，
    单条能到 272 字符 / 60+ token，而推理提示词总共只有 77 token——
    结果是标志条件被切成"外袖"这种没有信息的碎片。
    `description_en` 已经包含同样的信息，重复拼接只是浪费预算。
    只有在没有 `description_en` 时才退回结构字段（那是唯一的依据）。
    """
    structured = str(mark.get("description_en") or "").strip()
    if structured:
        return structured
    parts = []
    for key, label in (("outline", "outline"), ("internal_topology", "inner structure"),
                       ("color_relation", "colour relation"), ("repeat_unit", "repeat unit"),
                       ("material", "material"), ("relative_scale", "scale")):
        value = str(mark.get(key) or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    if not parts:
        parts.append(str(mark.get("description") or "").strip())
    return ", ".join(p for p in parts if p)


def mark_condition(mark) -> str:
    """单个标志的训练/推理条件文本。

    有 `description_en` 时**只用它**：它是"颜色优先、几何其次"的一句英文，
    语义密度最高，而且不夹中文——中文短名在 CLIP 里几乎一词一 token
    （实测 21 个汉字 ≈ 21 token），把宝贵的预算吃掉一半却只提供标签信息。
    没有结构化结果时才退回中文原文（此时没有更好的依据，如实使用）。
    """
    structured = str(mark.get("description_en") or "").strip()
    if structured:
        return structured
    text = _evidence_text(mark)
    name = str(mark.get("name") or "").strip()
    if name and text and not text.lower().startswith(name.lower()):
        return f"{name}: {text}"
    return text or name


def mark_label(mark) -> str:
    """标志的简短标签（中文名或英文名）：用于日志、审计与非提示词场景。"""
    return str(mark.get("name_zh") or mark.get("name") or mark.get("mark_id") or "").strip()


_BRIEF_DROP = {"a", "an", "the", "with", "of", "on", "in", "and", "decorated"}


def mark_brief_en(mark, words: int = 6) -> str:
    """标志的**简短英文点名**，用于 77 token 预算紧张的整图提示词。

    为什么不直接用 `mark_condition`（完整描述）：完整描述是给**训练样本**用的，
    在整图提示词里它太长——实测一条就要 90+ 字符 / 约 22 token，
    而人物提示词扣掉身份（约 22 token）、源图媒介、构图、姿态、质量词之后
    只剩十几个 token。更糟的是：裁剪样本的提示词里写全、推理时只写一小截，
    **训练与推理的条件分布不一致**，模型学到的是"跟着长描述走"，
    推理时拿不到长描述反而更容易画错。

    取名规则（只用库里已有字段，不编造）：
      1. `name_en` 优先——它是结构化阶段产出的短名（如
         `white cherry blossom pattern`）。但 `name_en` 常常**不含颜色**
         （`cloud pattern` / `bow decoration`），而颜色正是用户最看重的区分点，
         因此从 `description_en` 里补上开头的颜色词。
      2. 没有 `name_en` 时，取 `description_en` 的第一个短名词短语。
    """
    name = str(mark.get("name_en") or "").strip()
    description = str(mark.get("description_en") or "").strip()
    colours = []
    for token in re.split(r"[^A-Za-z]+", description):
        lowered = token.lower()
        if lowered in _COLOR_TOKENS and lowered not in colours:
            colours.append(lowered)
        if colours:
            break
    if name:
        # `name_en` 里已经表达了颜色就不再重复。判断用**词形包含**而不是整词相等：
        # 实测 `golden cloud pattern` / `Golden Pattern` 这类名字里的
        # `golden` 已经含 `gold`，整词比较会判成"没有颜色"，
        # 于是拼出 `gold golden cloud pattern` 这种重复（实测）。
        lowered_name = name.lower()
        if colours and not any(token in lowered_name for token in _COLOR_TOKENS):
            return f"{colours[0]} {name}".strip()
        return name
    if description:
        kept = [t for t in description.split() if t.lower() not in _BRIEF_DROP]
        return " ".join(kept[:words]).strip(" ,;:")
    return mark_label(mark)


def observed_marks(folder) -> list:
    """本角色**已有图像证据**的标志（visible/partial），没有证据的不返回。

    用途：训练条件与推理条件都只能描述"确有像素依据"的标志。
    """
    library = load_library(folder)
    if not library:
        return []
    observed = {row.get("mark_id") for row in load_observations(folder)
                if row.get("visibility") in ("visible", "partial")}
    return [mark for mark in library.get("marks") or []
            if mark.get("mark_id") in observed]


def _locatable(marks):
    """只保留"有边界的实体标志"。

    外观属性（配色/发色/瞳色/发型剪裁）没有任何位置框，把它们写进**标志条件**
    会与身份条件重复（身份条件已经覆盖发色/瞳色/配色），既挤占 CLIP 预算，
    又让概念分支以为自己该去学"头发颜色"这种整图属性。它们仍然留在标志库里，
    逐图可见状态照常记录（规格 §4），只是不参与标志条件的拼接。
    """
    return [m for m in marks or []
            if str(m.get("entity_mode") or "locatable") == "locatable"]


def _join_within_budget(pieces, max_chars) -> str:
    """按顺序把标志条件拼到预算内；**装不下的单条跳过，而不是终止整个拼接**。

    实测踩过（这一条代价很大）：结构化之后单条标志的条件可以到 272 字符
    （"三圈两线结绳装饰: light purple three concentric circles ..."），
    而预算 `MARK_CONDITION_MAX_CHARS` 是 200。早期写法在第一条超预算时直接 `break`，
    于是**所有**标志条件都变成了空串——整图条件里没有标志、裁剪样本的条件里也没有，
    而日志看起来一切正常（没有报错、没有告警）。
    正确做法：跳过超预算的那一条，继续尝试后面的短条目。
    """
    clause = ""
    for piece in pieces:
        piece = str(piece or "").strip()
        if not piece:
            continue
        if len(piece) > max_chars:
            continue          # 单条就超预算：它自己进不去，但不影响别的条目
        candidate = f"{clause}, {piece}" if clause else piece
        if len(candidate) > max_chars:
            continue          # 剩余预算装不下：继续找更短的
        clause = candidate
    return clause


def condition_items(folder, only_observed=True, max_chars=None, brief=False) -> list:
    """标志条件，**一条一个元素**（按重要性排序）。

    与 `condition_clause` 只差返回形态。为什么要分开（实测踩过）：
    标志描述自身含逗号，调用方若先拿到整串再按逗号切预算，
    描述会被切成没有图案信息的碎片（`light purple three concentric circles`
    这类描述被切碎后完全不可用）。按条返回，调用方就能整条取舍。

    `brief=True` 时用 `mark_brief_en` 的短点名而不是完整描述：
    整图提示词的预算只够点名（详见 `mark_brief_en` 的说明），
    完整描述留给概念分支的训练样本。
    """
    library = load_library(folder)
    if not library:
        return []
    marks = _locatable(library.get("marks") or [])
    if only_observed:
        observed = {row.get("mark_id") for row in load_observations(folder)
                    if row.get("visibility") in ("visible", "partial")}
        if observed:
            marks = [m for m in marks if m.get("mark_id") in observed]
    rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(marks, key=lambda m: (rank.get(str(m.get("user_importance")), 1),
                                           str(m.get("name") or "")))
    pieces = []
    for mark in ordered:
        text = mark_brief_en(mark) if brief else mark_condition(mark)
        if not text:
            continue
        if brief and _has_cjk(text):
            # 短点名**只收英文**：底模对中文图案描述几乎没有响应，
            # 而中文在 CLIP 里约 1 token/字——实测 `腰前绳结也为三圈两线样式`
            # 一条就要 33 token，是英文名（5 token）的 6 倍以上，
            # 等于用一个"模型看不懂的长串"挤掉好几条真正有效的英文标志。
            # 这些标志仍留在标志库里、逐图可见状态照常记录（规格 §4），
            # 只是不进整图提示词的短点名。
            continue
        if max_chars is not None and len(text) > max_chars:
            continue
        # **专属触发词必须出现在条件里**（规格 §2 第 2 行）：
        # 概念权重是"专属 token -> 这条标志"训练出来的，
        # 条件里没有那个 token，权重就不会被唤起——
        # 这正是"物体表面只得到一层很淡渐变"的根因。
        token = mark_token(mark, config.get_trigger(folder))
        pieces.append(f"{token} {text}".strip() if token else text)
    return pieces


def condition_clause(folder, max_chars=None, only_observed=True) -> str:
    """标志条件：按重要性排序拼接，受 CLIP 预算约束（单条超预算则跳过该条）。

    结构化之后的描述可能很长（实测单条到 272 字符），因此这里**必须**容忍
    "装不下这一条"而不是整体放弃——早期在第一条超预算时直接终止，
    结果是所有标志条件都变成空串（详见 `_join_within_budget`）。
    """
    max_chars = config.MARK_CONDITION_MAX_CHARS if max_chars is None else max_chars
    library = load_library(folder)
    if not library:
        return ""
    marks = _locatable(library.get("marks") or [])
    if only_observed:
        observed = {row.get("mark_id") for row in load_observations(folder)
                    if row.get("visibility") in ("visible", "partial")}
        if observed:
            marks = [m for m in marks if m.get("mark_id") in observed]
    rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(marks, key=lambda m: (rank.get(str(m.get("user_importance")), 1),
                                           str(m.get("name") or "")))
    return _join_within_budget([mark_condition(mark) for mark in ordered], max_chars)


def _normalise_rel_path(rel_path) -> str:
    """把"逐图证据里的键"与"调用方给的路径"统一成同一种写法。

    为什么需要（实测）：逐图证据里存的键是**纯文件名**（`019.png`、`001.JPG`），
    而调用方按"数据集内的相对路径"给值（`images/019.png`）。
    两边写法不一致时查表永远落空——而落空的后果是**静默的**：
    可见性门控会以为"这张图完全没有证据"，于是退回角色级条件，
    把这张图上看不到的特征又写回训练条件里（正是门控要防的那件事）。
    """
    if not rel_path:
        return ""
    text = str(rel_path).replace("\\", "/").strip()
    marker = "/images/"
    if marker in text:
        text = text.split(marker, 1)[1]
    elif text.startswith("images/"):
        text = text[len("images/"):]
    return text.lstrip("./")


def by_image_condition(folder, rel_path, max_chars=None) -> str:
    """单张图的标志条件：**只写该图可见证据支持的标志**（规格 §2 第 4 行要求）。

    背面图不会写进"胸前的徽记"这类不可见内容——看不到就不写，而不是照抄角色级短语。
    """
    max_chars = config.MARK_CONDITION_MAX_CHARS if max_chars is None else max_chars
    library = load_library(folder)
    if not library:
        return ""
    table = evidence_by_image(load_observations(folder)).get(
        _normalise_rel_path(rel_path)) or {}
    rank = {"high": 0, "medium": 1, "low": 2}
    pieces = []
    for mark in _locatable(library.get("marks") or []):
        row = table.get(mark.get("mark_id"))
        if not row or row.get("visibility") not in ("visible", "partial"):
            continue
        pieces.append((rank.get(str(mark.get("user_importance")), 1), mark_condition(mark)))
    return _join_within_budget([piece for _rank, piece in sorted(pieces, key=lambda i: i[0])],
                               max_chars)


def has_image_evidence(folder, rel_path) -> bool:
    """这张图**有没有逐图证据行**（不论可见与否）。

    用途（实测缺陷）：`dataset.mark_condition` 原先写 `自己的条件 or 角色级条件`，
    于是"这张图有证据、但一条都不可见"（例如只拍正脸的正面特写）也会退回角色级条件，
    把"胸口的图案、颈部的挂饰"写进一张根本看不到它们的图的训练条件里。

    区分两种"空"是关键：
      * **有证据行、但都不可见** → 这张图确实看不到标志，条件应当留空；
      * **完全没有证据行**（VLM 未启用、或整批都是 unknown 未落盘）→ 退回角色级条件
        是当时唯一可用的信息。
    """
    if not rel_path:
        return False
    try:
        rows = evidence_by_image(load_observations(folder))
    except Exception:                     # noqa: BLE001
        return False
    return bool(rows.get(_normalise_rel_path(rel_path)))


# ---------------------------------------------------------------------------
# 非人物承载计划（规格 §6：标志如何落到目标主体）
# ---------------------------------------------------------------------------
# 同一个徽记可以出现在人物胸前、猫的项圈饰片或杯子的可见表面；它仍是同一个标志，
# 承载关系不同。**不能**把原角色的"胸前坐标"复制到杯子的画面中心。
# 这里按"目标主体能提供什么承载面"来分配，而不是按标志在原角色身上的位置照搬。
_CARRY_HOST = {
    # 目标主体 -> 可用的承载方式（按优先级）
    "animal": ("wearable", "trim", "emblem", "palette"),
    "object": ("pattern", "emblem", "palette", "trim"),
    "human": ("palette", "emblem", "pattern", "trim", "wearable"),
}

# 承载方式 -> 目标区域（附着/表面支路用的区域名）
_CARRY_REGION = {
    "animal": {"wearable": "collar", "trim": "collar", "emblem": "chest",
               "palette": "surface", "pattern": "surface"},
    "object": {"pattern": "surface", "emblem": "surface", "palette": "surface",
               "trim": "surface"},
    "human": {},
}


def carry_plan(folder, subject_type, limit=None, observations=None, library=None) -> dict:
    """把**有真实证据**的标志分配到目标主体的承载方式上（规格 §6 的表格）。

    返回 {"subject_type", "assignments": [{mark_id, name, mark_class, carry,
    region, reason, condition}], "unassigned": [...], "notes": [...]}。

    分配原则（每条都有理由）：
      * 只分配有可见证据的标志——没有像素依据的标志不参与生成条件；
      * 承载方式由**目标主体能提供什么**决定（动物：可佩戴的颈圈/胸饰；物体：可见表面），
        而不是照搬标志在原角色身上的部位（"胸前坐标"不该复制到杯子中心）；
      * 配色类标志先走配色承载（它是整体色彩构成，不是一个可放置的物件）；
      * 目标主体给不出的承载方式如实记入 unassigned，不硬塞。
    """
    library = library or load_library(folder)
    if not library:
        return {"subject_type": subject_type, "assignments": [], "unassigned": [],
                "notes": ["没有统一标志库"]}
    observations = observations if observations is not None else load_observations(folder)
    observed = {row.get("mark_id") for row in observations
                if row.get("visibility") in ("visible", "partial")}
    kinds = _CARRY_HOST.get(str(subject_type or "unknown"), _CARRY_HOST["human"])
    regions = _CARRY_REGION.get(str(subject_type or "unknown"), {})
    # 用户用 `mark_refs.txt` **单独点名**的标志（有 links 指向参考图）：
    # 这些是用户自己认为"一眼可辨"的标志，且有干净的参考素材。
    # 实测踩过：只按标志类别排序时，`pattern` 类（衣服上的纹样裁剪）会排在
    # `emblem` 类之前并把它们挤出 limit，于是杯子上贴的全是"角色裁片照片"，
    # 而用户专门提供的黑樱花/白樱花/扇子/金色图案**一个都没用上**。
    designated = {link.get("mark_id") for link in (library.get("links") or [])
                  if link.get("mark_id")}
    assignments, unassigned, notes = [], [], []
    for mark in library.get("marks") or []:
        if str(mark.get("entity_mode") or "locatable") != "locatable":
            continue
        if mark.get("mark_id") not in observed:
            unassigned.append({"mark_id": mark.get("mark_id"), "name": mark.get("name"),
                               "reason": "没有可见证据，不能进入生成条件"})
            continue
        mark_class = str(mark.get("mark_class") or "detail")
        carry = mark_class if mark_class in kinds else None
        if carry is None:
            carry = "pattern" if "pattern" in kinds else kinds[0]
            notes.append(f"标志「{mark.get('name')}」的类型 {mark_class} 不在 "
                         f"{subject_type} 的承载方式里，退回 {carry}")
        assignments.append({
            "mark_id": mark.get("mark_id"), "name": mark.get("name"),
            "mark_class": mark_class, "carry": carry,
            "region": regions.get(carry, "surface"),
            "reason": config.MARK_CARRY_PLAN.get(carry, ""),
            "condition": mark_condition(mark),
            "origin_region": mark.get("region"),
            "user_designated": mark.get("mark_id") in designated,
            # 带上标志本身：`carry_items` 需要它的**专属触发词**才能把条件
            # 变成"被训练过的那个条件"（否则概念权重不会被唤起）
            "mark": mark,
        })
    # 承载顺序（三级）：
    #   1. **用户单独点名的标志优先**——用户提供 `mark_refs.txt` 就是指定
    #      "这一张图就是我要的那个标志"，而且实测这些参考图是干净的图案
    #      （白底 + 黑樱花 + 金花心），远比衣服裁片适合"一眼可辨"；
    #   2. 再按承载方式排序（可佩戴件与徽记优先成形，花纹做底）；
    #   3. 同类里按名称稳定排序，保证同样输入得到同样计划。
    order = {kind: index for index, kind in enumerate(kinds)}
    assignments.sort(key=lambda item: (0 if item.get("user_designated") else 1,
                                       order.get(item["carry"], 99),
                                       str(item["name"] or "")))
    if limit:
        assignments = assignments[:limit]
    if not assignments:
        notes.append(f"{subject_type} 没有任何有证据的标志可供承载")
    return {"subject_type": subject_type, "assignments": assignments,
            "unassigned": unassigned, "notes": notes}


def carry_items(folder, subject_type, limit=3, observations=None, library=None) -> list:
    """承载计划的提示词片段，**一条一个元素**（调用方按条裁预算）。

    与 `carry_clause` 的区别只在返回形态：这里返回列表而不是用逗号连成整串。
    为什么必须分开（实测踩过）：标志描述自身含逗号
    （`light purple three concentric circles with two intersecting lines on a
    white background`），一旦先连成整串再按逗号切分，描述就被切成无意义的碎片，
    而且第一条碎片就吃满预算、后面的标志一条都进不去——
    表现是"有承载计划、提示词里却没有任何标志"（非人物加法完全看不到的根因之一）。
    """
    plan = carry_plan(folder, subject_type, limit=limit,
                      observations=observations, library=library or load_library(folder))
    trigger = config.get_trigger(folder)
    pieces = []
    for item in plan["assignments"]:
        condition = str(item.get("condition") or "").strip()
        if not condition:
            continue
        # 每条都必须带**该标志的专属触发词**：概念权重是"专属 token -> 这条标志"
        # 训练出来的，条件里没有那个 token 就不会被唤起
        # （实测根因：物体表面因此只得到一层很淡的配色渐变）。
        mark = item.get("mark") or {}
        token = mark_token(mark, trigger) if mark else ""
        # **有条件地省略承载体前缀**：训练语料是 `token, <描述>`，
        # 而在 token 与描述之间插进 `this real repeating pattern:` 之类的前缀，
        # 会让条件偏离训练分布（实测：杯子图案只呈现一层很淡的渐变），
        # 并且每条前缀约 7 token，两条就把 `no human face` 这类必需约束挤出预算。
        # 需要更"像在描述承载方式"时用 `config.MARK_CARRY_EXPLICIT_PREFIX` 打开。
        explicit = bool(getattr(config, "MARK_CARRY_EXPLICIT_PREFIX", False))
        if explicit:
            if item["carry"] == "wearable":
                body = f"a wearable made from this real mark: {condition}"
            elif item["carry"] == "emblem":
                body = f"this real emblem adapted to the host surface: {condition}"
            elif item["carry"] == "pattern":
                body = f"this real repeating pattern: {condition}"
            elif item["carry"] == "trim":
                body = f"this real trim laid along the host edge: {condition}"
            else:
                body = f"character colour scheme: {condition}"
        else:
            body = condition
        pieces.append(f"{token}, {body}" if token else body)
    return pieces


def carry_clause(folder, subject_type, limit=3, observations=None, library=None) -> str:
    """承载计划的提示词片段：把标志的**真实描述**与承载方式一起写出来。

    实测教训（config.py 里记录的附着区域四轮实测）：概念权重学的是"角色裁片长什么样"，
    没有"这是一条戴在脖子上的项圈"这种位置/语义概念，单靠"加个配饰"很难成形。
    因此这里显式写出"把这个真实的标志做成颈圈饰片"，而不是泛泛地说"角色图案"。

    注意：直接用本函数做**预算裁剪**时不要按逗号切分（描述自身含逗号），
    应改用 `carry_items` 拿到分条列表。本函数保留给"整串直接用、不裁"的场合。
    """
    return ", ".join(carry_items(folder, subject_type, limit=limit,
                                 observations=observations, library=library))


def describe(library) -> str:
    """人类可读摘要（日志与验收记录用，不是提示词）。"""
    lines = [f"统一标志库: {library.get('character')} "
             f"标志 {len(library.get('marks') or [])} 个 "
             f"（marks.txt={library.get('files', {}).get('marks_txt')}，"
             f"mark_refs.txt={library.get('files', {}).get('mark_refs_txt')}）"]
    for mark in library.get("marks") or []:
        lines.append(f"  [{mark['mark_id']}] {mark['name']} "
                     f"区域={mark['region']} 类={mark['mark_class']} "
                     f"重要性={mark['user_importance']} 证据={mark['evidence_confidence']}"
                     f"{'（强调×%d）' % mark['emphasis_sources'] if mark.get('emphasis_sources', 1) > 1 else ''}")
    for gap in library.get("unresolved_gaps") or []:
        lines.append(f"  ! 缺口: {gap}")
    for warning in library.get("warnings") or []:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)
