# -*- coding: utf-8 -*-
"""character/mark_layout.py —— 从**角色自己的图像证据**里量出"标志在空间上怎么用"。

## 为什么要有这个模块（用户指出的问题）

用户（2026-10-06）：

> "樱花样式在角色服装上是以零星覆盖在裙摆上以飘落的形式存在，
>  而金色图案在角色服饰中虽然较大但不会说占较大视觉形成视觉主体
>  并且通常可能和金色祥云纹样式的线条组合使用……
>  需分清纹样图案这种花纹式的和配饰装饰这种实质性的"

也就是说，光知道"标志长什么样"不够，还要知道它在角色身上**怎么分布**：

* 樱花是**零星散布**的（不是单个居中大图）；
* 金色图案虽大但**不占据视觉主体**，且常与线条纹样**组合**；
* **花纹式**（可印在表面）与**实质性配饰**（实物）是两类，承载方式不同。

这三条都能从证据里**量出来**，不需要写死任何角色：

1. **覆盖率**：标志框面积 / 角色画面面积 —— 判断它是"视觉主体"还是"点缀"；
2. **散布度**：同一标志在多少张图上、出现位置有多分散 —— 判断是"零星散布"还是"集中一处"；
3. **组合性**：两个标志的框是否经常重叠/相邻 —— 判断是否"组合使用"。

把这三条量化后，承载与提示词就能按**该角色自己的用法**去摆放，
而不是把所有标志都当成"居中贴一张"。
"""
import numpy as np

import config

# 覆盖率达到这个比例即视为"视觉主体"（面积占比，非边长）
FOCAL_COVERAGE = 0.030
# 散布度（出现位置的归一化标准差）超过这个值视为"零星散布"
SCATTERED_SPREAD = 0.16


def mark_geometry(observations, mark_id):
    """收集某个标志在所有图上的框（**归一化坐标**）。

    实测：逐图证据里的 `box` 已经是 0~1 的归一化坐标（另有 `pixels` 是像素坐标），
    因此不需要再去取画面尺寸；早期版本要求 `frame`/`image_size` 字段，
    而证据里没有这两个字段，导致**全部判为 unknown**（21/21 条都没有几何量）。

    只有**确实定位到**（有真实框）的才计入——
    `unknown`/`absent` 那些没有框，不能用来推断空间用法（不编造）。
    """
    rows = []
    for row in observations or []:
        if row.get("mark_id") != mark_id:
            continue
        box = row.get("box")
        if not box or len(box) < 4:
            continue
        try:
            values = [float(v) for v in box[:4]]
        except (TypeError, ValueError):
            continue
        rows.append({"rel_path": row.get("rel_path"), "box": values})
    return rows


def spatial_behaviour(observations, mark_id):
    """量出一个标志的**空间用法**，返回审计字典。

    所有量都在**归一化坐标**上算，因此与图片尺寸无关：

      * `images`     : 证据图数量（有真实框的）
      * `coverage`   : 框面积的中位数（归一化面积，判断视觉权重）
      * `spread`     : 框中心的归一化标准差（判断散布程度）
      * `max_side`   : 框最长边的中位数（判断"大件"还是"小件"）
      * `role`       : `focal`（视觉主体）/ `scattered`（零星散布）/ `accent`（局部点缀）

    判据顺序（先看散布，再看大小）：
      散布度高 -> `scattered`（例如"零星覆盖在裙摆上"）
      否则面积大 -> `focal`（占据视觉主体）
      否则 -> `accent`（局部点缀）
    """
    rows = mark_geometry(observations, mark_id)
    if not rows:
        return {"images": 0, "role": "unknown", "reason": "没有可用的真实框"}
    coverages, sides, centres = [], [], []
    for row in rows:
        left, top, right, bottom = row["box"]
        area = max(0.0, (right - left)) * max(0.0, (bottom - top))
        coverages.append(float(area))
        sides.append(max(right - left, bottom - top))
        centres.append(((left + right) / 2.0, (top + bottom) / 2.0))
    centres = np.array(centres, dtype=np.float32)
    spread = float(np.mean(centres.std(axis=0))) if len(centres) > 1 else 0.0
    coverage = float(np.median(coverages))
    max_side = float(np.median(sides))
    if spread >= float(getattr(config, "MARK_LAYOUT_SCATTER_SPREAD", SCATTERED_SPREAD)):
        role = "scattered"
    elif coverage >= float(getattr(config, "MARK_LAYOUT_FOCAL_COVERAGE", FOCAL_COVERAGE)):
        role = "focal"
    else:
        role = "accent"
    return {"images": len(rows), "coverage": round(coverage, 4),
            "spread": round(spread, 4), "max_side": round(max_side, 4),
            "role": role,
            "note": {"focal": "在角色上占据视觉主体 -> 承载时可以作为主件",
                     "scattered": "在角色上是零星散布 -> 承载时应重复铺开而不是单张大图",
                     "accent": "在角色上是局部点缀 -> 承载时应小而少，不抢主体"}[role]}


def nature_of(mark):
    """区分**花纹式**（可印在表面）与**实质性**（物理实物）。

    依据 `mark_class`（由 `character.marks.classify_mark` 依描述词判定）+ 名称里的
    实物词。实测当前库里被误判的典型：
      * `扇子` -> `pattern`：扇子是**实物**（角色手里拿的），
        但分类规则看到"扇"就当花纹了，于是被当成 surface pattern 平铺到杯子上；
      * `淡蓝色上衣` / `外袖短至手肘`（衣服/部位描述）被归到 `wearable`；
      * `脖子上的金色挂饰` -> `emblem`，其实它是**实物挂饰**。

    返回 `"pattern"`（花纹，可印）或 `"accessory"`（实物，只能以实物形态出现）。
    """
    name = str(mark.get("name") or mark.get("canonical") or "")
    mark_class = str(mark.get("mark_class") or mark.get("kind") or "")
    # 注意：**不要用 `entity_mode` 来区分"花纹 / 配饰"**。
    # `entity_mode` 回答的是另一个问题——"这个标志能不能定位出框"
    # （`locatable` 有框、`describable` 只能描述），实测库里 21 条的 region 都是
    # `whole`，于是 `entity_mode` 一律返回 `describable`，
    # 用它过滤会把**全部**标志滤掉（承载计划变空）。真正的区分靠名称与类别。
    #
    # 部位/服装/颜色描述词：`淡蓝色发色`、`灰白色瞳色`、`单马尾`、`刘海裁齐`、
    # `颜色由淡蓝和深蓝组成` 这些是**角色自身的属性描述**，
    # 不是可以"加到器物上"的东西（实测库里 21 条有 6 条属于此类）。
    body_words = ("上衣", "外袖", "内袖", "裙", "裤", "刘海", "发色", "发髻", "发型",
                  "瞳", "皮肤", "马尾", "颜色", "色调")
    # 实物词表：出现即判为实质性配饰（不管分类规则怎么判）。
    # 实测踩到的典型：`扇子` 被 `_CLASS_RULES` 判成 `wearable`，
    # 而承载时按 `pattern` 平铺到杯子上——扇子是**实物**，不能当表面花纹刷。
    object_words = ("扇", "项圈", "挂饰", "坠", "盔", "刀", "剑", "弓",
                    "腰带", "绳", "蝴蝶结", "簪", "珠", "结")
    if any(word in name for word in object_words):
        return "accessory"
    if any(word in name for word in body_words):
        return "descriptive"
    if mark_class in {"pattern", "trim", "material", "motif", "print", "palette"}:
        return "pattern"
    if mark_class in {"wearable", "emblem", "accessory"}:
        # `emblem` 是语义徽记：它本身可以是印上去的纹章，因此按"花纹式"处理；
        # 但已经被实物词命中的在上面就返回 accessory 了。
        return "pattern" if mark_class == "emblem" else "accessory"
    return "pattern"


def layout_plan(library, observations, limit=None):
    """给出"这批标志该怎么被使用"的计划（每条带 nature 与 role）。

    只做**判定与说明**，不改动承载几何；承载方（`vision/carry_compose.py`）
    依此决定：花纹式 -> 可平铺/印；实质性 -> 只允许以实物形态出现（不做表面平铺）。
    """
    items = []
    for mark in (library or {}).get("marks") or []:
        mark_id = mark.get("mark_id")
        behaviour = spatial_behaviour(observations, mark_id)
        nature = nature_of(mark)
        if nature == "descriptive":
            # 角色自身的属性描述（发色/瞳色/发型/配色）不参与承载：
            # 它们不是"能加到器物上的东西"。
            continue
        items.append({"mark_id": mark_id, "name": mark.get("name"),
                      "nature": nature,
                      "mark_class": mark.get("mark_class"),
                      "role": behaviour.get("role"),
                      "coverage": behaviour.get("coverage"),
                      "spread": behaviour.get("spread"),
                      "images": behaviour.get("images"),
                      "note": behaviour.get("note") or behaviour.get("reason")})
    items.sort(key=lambda item: (-(item.get("coverage") or 0.0), str(item.get("name"))))
    if limit:
        items = items[:int(limit)]
    return items


def describe(items) -> str:
    lines = ["[layout] 标志的空间用法判定（来自角色自己的图像证据）:"]
    for item in items or []:
        lines.append(f"  - {str(item.get('name'))[:18]:<20} "
                     f"性质={item.get('nature'):<10} 角色={str(item.get('role')):<10} "
                     f"覆盖={item.get('coverage')} 散布={item.get('spread')} "
                     f"证据={item.get('images')} 张")
    return "\n".join(lines)
