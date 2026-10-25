# -*- coding: utf-8 -*-
"""test.py —— 推理主入口（本项目唯一正式推理入口）

    python test.py
    python test.py --character ayaka --input input/human2.png

流程：
    输入解析 -> 角色选择与角色包校验 -> VLM 主体理解（用完卸载）
    -> 加载角色权重 + 结构条件 -> 逐图受控迁移 -> 保存结果与权重审计

要点（对应 docs/new.txt 与方案 §7）：
- 必须加载所选角色的训练权重；缺失或不兼容时明确报错，不静默改用纯底模。
- 保留源图的形态、比例、动作、朝向、构图、穿戴关系与表现媒介；
  角色身份（配色、发色、服装结构、面部、标志性配饰与图案）来自权重与角色条件。
- 非人物保留形态与功能部位，只做表面与附件层面的角色化，不拟人化。
- 每张图的生成记录里写明实际加载的权重、强度与条件，运行完成 ≠ 视觉验收通过。
"""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

from PIL import Image  # 顶层导入：合成与掩版处理都要用，先前只在个别函数内部 import，
                       # 一旦新增调用点就会 NameError（实测踩过）。
import cv2  # 配色锚定的掩版同质性判据要用（LAB 离散度）；顶层导入避免同一坑
import numpy as np  # 同上

import config
from vision import carry_compose  # noqa: E402

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)
if config.HF_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

import image_utils  # noqa: E402
import model_utils  # noqa: E402
import subject_transfer  # noqa: E402
from character import cards as cards_module  # noqa: E402
from character import semantics  # noqa: E402
from character.runtime import select_cli_character  # noqa: E402
from vision.segmentation import (load_sam2, plausible_subject_mask, segment_object_surface,
                                 segment_subject, segment_subject_edges,
                                 subject_mask_from_regions, unload_sam)  # noqa: E402
from vision.color_hint import color_hint  # noqa: E402
from vision.subject_analyzer import (analyze_subject, attachment_masks, locate_regions,
                                     region_masks, subject_silhouette)  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def result_image_dir(folder):
    """角色成品图唯一目录，网页与 CLI 共用。"""
    return Path(config.RESULT_DIR) / f"{folder}_result"


def _print_palette(folder):
    """取该角色实测调色板（供曲面承印着色用）；取不到时返回空列表。"""
    try:
        palette, _note = carry_compose.character_palette(folder)
        return palette or []
    except Exception:      # noqa: BLE001
        return []


def diagnostic_dir(folder):
    """诊断图与中间图不混入成品目录。"""
    return Path(config.OUTPUT_DIR) / "diagnostics" / folder


def resolve_inputs() -> list:
    """解析输入：文件、目录（批量）或列表；同名文件会导致结果互相覆盖，因此拒绝。

    列表形态（`--input a.png b.png` 或 `--input <目录>`）里的每一项**都可能是目录**，
    必须逐项展开：早期实现只把顶层当目录处理，列表里的目录会被当成图片文件校验，
    于是 `--input <目录>` 直接报"不是存在的图片文件"（实测踩过）。
    """
    value = config.INPUT_IMAGE
    entries = list(value) if isinstance(value, (list, tuple)) else [value]
    paths = []
    for entry in entries:
        if os.path.isfile(entry):
            paths.append(os.path.abspath(entry))
        elif os.path.isdir(entry):
            paths.extend(sorted((os.path.join(entry, name) for name in os.listdir(entry)
                                 if name.lower().endswith(IMAGE_EXTS)),
                                key=lambda p: os.path.basename(p).lower()))
        else:
            raise FileNotFoundError(f"输入既不是文件也不是目录: {entry}")
    for path in paths:
        if not os.path.isfile(path) or not path.lower().endswith(IMAGE_EXTS):
            raise ValueError(f"--input 必须是存在的图片文件: {path}")
    stems = {}
    for path in paths:
        stem = Path(path).stem.casefold()
        if stem in stems:
            raise ValueError(f"输入图片重名，会产生互相覆盖的结果: "
                             f"{stems[stem]} 与 {path}")
        stems[stem] = path
    return paths


def resolve_weight(folder) -> str:
    """确定并校验本次要加载的角色权重；缺失即报错（不静默用纯底模）。"""
    audit = semantics.audit_switch(folder)
    if not audit["ready"]:
        raise FileNotFoundError(
            f"角色 {folder} 的角色包不可用：{audit.get('reason')}"
            f"{audit.get('missing') or ''}\n请先运行: python train.py --character {folder}")
    identity = next((w for w in audit["weights"] if w["role"] == "identity"),
                    audit["weights"][0] if audit["weights"] else None)
    if identity is None:
        raise FileNotFoundError(f"角色 {folder} 的角色包内没有权重条目")
    path = semantics.weight_path(folder, identity["role"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"角色权重文件不存在: {path}")
    print(f"[test] 角色包: {semantics.character_dir(folder)}")
    print(f"[test] 权重: {os.path.basename(path)}（role={identity['role']}）")
    return path


def resolve_concept_weight(folder, subject_types):
    """概念权重（配饰/图案/花纹）：只在非人物输入上启用。

    分工原则（用户重申）：主权重管"角色本身"，概念权重管"把标志加到主体表面"。
    人物身份已由主权重承担，概念权重在人物上默认不启用（scale=0 等于不加载）。
    若场次里既有非人物又没有概念权重，明确提示但不阻断——非人物会退化为"只有配色与服装语义"。
    """
    audit = semantics.audit_switch(folder)
    entry = next((w for w in audit.get("weights") or [] if w["role"] == "concept"), None)
    if entry is None:
        if any(kind in {"animal", "object"} for kind in subject_types):
            print("[test] 提示: 该角色还没有概念权重，非人物只能用主权重做表面角色化；"
                  "重新训练一次即可自动产出概念分支")
        return None, {}
    path = semantics.weight_path(folder, "concept")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"角色包声明了概念权重但文件不存在: {path}")
    return path, entry


def build_pipeline(weight_path, device, dtype, concept_path=None):
    components = model_utils.load_components(device=device, dtype=dtype)
    pipe, use_controlnet = model_utils.build_img2img(components, with_controlnet=config.USE_CONTROLNET)
    if not use_controlnet:
        raise RuntimeError(
            "结构条件不可用（ControlNet 缺失或加载失败）。本项目要求保留源图形态与构图，"
            "拒绝在没有结构条件的情况下继续；请修复 models/controlnet-canny-sdxl-1.0。")
    audit = model_utils.load_adapter(pipe, weight_path, scale=config.LORA_SCALE)
    if concept_path:
        # 概念权重以第二个 adapter 挂上，人物场景强度 0（等于不启用），非人物场景由调用方调高
        model_utils.load_adapter(pipe, concept_path, scale=config.CONCEPT_HUMAN_SCALE,
                                 adapter_name="concept")
        audit = dict(audit, concept={"path": concept_path,
                                     "scale_human": config.CONCEPT_HUMAN_SCALE,
                                     "scale_non_human": config.CONCEPT_NON_HUMAN_SCALE})
    return pipe, audit


def reference_images(folder, profile, card) -> list:
    """为受控参考增强准备参考图（选择依据来自该角色自己的自动标注与标志证据）。

    `semantics.select_references` 会给出两种槽位：
      * face / costume：该角色自己的整图（按景别挑）；
      * mark：用户提供的**单独标志参考图裁剪**（规格 §3：保留真实参考图在推理时使用）。
    两种都按参考槽的固定顺序送入，槽位顺序必须与 `set_ip_adapter_scale` 的强度顺序一致。
    """
    entries = semantics.select_references(folder)
    prepared = []
    for entry in entries:
        path = entry.get("path") or os.path.join(
            config.DATASET_DIR, folder, "images", entry["rel_path"])
        if not os.path.isfile(path):
            print(f"[test] 参考图缺失，跳过: {path}")
            continue
        image = image_utils.load_image(path)
        image.thumbnail((config.REFERENCE_RENDER, config.REFERENCE_RENDER))
        entry = dict(entry, path=path, image=image)
        prepared.append(entry)
        print(f"[test] 参考图[{entry['slot']}]: {entry['rel_path']} "
              f"标注={entry['tags']}")
    if not prepared:
        print("[test] 没有可用于参考增强的图片（缺自动标注），该增强将跳过")
    return prepared


def generate_one(pipe, package, source_image, device, seed, index, references=None):
    settings = package["settings"]
    structure = package["structure"]
    if structure.size != source_image.size:
        structure = structure.resize(source_image.size)
    return _render(pipe, package["positive"], package["negative"], source_image,
                   structure, settings, device, seed, references)


def _render(pipe, positive, negative, source, structure, settings, device, seed,
            references=None):
    apply_subject_scales(pipe, settings)
    generator = torch.Generator("cuda" if device == "cuda" else "cpu").manual_seed(seed)
    kwargs = dict(
        prompt=positive, negative_prompt=negative,
        image=source, control_image=structure,
        strength=settings["strength"],
        controlnet_conditioning_scale=settings["control_scale"],
        control_guidance_end=settings.get("control_end", config.CONTROL_GUIDANCE_END),
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        guidance_scale=config.GUIDANCE_SCALE,
        generator=generator,
    )
    if references:
        # IP-Adapter 的参考图槽位顺序必须与 set_ip_adapter_scale 的强度顺序一致
        kwargs["ip_adapter_image"] = [r["image"] for r in references]
    started = time.time()
    result = pipe(**kwargs).images[0]
    # 把"这一遍真正用到的条件"记录下来：同一个种子下结果不一致时，
    # 只有把这些值打印出来才能判断差异来自参数还是别的环节（实测踩过这个坑）。
    print(f"[test]   渲染: strength={settings['strength']} "
          f"目标适配器={settings.get('scale_target', 'default')} "
          f"身份={settings.get('identity_scale', settings['lora_scale'])} "
          f"概念={settings.get('concept_scale', 0.0)} "
          f"control={settings['control_scale']} "
          f"end={settings.get('control_end', config.CONTROL_GUIDANCE_END)} "
          f"steps={config.NUM_INFERENCE_STEPS} cfg={config.GUIDANCE_SCALE} "
          f"src={source.size} seed={seed}", flush=True)
    print(f"[test]   渲染耗时 {time.time() - started:.1f}s", flush=True)
    return result


def apply_subject_scales(pipe, settings):
    """按 `settings` 提交**两个**适配器的强度，返回审计 dict。

    为什么必须一次提交两个、而且必须写明适配器名（实测踩过的静默 bug）：
    `_render` 原先写的是 `set_adapter_scale(pipe, settings["lora_scale"])`，
    缺省适配器名是身份那一支，于是 config 里注释为"概念权重强度"的
    `OBJECT_SURFACE_LORA_SCALE=1.45` / `ATTACH_LORA_SCALE=1.60` 实际调的是**身份**权重；
    更糟的是非人物分支先设好的 `NON_HUMAN_IDENTITY_SCALE=0.0` 会在下一行被
    覆盖成 `OBJECT_LORA_SCALE=0.85`——"非人物只做加法、不启用身份权重"
    这条设计从来没有真正执行过，而日志看起来一切正常。
    """
    target = str(settings.get("scale_target") or "default")
    if target == "concept":
        # 非人物：这一遍由**概念**权重承担，身份权重按配置（默认 0，只做加法）
        identity = float(settings.get("identity_scale",
                                      getattr(config, "NON_HUMAN_IDENTITY_SCALE", 0.0)))
    else:
        # 人物：这一遍由**身份**权重承担，概念权重按配置（默认 0）
        identity = float(settings["lora_scale"])
    concept = settings.get("concept_scale")
    applied = {}
    # 两处都写进同一张登记表，因此提交顺序不再影响最终值
    model_utils.set_adapter_scale(pipe, identity, adapter_name="default")
    applied["default"] = identity
    if concept is not None:
        model_utils.set_adapter_scale(pipe, float(concept), adapter_name="concept")
        applied["concept"] = float(concept)
    return applied


def character_color_reference(pipe, card, device, seed=None):
    """让**模型自己**把角色画一遍，作为该角色的颜色参照（返回 PIL 图 或 None）。

    为什么需要它：角色档案里存的是颜色**名字**（"light blue" → RGB），而用户要的是
    "和最还原角色的形象一致"。名字到具体色值之间有偏差，且不同角色的命名习惯不同。
    最短路径是让模型在**纯文生图**条件下把角色画出来——那是它对"这个角色长什么样"的
    最佳表达——再从这张图里取颜色。

    用法上它是**自适应**的：文生图拿到的肤色、发色、服装色都直接来自模型对触发词的
    理解，不依赖任何角色名或人工调色，换角色同样成立。
    失败一律返回 None，由调用方退回"按档案颜色名"的老路径（不静默失败）。
    """
    if not getattr(config, "COLOR_REFERENCE_FROM_T2I", False):
        return None
    try:
        # 必须用**文生图**管线：图生图管线缺 image 参数会直接报
        # "Input is in incorrect format"（实测）。因此这里单独装配一次文生图管线，
        # 两者共享同一份 UNet 与文本编码器，权重无需重新加载。
        components = {"pipe": pipe, "device": device, "dtype": next(pipe.unet.parameters()).dtype}
        text_pipe, _ = model_utils.build_text2img(components, with_controlnet=False)
        prompt = subject_transfer.character_reference_prompt(card)
        generator = torch.Generator("cuda" if device == "cuda" else "cpu").manual_seed(
            config.INFERENCE_SEED if seed is None else seed)
        image = text_pipe(prompt=prompt, negative_prompt=config.NEGATIVE_PROMPT,
                          num_inference_steps=config.COLOR_REFERENCE_STEPS,
                          guidance_scale=config.GUIDANCE_SCALE,
                          height=config.COLOR_REFERENCE_SIZE,
                          width=config.COLOR_REFERENCE_SIZE,
                          generator=generator).images[0]
    except Exception as exc:
        print(f"[test] 角色颜色参照生成失败（退回档案颜色名）: {type(exc).__name__}: {exc}")
        return None
    path = diagnostic_dir(config.CHARACTER_ID) / (
        f"{card.get('trigger') or 'character'}_color_reference.png")
    image_utils.save_image(image, path)
    print(f"[test] 角色颜色参照已生成: {path}（提示词: {prompt}）")
    return image


def character_color_lookup(reference, profile):
    """从颜色参照图里取出"发色 / 整体配色"的实测色值，作为配色锚定的目标色。

    取法（只用像素统计）：图像中心偏上区域里**最亮的那一部分**像素的色度中位数，
    近似代表发色；整幅较亮区域的色度中位数代表整体配色。取不到时返回空 dict，
    调用方退回档案颜色名。
    """
    import numpy as np
    import cv2
    if reference is None:
        return {}
    array = np.asarray(reference.convert("RGB"))
    height, width = array.shape[:2]
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB).astype(np.float32)
    head = lab[int(height * 0.08):int(height * 0.45), int(width * 0.25):int(width * 0.75)]
    if head.size < 64:
        return {}
    values = head.reshape(-1, 3)
    bright = values[values[:, 0] >= np.percentile(values[:, 0], 65)]
    hair_lab = np.median(bright, axis=0)
    hair_rgb = cv2.cvtColor(np.array([[hair_lab]], dtype=np.float32), cv2.COLOR_LAB2RGB)[0, 0]
    print(f"[test] 参照色: 发色 RGB={[int(v) for v in hair_rgb]}（LAB {hair_lab.round(1)}）")
    return {"hair_rgb": [int(round(float(v))) for v in hair_rgb]}


def refine_regions(pipe, folder, subject, base_result, regions, profile, card, device, seed,
                   source_image=None):
    """局部强化重绘：对头发/服装区域单独以**更高强度**重绘，再合成回去。

    源图的深色头发与深色衣着会在整图 img2img 中锁住初值，整图强度又不能再提高
    （会破坏姿态与构图）。因此这里对点名区域单独重绘：
    区域外逐像素等于当前结果，背景/姿态/构图不受影响。
    regions 的每一项可以是 SAM 细化后的像素掩版（首选）或视觉模型给的矩形框（退化）。

    ## `source_image`：每一块都从**源图**起画，而不是在上一次的结果上再画

    实测问题（贯穿多轮的"脸糊、画面发灰"）：原实现把 `current`（累积结果）
    既当"重绘的初值"又当"合成的底"，于是每个区域都是**在已经生成过的图上再生成一次**。
    同一个头部会被重绘两次（整图一遍 + 头部一遍），噪声累积——
    出图上人脸只剩一团色块、头发成片状，而每一步单独看都"符合规格"。

    现在把所有区域的重绘**初值统一改成源图**：
      * 每一遍都是"源图 + 该区域的条件"，互不叠加，质量不再随区域数量下降；
      * 合成仍然逐区域进行（只在该区域的掩版内取重绘结果），**区域外逐像素等于当前结果**，
        因此姿态、构图、背景的保护不受影响；
      * 代价是各区域只看得到源图，看不到彼此的改动——但区域本来就互不重叠，
        最终像素仍由逐区域合成得到。

    传 `source_image=None` 时退回旧行为（在累积结果上重绘），便于对照。
    """
    if not regions:
        return base_result, []
    # 线条类输入整条跳过：区域强化那一遍用的图像是**已经生成过的**结果，
    # 它的高频占比已经降到线稿阈值以下，于是"按图像统计量判断是否线稿"在这里会失效
    # （实测：settings 里的 LINEART_REGION_REFINE=False 被绕过，线稿仍被重绘成一块发白色块）。
    # 判据改用**主体分析给出的媒介**——那是源图的属性，不会因为中间结果而改变。
    style = str((subject or {}).get("style") or "").lower()
    if ("line art" in style or "sketch" in style) and not getattr(
            config, "LINEART_REGION_REFINE", True):
        print("[test]   线条类输入：跳过局部强化重绘（那一遍会把线条抹掉）")
        return base_result, []
    if source_image is None or not getattr(config, "REGION_REFINE_FRESH_INIT", True):
        start = base_result
        init_note = "初值=当前结果（旧行为）"
    else:
        start = source_image
        init_note = "初值=源图（每一块独立重绘，不在生成结果上再生成）"
    print(f"[test]   局部强化 {init_note}")
    audits = []
    current = base_result
    for name, info in regions.items():
        box = info.get("box")
        mask = info.get("mask")
        if not box and mask is None:
            continue
        package = subject_transfer.prepare_region(folder, subject, start, box or (0, 0, 1, 1),
                                                  profile, card)
        if not package:
            continue
        if mask is None:
            if getattr(config, "REGION_REFINE_REQUIRE_SAM_MASK", False):
                # 矩形掩版会在人像身上切出方块状接缝（实测：human2 的手臂与肩上出现直角块）。
                # 宁可这一块不做强化（整图那一遍仍然覆盖它），也不留硬边。
                print(f"[test]   区域 {name} 没有像素级掩版，跳过强化（不用矩形掩版，避免硬边）")
                continue
            mask = subject_transfer.region_mask(start, box)
            source = "box"
        else:
            source = "sam_mask"
        overlay = _render(pipe, package["positive"], package["negative"], start,
                          image_utils.extract_canny(start), package["settings"],
                          device, seed + 1)
        merged = subject_transfer.merge_region(current, overlay, mask)
        coverage = sum(mask.histogram()[128:]) / float(mask.size[0] * mask.size[1])
        audits.append({"region": name, "target": info.get("target"), "mask_source": source,
                       "box": [round(v, 4) for v in (box or ())],
                       "strength": package["settings"]["strength"],
                       "lora_scale": package["settings"]["lora_scale"],
                       "mask_coverage": round(coverage, 4),
                       "fresh_init": start is source_image,
                       "sam_refined": bool(info.get("sam_refined")),
                       "sam_note": info.get("sam_note", "")})
        print(f"[test]   局部强化 {name}({source}): 覆盖 {coverage:.1%}", flush=True)
        current = merged
    return current, audits


def attach_additions(pipe, folder, subject, source, base_result, subject_mask, profile, card,
                     device, seed, head_box=None, head_mask=None, body_box=None,
                     subject_type=None, surface_mask=None, neck_box=None):
    """非人物"加法"支路。动物与物体走两条不同的路：

    * **动物**：附着点机制——只在头顶（配饰）、颈部（项圈）、胸部（饰带）生成，
      主体本身的形态、花纹、质感逐像素保留。区域由 subject_analyzer.attachment_masks
      锚定头部给出。
    * **物体**：表面图案支路——物体没有可附着的脖颈/头顶，用户要的是
      "把角色配色、配饰、图案、花纹用到杯身的图案上"。实测 SAM 对物体
      （杯子+桌面）会把背景一并算成主体，因此这里**直接用视觉模型的物体框**
      作为表面范围，框外逐像素保留源图。
    """
    if (subject_type == "object" and config.OBJECT_SURFACE and body_box):
        return _surface_decoration(pipe, folder, subject, body_box, base_result,
                                   profile, card, device, seed, surface_mask=surface_mask)
    if subject_type == "object":
        # 物体表面支路没进来时必须说明**为什么**：静默跳过会让整条"加法"看起来
        # 配置齐全却什么都没做（实测踩过——成品与源图逐字节相同，
        # 而日志里只有一句"没有产出任何附着区域"，看不出是哪一步断的）。
        reasons = []
        if not config.OBJECT_SURFACE:
            reasons.append("OBJECT_SURFACE=False")
        if not body_box:
            reasons.append("没有物体框（body_box 为空）")
        print(f"[test]   物体表面图案未执行: {'；'.join(reasons) or '未知原因'}")
    attachments = attachment_masks(subject_mask, head_box=head_box, head_mask=head_mask,
                                   neck_box=neck_box)
    if not attachments:
        print("[test]   附着区域不可用（没拿到头部定位或主体蒙版过小），非人物不做加法")
        return base_result, []
    audits = []
    current = base_result
    for name, mask in attachments.items():
        box = _normalized_bbox(mask, current.size)
        package = subject_transfer.prepare_attachment(folder, subject, current, box, name,
                                                      profile, card)
        if not package:
            continue
        overlay = _render(pipe, package["positive"], package["negative"], current,
                          image_utils.extract_canny(current), package["settings"],
                          device, seed + 2)
        current = subject_transfer.merge_region(current, overlay, mask)
        coverage = sum(mask.histogram()[128:]) / float(mask.size[0] * mask.size[1])
        audits.append({"region": f"attachment_{name}", "mask_coverage": round(coverage, 4),
                       "box": [round(v, 4) for v in box],
                       "strength": package["settings"]["strength"],
                       "lora_scale": package["settings"]["lora_scale"],
                       "control_scale": package["settings"]["control_scale"]})
        # 把每个附着区域本身也落盘：出问题时要能直接看出"改的是哪一块"，
        # 而不是靠猜（实测踩过：猫额头上出现一条灰带，事后无法从记录里定位来源）。
        image_utils.save_image(mask, diagnostic_dir(folder) / (
            f"{folder}_{subject.get('species') or 'subject'}_attach_{name}.png"))
        print(f"[test]   加法附着 {name}: 覆盖 {coverage:.1%}", flush=True)
    if not audits:
        print("[test]   加法支路没有产出任何附着区域")
    return current, audits


def _surface_decoration(pipe, folder, subject, box, base_result, profile, card, device, seed,
                        surface_mask=None):
    """物体表面图案支路：在物体表面画角色的图案，表面之外逐像素保留源图。

    表面范围优先用"物体轮廓内的掩版"（SAM 按物体框分割 + 最大连通块 + 覆盖率检查），
    拿不到可信掩版时才退回用视觉模型的框——实测杯子这类物体会把整张桌面算进掩版，
    直接用掩版会把图案画到背景上。
    """
    normalized = _clamp_box(box)
    if normalized is None:
        print("[test]   物体框不可用，跳过表面图案")
        return base_result, []
    # 视觉模型的物体框会漂移（实测同一张杯子图有时框住杯子、有时框到桌面），
    # 用边缘密度把它挪到真正有物体的位置；判据只用图像自身统计量。
    from vision.detail_regions import refine_object_box
    normalized, box_note = refine_object_box(base_result, normalized)
    print(f"[test]   物体表面范围: {box_note}")
    if surface_mask is not None:
        mask = surface_mask.convert("L")
        source = "subject_mask"
        if mask.size != base_result.size:
            mask = mask.resize(base_result.size, Image.NEAREST)
    else:
        mask = subject_transfer.region_mask(base_result, normalized, feather=0)
        source = "model_box"
    package = subject_transfer.prepare_surface(folder, subject, base_result, normalized,
                                               profile, card)
    if not package:
        return base_result, []
    overlay = _render(pipe, package["positive"], package["negative"], base_result,
                      image_utils.extract_canny(base_result), package["settings"],
                      device, seed + 3)
    merged = subject_transfer.merge_region(base_result, overlay, mask)
    coverage = sum(mask.histogram()[128:]) / float(mask.size[0] * mask.size[1])
    audit = [{"region": "surface", "subject_type": "object", "mask_source": source,
              "box": [round(v, 4) for v in normalized], "mask_coverage": round(coverage, 4),
              "strength": package["settings"]["strength"],
              "lora_scale": package["settings"]["lora_scale"],
              "control_scale": package["settings"]["control_scale"]}]
    print(f"[test]   表面图案({source}): 覆盖 {coverage:.1%}")

    # ---- 确定性标志承载（真实像素贴合）----
    # 扩散那一步只负责配色氛围：实测把承载条件调到与训练同形、
    # 概念强度 1.45/2.5、重绘强度 0.62/0.80，杯子图改写 81~85% 却**画不出任何标志**
    # （概念学的是"角色裁片"，缺少"画到物体表面"的承载语义）。
    # 而用户要的是"一眼可辨为该角色独有标志"，因此这里把**用户参考图里标志的真实像素**
    # 按可承载区域贴上去：形状与颜色 100% 来自真实素材。
    if getattr(config, "MARK_CARRY_COMPOSITE", False):
        try:
            from vision import carry_compose
            import numpy as _np
            # 可承载区域：默认用**物体框内缩一圈**（保守）。
            # 为什么默认不用像素级掩版：实测该掩版与图像不同尺寸/未对齐，
            # 缩放到画布后换算出的位置落在杯身**左侧之外**，图案被贴到了背景上；
            # 而物体框的位置与画面上的杯子一致（人工核对标注图
            # `output/_probe/carry/cup_regions.png`）。位置宁可保守，也不要贴错地方。
            box_left = int(normalized[0] * merged.size[0])
            box_top = int(normalized[1] * merged.size[1])
            box_right = int(normalized[2] * merged.size[0])
            box_bottom = int(normalized[3] * merged.size[1])
            # 优先**在当前画布上**用图像统计量定位主体表面：归一化框可能来自
            # 另一尺寸/另一帧的检测，换算过来会整体偏移（实测把图案贴到了杯子两侧背景）。
            # 框只作为搜索范围，边界由画布自身统计量决定。
            region_array, region_note = carry_compose.subject_surface_region(
                merged, hint_box=(box_left, box_top, box_right, box_bottom),
                min_pixels=int(config.MARK_CARRY_MIN_REGION_PIXELS))
            if region_array is None:
                # 兜底：物体框内缩一圈（保守，但至少落在框里）
                inset = float(getattr(config, "MARK_CARRY_BOX_INSET", 0.12))
                pad_x = int((box_right - box_left) * inset)
                pad_y = int((box_bottom - box_top) * inset)
                region_array = _np.zeros((merged.size[1], merged.size[0]), dtype=bool)
                region_array[box_top + pad_y:box_bottom - pad_y,
                             box_left + pad_x:box_right - pad_x] = True
                origin = "model_box_inset"
                print(f"[test]   可承载区域（框内缩兜底）: {region_note}")
            else:
                origin = "image_statistics"
                print(f"[test]   可承载区域: {region_note}")
            if int(region_array.sum()) < int(config.MARK_CARRY_MIN_REGION_PIXELS):
                print(f"[test]   确定性承载跳过: 可承载区域过小"
                      f"（{int(region_array.sum())} 像素）")
            else:
                composed, carry_audit = carry_compose.apply_carry_plan(
                    folder, merged, subject_type="object", region_mask=region_array)
                print(carry_compose.describe(carry_audit))
                audit.append({"region": "mark_carry", "origin": origin,
                              "region_pixels": int(region_array.sum()),
                              # 把区域掩版一并带出：主循环的合成阶段需要它来判定
                              # "这块是可以改写的"，否则贴好的图案会被还原成源图
                              # （实测：出图与完全不承载逐像素相同，日志却显示铺放成功）。
                              "region_array": region_array,
                              "items": carry_audit})                # 承载结果必须**只落在可承载区域内**：区域外逐像素保留上一步结果，
                # 否则图案会糊到背景上（与"只做加法"的要求冲突）。
                # 做法是把"承载带来的改动量"限制在区域内，而不是整图替换 —
                # 直接 `Image.composite` 会把区域外也换成承载图的像素，
                # 而那一步的输入与输出在区域外本来就应当相同。
                before = _np.asarray(merged.convert("RGB")).astype(_np.int16)
                after = _np.asarray(composed.convert("RGB")).astype(_np.int16)
                delta = (after - before) * region_array[:, :, None]
                merged = Image.fromarray(
                    _np.clip(before + delta, 0, 255).astype(_np.uint8), mode="RGB")
                outside = float(_np.abs(delta[~region_array]).mean()) if \
                    (~region_array).any() else 0.0
                print(f"[test]   确定性承载已限制在可承载区域内"
                      f"（区域外平均改动 {outside:.3f}）")
        except Exception as exc:      # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[test]   确定性承载失败: {type(exc).__name__}: {exc}")
    return merged, audit


def _union_mask(regions, size):
    """把多个区域掩版并成一张（用于"只对改写区域做处理"的场合）。"""
    if not regions:
        return None
    import numpy as np
    from PIL import Image
    merged = None
    for mask in regions.values():
        if mask is None:
            continue
        layer = mask.convert("L")
        if layer.size != size:
            layer = layer.resize(size, Image.NEAREST)
        array = np.asarray(layer) > 128
        merged = array if merged is None else (merged | array)
    if merged is None:
        return None
    return Image.fromarray((merged * 255).astype(np.uint8), mode="L")


def _clamp_box(box):
    """把归一化框收进 [0,1] 且保证有最小面积；不可用返回 None。"""
    if not box or len(box) != 4:
        return None
    left, top, right, bottom = (float(max(0.0, min(1.0, v))) for v in box)
    if right - left < 0.04 or bottom - top < 0.04:
        return None
    return (left, top, right, bottom)


def _expand_box(box, ratio=0.06):
    """把归一化框按边长比例外扩（用于"掩版裁进框范围"这类场合，给一点容差）。"""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    return (max(0.0, left - width * ratio), max(0.0, top - height * ratio),
            min(1.0, right + width * ratio), min(1.0, bottom + height * ratio))


def _normalized_bbox(mask, size):
    bbox = mask.convert("L").point(lambda v: 255 if v >= 128 else 0).getbbox()
    if not bbox:
        return (0.0, 0.0, 1.0, 1.0)
    width, height = size
    left, top, right, bottom = bbox
    return (left / width, top / height, right / width, bottom / height)


def main():
    started = time.time()
    image_utils.create_dirs()
    folder = config.CHARACTER_ID
    dtype = torch.float16 if config.MIXED_PRECISION == "fp16" else torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = resolve_inputs()
    if not inputs:
        print("[test] 没有可处理的输入图片")
        return
    print(f"[test] 输入 {len(inputs)} 张: {[os.path.basename(p) for p in inputs]}")

    weight_path = resolve_weight(folder)
    profile = semantics.published_profile(folder)
    if profile is None:
        raise FileNotFoundError(
            f"角色 {folder} 的角色包缺少训练时的语义档案快照，请重新训练")
    card = cards_module.build_card(profile)
    print(cards_module.describe(card))

    # ---- 准备阶段：读取并统一尺寸（保留完整构图，不裁剪）----
    sources = {}
    for path in inputs:
        image = image_utils.load_image(path)
        image = image_utils.resize_fit(image, config.INFERENCE_RESOLUTION,
                                       max_long=config.INFERENCE_MAX_LONG)
        sources[path] = image
        print(f"[test] {os.path.basename(path)} -> {image.size[0]}x{image.size[1]}")

    # ---- VLM 段：主体理解 + 局部强化区域定位（用完立刻卸载，8GB 显存必须分时）----
    subjects = {}
    region_boxes = {}
    client = None
    if config.USE_VLM:
        from vlm.vlm_utils import VLMClient
        client = VLMClient()
        if not client.available:
            raise RuntimeError(
                f"VLM 不可用（{client.load_error}）。主体理解与保留条件依赖它；"
                "如需强行继续，请在 config 中显式设置 USE_VLM=False（会退化为无主体类型的通用路径）。")
    try:
        for path in inputs:
            subjects[path] = analyze_subject(client, sources[path])
            print(f"[test] {os.path.basename(path)}: type={subjects[path].get('subject_type')} "
                  f"style={subjects[path].get('style')} framing={subjects[path].get('framing')}")
            if getattr(config, "REGION_REFINE", False):
                wanted = subjects[path].get("subject_type") in {"human", "unknown"}
                wanted = wanted or (subjects[path].get("subject_type") in {"animal", "object"}
                                    and getattr(config, "REGION_REFINE_NON_HUMAN", False))
                # **定位与"强化重绘"是两件事**，不能共用一个开关（实测踩过）：
                # `REGION_REFINE_NON_HUMAN=False` 的本意是"不要在非人物身上做
                # 高强度替换式重绘"（实测会把猫头顶的花纹盖成一块灰斑），
                # 但早期它同时把**区域定位**也关掉了，于是 `body_box` 永远为空——
                # 物体表面图案支路因为"没有物体框"直接跳过，
                # cat/cup 的成品与源图逐**字节**相同（max diff = 0）。
                # 非人物的"加法"本来就要靠这些框：物体表面图案需要物体框，
                # 动物附着需要颈部框。因此定位按"是否需要加法"来决定。
                needs_regions = (getattr(config, "ATTACH_NON_HUMAN", False)
                                 or getattr(config, "OBJECT_SURFACE", False))
                if not wanted and needs_regions:
                    wanted = True
                    print(f"[test] {os.path.basename(path)}: 为非人物加法定位区域"
                          f"（不启用强化重绘）")
                if wanted:
                    # 区域定位必须在 VLM 段内完成（随后 VLM 会被卸载以腾出显存）
                    region_boxes[path] = locate_regions(client, sources[path], subjects[path])
    finally:
        if client is not None:
            client.unload()
            client = None
            gc.collect()
            torch.cuda.empty_cache()

    # ---- 分割段：非人物限定编辑范围；人物用 SAM 细化局部重绘区域 ----
    sam = None
    need_subject_mask = any(subjects[p].get("subject_type") in {"animal", "object"}
                            for p in inputs)
    need_region_mask = any(region_boxes.get(p) for p in inputs)
    if (need_subject_mask or need_region_mask) and config.USE_SAM:
        sam = load_sam2()
        if sam is None and need_subject_mask:
            raise RuntimeError(
                "需要主体蒙版的输入（动物/物品）存在，但 SAM2 不可用；"
                "拒绝在没有蒙版的情况下继续，否则主体外的背景与画风会被改写。")
        if sam is None:
            print("[test] SAM2 不可用：局部重绘退回矩形掩版（范围更粗，但不会静默不做）")
    masks = {}
    surface_masks = {}
    pending_silhouette = set()      # 需要"生成差异法"求主体轮廓掩版的输入
    if sam is not None:
        for path in inputs:
            if subjects[path].get("subject_type") in {"animal", "object"}:
                masks[path] = segment_subject(sources[path], sam)
                print(f"[test] {os.path.basename(path)}: 主体蒙版 "
                      f"{'已取得' if masks[path] else '不可靠，将跳过该图'}")
            # 物体：表面范围优先用**边缘密度选出的主体分割**——视觉模型的物体框会漂移
            # （同一张杯子图有时框住杯子、有时框到右下角桌面），而按 IoU 选主体又会选中
            # 平滑的桌面；物体自身的边缘密度（杯口/把手/高光）才是可用的判据。
            # 拿不到可信掩版时才退回自动分割 / 视觉模型框。
            if subjects[path].get("subject_type") == "object":
                body_box = (region_boxes.get(path) or {}).get("body", {}).get("box")
                if not config.OBJECT_SURFACE_USE_SAM_MASK:
                    print(f"[test] {os.path.basename(path)}: 物体表面按配置不使用分割掩版，"
                          "改用物体框限定范围")
                else:
                    auto_mask = segment_subject(sources[path], sam)
                    ok, note = plausible_subject_mask(
                        auto_mask, max_coverage=config.OBJECT_SURFACE_MAX_COVERAGE)
                    if ok:
                        surface_masks[path] = auto_mask
                        print(f"[test] {os.path.basename(path)}: 物体表面用自动分割掩版（{note}）")
                    elif auto_mask is not None and body_box:
                        # 自动分割的覆盖率过大（实测杯子这张图 SAM 给的是整幅画面，63%），
                        # 但它的像素里**确实包含**物体；而视觉模型的框又会漂移、直接当范围用
                        # 会把图案画到背景上。折中：把掩版裁进"框外扩一档"的范围——
                        # 这样既保留物体轮廓，又不会把整片背景带进来。
                        clipped = _intersect(subject_mask=auto_mask,
                                             region_mask=subject_transfer.region_mask(
                                                 sources[path], _expand_box(body_box, 0.06),
                                                 feather=0))
                        if clipped is not None:
                            surface_masks[path] = clipped
                            print(f"[test] {os.path.basename(path)}: 自动分割掩版超出范围"
                                  f"（{note}），已裁进物体框外扩范围后使用")
                    if surface_masks.get(path) is None:
                        print(f"[test] {os.path.basename(path)}: 掩版不可用（{note}），"
                              "退回用物体框限定范围")
        # 用 SAM 把"大致框"细化成像素级掩版（只在 VLM 段之后、扩散段之前做一次）
        for path in inputs:
            regions = region_boxes.get(path) or {}
            if not regions:
                continue
            refined = region_masks(sources[path], sam, regions)
            for name, (mask, audit) in refined.items():
                regions[name].update({"mask": mask, "sam_refined": audit.get("refined", False),
                                      "sam_note": audit.get("reason", "")})
                if mask is None:
                    print(f"[test] {os.path.basename(path)}: {name} 的 SAM 细化未采用"
                          f"（{audit.get('reason')}），退回矩形掩版")
            # 非人物：区域重绘必须限制在主体内，否则会连背景一起重画。
            # 主体蒙版已经算出来了，这里直接取交集，不再多问一次模型。
            subject_type = subjects[path].get("subject_type")
            if subject_type in {"animal", "object"} and masks.get(path) is not None:
                for name in list(regions):
                    current = regions[name].get("mask")
                    base = masks[path] if current is None else current
                    regions[name]["mask"] = _intersect(subject_mask=masks[path],
                                                       region_mask=base)
                    regions[name]["clipped_to_subject"] = True
                    print(f"[test] {os.path.basename(path)}: {name} 掩版已限制在主体内")
            # 人物/未知：再要一张**沿人物有机轮廓**的整体掩版。
            # 没有它就只能按"头发+衣服"两块合成，脸与皮肤仍是源图，于是成品里
            # "写实脸 + 动漫衣服"拼在一起——这正是"分界没做好、画风不一致"的来源。
            # 掩版由"生成差异 ∩ 区域邻域"给出，因此必须等到扩散结果出来之后才能算，
            # 这里只登记"稍后要算"的意图。
            if (subject_type in {"human", "unknown"}
                    and getattr(config, "SUBJECT_SILHOUETTE_COMPOSITION", False)):
                pending_silhouette.add(path)
        unload_sam(sam)
        sam = None
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 生成段 ----
    concept_path, _concept_entry = resolve_concept_weight(
        folder, [subjects[p].get("subject_type") for p in inputs])
    pipe, adapter_audit = build_pipeline(weight_path, device, dtype, concept_path=concept_path)
    # 文生图管线（与 img2img 共享 UNet 与文本编码器，因此 LoRA 一起生效）。
    # 用途：曲面承印那一步要"让模型画一张角色的平面纹样"——那是**文生图**任务，
    # 对 img2img 管线调用会因缺 `image` 直接报
    # `image must be passed and be one of PIL image, ...`（实测）。
    # 这里预先装配好并显式传给 `surface_print.generate_design`，
    # 避免在调用点临时装配（那条路径实测也失败过一次，同样报 TypeError）。
    surface_text_pipe = None
    try:
        surface_text_pipe, _sctl = model_utils.build_text2img(
            {"pipe": pipe, "device": device, "dtype": dtype}, with_controlnet=False)
        print("[test] 已装配文生图管线（供曲面承印生成角色纹样）")
    except Exception as exc:      # noqa: BLE001
        print(f"[test] 文生图管线装配失败（曲面承印将退回确定性承载）: "
              f"{type(exc).__name__}: {exc}")
    references = []
    reference_audit = {"enabled": False}
    if config.USE_REFERENCE_ENHANCEMENT:
        references = reference_images(folder, profile, card)
        if references:
            reference_audit = model_utils.load_reference_adapter(
                pipe, [config.REFERENCE_FACE_SCALE, config.REFERENCE_COSTUME_SCALE])
            reference_audit["enabled"] = True
            reference_audit["references"] = [r["rel_path"] for r in references]
        else:
            reference_audit["reason"] = "没有可用参考图"

    # ---- 初值配色改写：把被点名区域的初值改向角色配色（只改色度，保留明暗与纹理）----
    # 只对人物做：非人物的设计是"加法"（加配饰/图案/点缀），把主体整块染成角色配色
    # 属于"替换"而不是"加法"。实测：猫被整只染成蓝色，正是这里造成的。
    color_hints = {}
    if config.COLOR_HINT:
        for path in inputs:
            if subjects[path].get("subject_type") in {"animal", "object"}:
                color_hints[path] = {"applied": False,
                                     "reason": "非人物：配色改写属于替换而非加法，已跳过"}
                continue
            region_masks_for_image = {}
            for name, info in (region_boxes.get(path) or {}).items():
                if info.get("mask") is None:
                    continue
                # **只在模型真正定位到的区域上做配色锚定**：头部定位失败时退回的
                # "整幅上半部"会把整个人物框进去（实测：线稿样例的裙子、脸颊、
                # 手臂全被涂成角色色）。宁可不锚定，也不要乱涂。
                if not info.get("located", True):
                    color_hints.setdefault(path, {"applied": False, "regions": [],
                                                  "reasons": []})
                    color_hints[path]["reasons"].append(
                        f"{name}: 定位是退化假设，跳过配色锚定")
                    continue
                # 掩版**合理性**检查：要挡住的是"整幅上半身"这类退化掩版，
                # 而不是"头发本来就铺得很开/垂得很长"的正常掩版。
                #
                # 实测（human2，2026-10-05）：长直发被风吹开，头发掩版宽 70%、面积 22%、
                # bbox 从 y=196 一直垂到画面底部。旧判据只看**宽度**就把它判成
                # "不像头发区"直接跳过，于是**发色锚定从来没有生效过**——
                # 出图头发一直是源图的深棕色，看起来像"权重完全没学到角色"，
                # 而真正原因在这条几何判据上。
                #
                # 换成宽度+面积+触底的组合判据后又踩了一次：这张图的头发**确实**垂到画幅底部，
                # 触底判据同样误杀。宽度、面积、触底都只是**代理指标**，
                # 它们区分不了"一大片头发"与"头发+躯干混在一起"。
                #
                # 真正有区分度的是**材质同质性**：头发是一整片同色材质，
                # 而退化掩版必然同时包含头发、皮肤与衣物三种材质。
                # 因此改用"颜色离散度"判据：掩版内像素在 LAB 上的标准差超过阈值
                # 就说明它混了多种材质，判为退化掩版。阈值可配，默认 34。
                spread_limit = float(getattr(config, "COLOR_ANCHOR_MAX_MASK_SPREAD", 34.0))
                layer = info["mask"].convert("L")
                array = np.asarray(layer) > 128
                if array.any():
                    rgb = np.asarray(sources[path].convert("RGB"))
                    pixels = rgb[array].reshape(-1, 1, 3)
                    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3)
                    spread = float(np.mean(lab.std(axis=0)))
                    if spread > spread_limit:
                        color_hints.setdefault(path, {"applied": False, "regions": [],
                                                      "reasons": []})
                        color_hints[path]["reasons"].append(
                            f"{name}: 掩版内颜色离散度 {spread:.1f}（>{spread_limit:.1f}），"
                            "混了多种材质，判为退化掩版，跳过配色锚定")
                        continue
                # 极端兜底：掩版几乎覆盖整幅画面时无论如何都不是"某个区域"
                coverage_limit = float(getattr(config, "COLOR_ANCHOR_MAX_MASK_COVERAGE", 0.60))
                if array.any():
                    coverage = float(array.mean())
                    if coverage > coverage_limit:
                        color_hints.setdefault(path, {"applied": False, "regions": [],
                                                      "reasons": []})
                        color_hints[path]["reasons"].append(
                            f"{name}: 掩版覆盖 {coverage:.0%}（>{coverage_limit:.0%}），"
                            "接近整幅画面，跳过配色锚定")
                        continue
                # 服装区默认不锚定：它与"整幅重绘"叠加会把衣服整体刷成角色色，
                # 而用户要的是"头发与服装结构像角色"，不是把画面调色。
                if name not in set(getattr(config, "COLOR_ANCHOR_REGIONS",
                                           ("head", "hair", "face"))):
                    color_hints.setdefault(path, {"applied": False, "regions": [],
                                                  "reasons": []})
                    color_hints[path]["reasons"].append(f"{name}: 不在锚定区域列表内，跳过")
                    continue
                region_masks_for_image[name] = info.get("mask")
            if not region_masks_for_image:
                continue
            # 配色锚定是**可选的画质增强**，它出问题不能带走整批出图。
            # 实测踩过：`color_hint` 内部一个未赋值变量抛 `UnboundLocalError`，
            # 由于这里没有兜底，**整批 4 张图一张都没出**——
            # 一个"锦上添花"的步骤不该有这种权限。现在失败就退回未锚定的源图，
            # 并把原因如实记进审计。
            try:
                hinted, hint_audit = color_hint(sources[path], region_masks_for_image,
                                                profile)
            except Exception as exc:              # noqa: BLE001
                print(f"[test] {os.path.basename(path)}: 配色锚定失败，"
                      f"改用未锚定的源图（{type(exc).__name__}: {exc}）", flush=True)
                traceback.print_exc()
                color_hints[path] = {"applied": False, "regions": [],
                                     "reasons": [f"锚定失败: {type(exc).__name__}: {exc}"]}
                continue
            if color_hints.get(path):
                # 保留"为什么跳过"的审计，同时合并真正执行的区域
                hint_audit = dict(hint_audit)
                hint_audit["reasons"] = (color_hints[path].get("reasons") or []) + \
                    list(hint_audit.get("reasons") or [])
            color_hints[path] = hint_audit
            if hint_audit["applied"]:
                sources[path] = hinted
                stem = f"{folder}_{Path(path).stem}"
                image_utils.save_image(
                    hinted, diagnostic_dir(folder) / f"{stem}_color_hint_init.png")

    results, records = {}, []
    try:
        for index, path in enumerate(inputs):
            subject = subjects[path]
            non_human = subject.get("subject_type") in {"animal", "object"}
            if non_human and not masks.get(path):
                print(f"[test] {os.path.basename(path)}: 主体蒙版不可靠，跳过（不用错误掩膜强行修改）")
                records.append({"source": path, "status": "skipped_unreliable_mask"})
                continue
            package = subject_transfer.prepare(folder, subject, sources[path], profile, card)
            # 适配器强度**只在一处决定**（`apply_subject_scales`），并把实际提交的值
            # 记进 `adapter_audit["per_image"]`：出图上"权重没起作用"这类问题，
            # 靠日志里的这两个数字才能立刻分辨
            # （实测踩过：身份权重被后续渲染覆盖成 0.85，日志却看起来完全正常）。
            # 注意 `adapter_audit` 是 init 段构造的 **dict**（记录加载了什么权重），
            # 不是 list——早期写成 `.append` 会在第一张图上直接崩掉。
            scales_applied = apply_subject_scales(pipe, package["settings"])
            per_image = adapter_audit.setdefault("per_image", {})
            per_image[os.path.basename(path)] = {
                "subject_type": subject.get("subject_type"),
                "scale_target": package["settings"].get("scale_target"),
                **{k: round(v, 3) for k, v in scales_applied.items()}}
            print(f"[test] {os.path.basename(path)}: 适配器强度 "
                  f"身份={scales_applied.get('default')} 概念={scales_applied.get('concept')}",
                  flush=True)
            if index == 0:
                print(subject_transfer.describe(package))
            print(f"[test] {os.path.basename(path)}: {subject.get('subject_type')} -> 迁移中",
                  flush=True)
            result = generate_one(pipe, package, sources[path], device,
                                  config.INFERENCE_SEED, index, references=references)
            region_audit, attach_audit = [], []
            # 非人物"加法"的两条支路**各有自己的开关**，不能用一个总闸门控制：
            #   * 物体表面图案：`config.OBJECT_SURFACE`
            #   * 动物附着配饰：`config.ATTACH_NON_HUMAN`
            # 实测踩过（代价很大）：早期这里只判 `config.ATTACH_NON_HUMAN`，
            # 而该开关因为"动物配饰渲染不出可辨认形状"被置为 False，
            # 于是**物体表面图案支路从来没运行过**——cat/cup 的 `region_refine`
            # 与 `attachment` 都是空，成品与源图逐**字节**相同（max diff = 0），
            # 而 `OBJECT_SURFACE` 一直是 True、承载计划也有 3 条分配，
            # 整条"加法"链路看起来配置齐全却什么都没做。
            want_surface = (subject.get("subject_type") == "object"
                            and config.OBJECT_SURFACE)
            allow_refine = (not non_human) or config.REGION_REFINE_NON_HUMAN
            if want_surface or (non_human and config.ATTACH_NON_HUMAN):
                # 加法支路：只在头顶/颈部/胸部施加概念条件，主体其余部分保留
                head_info = (region_boxes.get(path) or {}).get("head") or {}
                body_info = (region_boxes.get(path) or {}).get("body") or {}
                neck_info = (region_boxes.get(path) or {}).get("neck") or {}
                result, attach_audit = attach_additions(
                    pipe, folder, subject, sources[path], result, masks[path], profile, card,
                    device, config.INFERENCE_SEED,
                    head_box=head_info.get("box"), head_mask=head_info.get("mask"),
                    body_box=body_info.get("box"), subject_type=subject.get("subject_type"),
                    surface_mask=surface_masks.get(path), neck_box=neck_info.get("box"))
            elif region_boxes.get(path) and allow_refine:
                # 人物（或显式允许的非人物）走区域强化。
                # 非人物区域现在**也会被定位**（加法支路需要那些框），
                # 但定位不等于要在这里做替换式重绘——实测那样会把猫头顶的花纹
                # 盖成一块与周围毛发不连续的灰斑，与"只做加法"的要求冲突。
                # 因此非人物只有 `REGION_REFINE_NON_HUMAN=True` 时才走这一支。
                result, region_audit = refine_regions(
                    pipe, folder, subject, result, region_boxes[path], profile, card,
                    device, config.INFERENCE_SEED, source_image=sources[path])
            # 主体掩版在扩散结果之后才确定，因此这里只保留原始结果，
            # 合成统一放在掩版求出来之后（见下方 composition_mask 段）。
            raw_result = result
            # 人物：用"生成差异 ∩ 区域邻域"求主体轮廓掩版（贴着真实轮廓），
            # 在扩散结果出来之后才算。差异法失败时退回按点名区域合成。
            if path in pending_silhouette:
                # 注意作用域：循环里用的是 `regions`（本图更新的那份），不是 `region_boxes`
                # ——后者是 VLM 段留下的原始字典。用错名字会让这里拿到别的图的数据。
                silhouette = subject_silhouette(
                    raw_result, {n: i.get("mask") for n, i in regions.items()},
                    reference=sources[path])
                if silhouette is not None:
                    masks[path] = silhouette
                    print(f"[test] {os.path.basename(path)}: 主体轮廓掩版已取得"
                          f"（合成按整块主体，背景逐像素保留）", flush=True)
                else:
                    print(f"[test] {os.path.basename(path)}: 未取得整体轮廓掩版，"
                          "退回按点名区域合成")
            # 细节保留：不需要改的像素还原成源图（涂抹感的实测根因）
            edited_regions = {}
            if non_human:
                edited_regions = attachment_masks(masks.get(path))
                # **确定性承载的区域也要算进"允许改写的地方"**。
                # 实测踩过：非人物的 `attachment_masks()` 在附着件关闭时返回空，
                # 于是下面的 `_compose` 认为"没有任何区域该被改写"，
                # 把已经贴好的标志图案整块**还原成源图**——
                # 出图与"完全不承载"逐像素相同，而日志里却显示"铺放 12 次"。
                carry_regions = [item.get("region_array") for item in (attach_audit or [])
                                 if item.get("region") == "mark_carry"]
                for array in carry_regions:
                    if array is None:
                        continue
                    edited_regions = dict(edited_regions or {})
                    edited_regions["mark_carry"] = Image.fromarray(
                        (array * 255).astype("uint8"), mode="L")
                composition_mask = masks.get(path)
            elif masks.get(path) is not None and getattr(
                    config, "SUBJECT_SILHOUETTE_COMPOSITION", False):
                # 人物：整块主体都取自生成结果（否则脸/皮肤仍是源图，
                # 与角色化的头发衣服拼在一起就是"媒介拼接"）
                edited_regions = {"subject": masks[path]}
                composition_mask = masks[path]
            else:
                edited_regions = {name: info.get("mask")
                                  for name, info in (region_boxes.get(path) or {}).items()
                                  if info.get("mask") is not None}
                composition_mask = None
            if composition_mask is not None:
                # 沿有机轮廓合成：主体内用生成结果，主体外逐像素等于源图（含羽化）。
                # 人物的 composition_mask 本身就是整块主体轮廓；
                # 非人物只用"附着区域 ∩ 主体"（加法支路的设计：主体其余部分不动）。
                limit = _union_mask(edited_regions, sources[path].size)
                if non_human and limit is not None:
                    composition_mask = _intersect(subject_mask=composition_mask,
                                                  region_mask=limit)
                result = _compose(sources[path], result, composition_mask,
                                  outward=bool((not non_human) and masks.get(path) is not None))
                if not non_human:
                    image_utils.save_image(
                        composition_mask, diagnostic_dir(folder)
                        / f"{folder}_{Path(path).stem}_subject_mask.png")
            # 画质三道处理（顺序有讲究）：
            #   1) 只在确有变化处采用生成结果 → 未改写区域逐像素等于源图（治"整图发糊"）
            #   2) 改写区域做反锐化 → 补回扩散造成的软化（治"分界处糊—锐突变"）
            #   3) 区域明度对回源图 → 抑制整图重绘顺手提亮（治"淡带"）
            # 主体轮廓合成模式下跳过第 1 步与第 3 步：
            #   * 第 1 步会按"与源图色差"逐像素决定是否采用生成结果，于是在已被主体掩版
            #     处理过的区域内**打出窟窿**（把源图像素塞回生成结果里），实测表现为
            #     "画面像被撕过、一块块补丁"；主体轮廓已经明确，这一步没有增益。
            #   * 第 3 步按区域平均明度整体平移，在"生成结果本来就该更亮"的地方
            #     （浅色头发）会把主体压暗，实测让成品偏灰。
            silhouette_mode = bool(composition_mask is not None and not non_human
                                   and getattr(config, "SUBJECT_SILHOUETTE_COMPOSITION", False))
            if silhouette_mode:
                preserve_audit = {"applied": False,
                                  "reason": "主体轮廓合成模式：掩版已明确，跳过按色差的逐像素筛选"}
                luminance_audit = {"applied": False,
                                   "reason": "主体轮廓合成模式：不做区域明度平移（会压暗浅色主体）"}
            else:
                result, preserve_audit = subject_transfer.limit_change_to_regions(
                    sources[path], result, edited_regions,
                    max_distance=(None if config.REGION_ONLY_COMPOSITION
                                  else config.LIMIT_CHANGE_MAX_DISTANCE))
            result, sharpen_audit = subject_transfer.sharpen_region(
                sources[path], result, _union_mask(edited_regions, sources[path].size))
            if not silhouette_mode:
                # 做过配色锚定的区域**不能**再"对回源图"：锚定有意改了该区域的平均明度，
                # 而对回源图是拿锚定后的图当基准，等于把刚搬过去的明度又搬回来
                # （实测 human2 头部被平移 -43.8，发色锚定被整个抵消）。
                # 注意 `color_hints[path]["regions"]` 里放的是**审计字典**
                # （含 region/target_rgb 等，见 vision/color_hint.py），
                # 直接 set(...) 会报 `unhashable type: 'dict'`（实测）。
                anchored = {str(entry.get("region")) for entry in
                            ((color_hints.get(path) or {}).get("regions") or [])
                            if isinstance(entry, dict) and entry.get("region")}
                result, luminance_audit = subject_transfer.match_region_luminance(
                    sources[path], result, edited_regions, skip_regions=anchored)
            # ---- 确定性标志承载：**最后一步**才贴上去 ----
            # 为什么必须放在最后（实测踩过两次）：贴在中间的结果会被后续环节整块还原——
            #   1. 非人物 `attachment_masks()` 为空时，`_compose` 认为"没有任何区域该被改写"；
            #   2. 即使把承载区域并进 `edited_regions`，`composition_mask`
            #      （主体掩版，实测只覆盖 25%）与它取交集后仍然把贴好的图案裁掉。
            # 放在合成、反锐化、明度匹配**之后**：这是在"已经定稿的主体像素"上作画，
            # 不存在被覆盖的问题，也天然满足"区域外逐像素等于当前结果"。
            carry_arrays = [item.get("region_array") for item in (attach_audit or [])
                            if item.get("region") == "mark_carry" and
                            item.get("region_array") is not None]
            # **兜底**：`_surface_decoration` 在"表面图案包不可用"时会提前返回，
            # 于是 `attach_audit` 里根本没有 `mark_carry` 条目，
            # 承载与配色迁移就整条被跳过——出图与"什么都没做"逐像素相同，
            # 而日志里只有前面那句"表面图案未执行"，看不出承载也没跑（实测踩过）。
            # 这里补一条：只要是非人物物体且承载开启，就自己定位一次主体范围。
            if (not carry_arrays and not non_human
                    and getattr(config, "MARK_CARRY_COMPOSITE", False)
                    and (subjects[path] or {}).get("subject_type") == "object"):
                try:
                    from vision import carry_compose as _carry0
                    _mask0, _note0 = _carry0.subject_appearance_region(
                        result, hint_box=None,
                        min_pixels=int(config.MARK_CARRY_MIN_REGION_PIXELS))
                    if _mask0 is not None:
                        carry_arrays = [_mask0]
                        print(f"[test]   承载兜底定位主体范围: {_note0}")
                    else:
                        print(f"[test]   承载跳过（兜底定位失败）: {_note0}")
                except Exception as _exc:      # noqa: BLE001
                    print(f"[test]   承载兜底定位失败: {type(_exc).__name__}: {_exc}")
            # ---- 曲面承印：**让模型画纹样，几何负责落到器物上** ----
            #
            # 这是响应"效果很差"批评的架构改动。原先的做法是用代码把标志图
            # 缩放后贴到器物上（`carry_compose`），实测必然出现"贴纸感"：
            # 没有曲面透视、没有材质相互作用、不随器物受光变化。
            #
            # 实测三条结论支撑现在的分工：
            #   1. 让模型直接把纹样画到**摄影器物的曲面**上 -> 画不出来
            #      （只得到一层淡杂色，`_probe_diffusion_paint.py` B/C/D 三档）；
            #   2. 让模型在**平面画布**上画角色纹样 -> 画得出来；
            #   3. 提示词**必须写进角色的具体配色**：泛泛要"角色纹样"得到泛用日式
            #      粉樱花（蓝色占比 0.0%），写上配色后蓝色占比 81.6% 且最接近
            #      角色主导色（`_probe_visual_language.py` V0 vs V1）。
            # 因此：模型负责"画什么"，`vision/surface_print.py` 负责
            # 柱面重投影 + 按角色调色板着色 + 跟随器物明暗 + 沿轮廓羽化。
            if (carry_arrays and getattr(config, "SURFACE_PRINT_ENABLED", True)
                    and (subjects[path] or {}).get("subject_type") == "object"):
                try:
                    import numpy as _np
                    from vision import surface_print as _print
                    base_array = _np.asarray(result.convert("RGB"))
                    design_image, design_prompt = _print.generate_design(
                        pipe, folder, palette=_print_palette(folder), device=device,
                        text_pipe=surface_text_pipe)
                    design_array = _np.asarray(design_image.convert("RGB"))
                    # 把生成的纹样落盘：出图不好时要能直接看出"模型画了什么"，
                    # 而不是只从成品反推（本轮实测就吃过这个亏：成品是杂色，
                    # 但没法从记录里判断是纹样本身差还是承印环节把它弄糊了）。
                    try:
                        design_path = diagnostic_dir(folder) / (
                            f"{Path(path).stem}_surface_design.png")
                        image_utils.save_image(design_image, design_path)
                        print(f"[test]   曲面纹样已落盘: {design_path}")
                    except Exception:      # noqa: BLE001
                        pass
                    merge_mask = _np.asarray(carry_arrays[0]).astype(bool)
                    printed, print_audit = _print.apply(
                        design_array, base_array, merge_mask,
                        palette=_print_palette(folder))
                    # ---- 承印之后，把**标志落笔区还原**（标志是视觉主体，底纹是衬底）----
                    #
                    # 实测问题：确定性承载先把标志放好（日志"铺放 3 次"），
                    # 紧接着曲面承印把整个可承载区域整块覆盖成生成出来的底纹，
                    # 于是成品上**一枚标志都看不见**，只剩一片喷绘图案——
                    # 用户看到的就是"规整复制排列的图案"，而不是"角色的标志被自然运用"。
                    # 还原用的是承载阶段留下的快照（标志已融合好、底纹未印），
                    # 因此标志的受光、材质与边缘都保持融合后的样子。
                    restored = 0
                    try:
                        from vision import carry_compose as _carry_restore
                        snapshot = getattr(_carry_restore.apply_carry_plan,
                                           "last_canvas", None)
                        boxes = getattr(_carry_restore.apply_carry_plan,
                                        "last_boxes", []) or []
                        if (snapshot is not None and boxes
                                and getattr(config, "SURFACE_PRINT_RESTORE_MARKS", True)):
                            pad = int(getattr(config, "SURFACE_PRINT_RESERVE_PAD", 6))
                            for box in boxes:
                                left, top, right, bottom = [int(v) for v in box]
                                left = max(0, left - pad)
                                top = max(0, top - pad)
                                right = min(printed.shape[1], right + pad)
                                bottom = min(printed.shape[0], bottom + pad)
                                if right <= left or bottom <= top:
                                    continue
                                printed[top:bottom, left:right] = \
                                    snapshot[top:bottom, left:right]
                                restored += (right - left) * (bottom - top)
                            print(f"[test]   承印后已还原标志落笔区: {restored} 像素"
                                  f"（{len(boxes)} 个框；标志保持为视觉主体）")
                        elif not getattr(config, "SURFACE_PRINT_RESTORE_MARKS", True):
                            print("[test]   承印后不还原标志（SURFACE_PRINT_RESTORE_MARKS=False）")
                        else:
                            print("[test]   承印后无可还原的标志落笔框")
                    except Exception as _restore_exc:      # noqa: BLE001
                        print(f"[test]   承印后还原标志失败（保留承印结果）: "
                              f"{type(_restore_exc).__name__}: {_restore_exc}")
                    delta = _np.abs(printed.astype(_np.int16)
                                    - base_array.astype(_np.int16)).max(axis=2)
                    inside = int(merge_mask.sum())
                    changed = int((delta > 6).sum())
                    outside_max = int(delta[~merge_mask].max()) \
                        if (~merge_mask).any() else 0
                    result = Image.fromarray(printed)
                    print(f"[test]   曲面承印: 提示词={design_prompt[:110]}…")
                    print(f"[test]   曲面承印: 区域内 {inside} 像素，改写 {changed}，"
                          f"区域外最大改动 {outside_max}")
                    attach_audit = list(attach_audit or []) + [
                        {"region": "surface_print", "applied": print_audit.get("applied"),
                         "audit": print_audit, "changed_pixels": changed,
                         "outside_max": outside_max, "prompt": design_prompt,
                         "region_pixels": inside, "restored_pixels": restored}]
                except Exception as exc:      # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    print(f"[test]   曲面承印失败（回退确定性承载）: "
                          f"{type(exc).__name__}: {exc}")
                    carry_arrays = carry_arrays          # 保持回退路径可用
                    fallback_needed = True
                else:
                    fallback_needed = False
                    carry_arrays = []                    # 已承印，不再走贴图
            else:
                fallback_needed = bool(carry_arrays)
            if carry_arrays:
                try:
                    import numpy as _np
                    from vision import carry_compose as _carry
                    # **承载区域用"整个主体"的掩版，而不是"可贴的平坦区"。**
                    # 实测教训：平坦区判据在陶杯上只覆盖杯身左半边一小块，
                    # 用它做配色迁移会得到"一条竖色带"（用户看到的就是这个），
                    # 而且承载步骤还会把区域外的改动全部丢掉。
                    # 主体掩版由 SAM 给出（种子点 = 亮度高分位像素），
                    # 因此配色能覆盖整只杯子，同时不溢出到桌面。
                    carry_mask = carry_arrays[0]
                    appearance, appearance_note = _carry.subject_appearance_region(
                        result, hint_box=None,
                        min_pixels=int(config.MARK_CARRY_MIN_REGION_PIXELS))
                    if appearance is not None:
                        carry_mask = appearance
                        print(f"[test]   承载/配色范围: {appearance_note}")
                    else:
                        print(f"[test]   承载/配色范围退化为可贴区域: {appearance_note}")
                    painted, carry_audit = carry_compose.apply_carry_plan(
                        folder, result, subject_type="object",
                        region_mask=carry_mask)
                    before = _np.asarray(result.convert("RGB")).astype(_np.int16)
                    after = _np.asarray(painted.convert("RGB")).astype(_np.int16)
                    changed = int((_np.abs(after - before).max(axis=2) > 6).sum())
                    result = painted
                    print(carry_compose.describe(carry_audit))
                    print(f"[test]   确定性标志承载已应用（改写 {changed} 像素）")
                    # 配色迁移与承载是**两个动作**，各自单独记一条审计：
                    # 合成一条会让"配色到底有没有生效"无法从记录里判断。
                    style_items = [item for item in carry_audit
                                   if item.get("region") == "style_transfer"]
                    mark_items = [item for item in carry_audit
                                  if item.get("region") != "style_transfer"]
                    for item in (attach_audit or []):
                        if item.get("region") == "mark_carry":
                            item["items"] = mark_items
                            item["applied_pixels"] = changed
                    attach_audit = list(attach_audit or []) + [
                        {"region": "style_transfer_final", "items": style_items,
                         "region_pixels": int(carry_mask.sum()),
                         "applied_pixels": changed}]
                except Exception as exc:      # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    print(f"[test]   确定性承载失败: {type(exc).__name__}: {exc}")
            stem = f"{folder}_{Path(path).stem}"
            out_path = str(result_image_dir(folder) / f"{stem}_result.png")
            image_utils.save_image(result, out_path)
            results[path] = out_path
            records.append({
                "source": path, "result": out_path, "status": "generated",
                "subject": subject, "settings": package["settings"],
                "positive": package["positive"], "negative": package["negative"],
                "hair_boundary": package["hair_boundary"],
                "weight": adapter_audit, "style_clause": package["style_clause"],
                "coverage_gaps": package["coverage_gaps"],
                "region_refine": region_audit,
                # 落盘前剔除内部用的 ndarray（`region_array` 只在本次运行内用于合成判定，
                # 直接 json.dump 会抛 "Object of type ndarray is not JSON serializable"）
                "attachment": [{k: v for k, v in item.items() if k != "region_array"}
                               for item in attach_audit],
                "source_preserve": preserve_audit,
                "luminance_match": luminance_audit,
                "color_hint": color_hints.get(path),
                "reference_enhancement": reference_audit if index == 0 else None,
            })
            print(f"[test] 已保存: {out_path}")
    except Exception:
        # 主循环出错时必须打印完整 traceback：只留 `except` 不打印的话，
        # 终端上只剩最后一行异常信息，看不出是哪个文件哪一行（实测为此多跑了两轮完整流程）。
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            if reference_audit.get("enabled"):
                model_utils.unload_reference_adapter(pipe)
            model_utils.unload_adapters(pipe)
        except Exception as exc:
            print(f"[test] 卸载适配器时提示: {exc}")

    record_path = Path(config.LOGS_DIR) / "generation" / f"{folder}_generation.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(record_path, "w", encoding="utf-8") as stream:
        json.dump({
            "character": folder, "trigger": card.get("trigger"),
            "base_model": config.BASE_MODEL_PATH,
            "weight": adapter_audit,
            "reference_enhancement": reference_audit,
            "seed": config.INFERENCE_SEED,
            "elapsed_seconds": round(time.time() - started, 1),
            "index": {Path(p).stem: r for p, r in results.items()},
            "records": records,
            "note": ("运行完成不等于视觉验收通过：结构化记录与图片都需人工复核"
                     "（方案 §7.7、§11）"),
        }, stream, ensure_ascii=False, indent=2)
    print("=" * 64)
    print(f"[test] 完成 {len(results)}/{len(inputs)} 张，用时 {time.time() - started:.0f}s")
    print(f"[test] 结果目录: {result_image_dir(folder)}")
    print(f"[test] 生成记录: {record_path}（含实际加载权重与条件）")
    print("=" * 64)


def _compose(source, generated, mask, outward=None):
    """主体蒙版合成：主体内用生成结果，主体外逐像素保持源图（含羽化）。

    必须用"内部实心 + 只在外侧渐变"的 alpha（feather_alpha），不能直接高斯模糊整张掩版：
    直接模糊会让掩版外的整圈也变成半透明，于是生成结果里的背景从主体四周渗出来，
    在主体上方形成一圈幽灵般的亮环（实测症状：杯子/猫的头部上方出现亮斑带）。

    outward=True 时改用"掩版向外扩张 + 向外羽化"：过渡带落在主体**外侧**（通常是平滑的
    背景），因此既不会留下硬边，也不会把源图的飘散发丝叠在生成结果上。
    这是主体轮廓合成专用的模式（`SUBJECT_SILHOUETTE_*`）。
    """
    from PIL import Image, ImageFilter
    if mask.size != source.size:
        mask = mask.resize(source.size, Image.NEAREST)
    if outward:
        mask = mask.convert("L").filter(
            ImageFilter.MaxFilter(max(3, int(config.SUBJECT_SILHOUETTE_DILATE) | 1)))
        alpha = subject_transfer.feather_alpha(
            mask, feather=config.SUBJECT_SILHOUETTE_FEATHER, interior=False)
    else:
        alpha = subject_transfer.feather_alpha(mask, feather=config.MASK_FEATHER)
    return Image.composite(generated.convert("RGB"), source.convert("RGB"), alpha)


def _intersect(subject_mask, region_mask):
    """两个掩版取交集（用于把非人物区域重绘限制在主体轮廓内）。"""
    from PIL import ImageChops
    if region_mask is None:
        return subject_mask
    if region_mask.size != subject_mask.size:
        region_mask = region_mask.resize(subject_mask.size, Image.NEAREST)
    return ImageChops.darker(subject_mask.convert("L"), region_mask.convert("L"))


if __name__ == "__main__":
    select_cli_character(allow_inputs=True)
    main()
