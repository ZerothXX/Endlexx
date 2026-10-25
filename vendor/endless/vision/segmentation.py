# -*- coding: utf-8 -*-
"""
vision/segmentation.py —— SAM2 主体分割（规格 §9，含降级）

难点说明：本地 models/sam2 是 **视频模型**（config.json: architectures=['Sam2VideoModel']，
model_type='sam2_video'），但本项目只需要单图主体分割。经核实 transformers 4.56.2
源码，本模块的实现依据如下：

1. Sam2VideoModel 的公共 forward 是"流式视频推理"接口，签名只接受
   inference_session / frame_idx / frame / reverse（modeling_sam2_video.py:1696），
   无法直接传单图 pixel_values + 提示点。
2. 单图分割唯一可用入口是 Sam2VideoModel._single_frame_forward
   （modeling_sam2_video.py:1854），签名与 Sam2Model.forward 完全一致
   （pixel_values / input_points / input_labels / input_boxes / multimask_output），
   返回 Sam2VideoImageSegmentationOutput（modeling_sam2_video.py:603）：
   - iou_scores       (B, point_batch, num_masks=3)   每个 mask 的 IoU 预测分数
   - pred_masks       (B, point_batch, H_low, W_low)  低分辨率最佳 mask
   - high_res_masks   (B, point_batch, 1024, 1024)    高分辨率最佳 mask（1024=image_size）
   - object_score_logits (B, point_batch, 1)          对象出现分数（>0 表示该点处有对象）
   该方法在模型内部被 _use_mask_as_output 正常调用（modeling_sam2_video.py:2078），
   是活跃代码路径而非废弃接口。
3. 提示点坐标链路（无需手工归一化）：
   用户给原图像素坐标 -> Sam2VideoProcessor 等比缩放至 1024 空间
   （processing_sam2_video.py:199-227）-> prompt encoder 内除以 1024 归一化到 [0,1]
   （modeling_sam2_video.py:1139-1154）。
4. 后处理：processor.post_process_masks(masks, original_sizes, ...) 把 1024 空间
   的 mask 去 pad 并 resize 回原图尺寸（image_processing_sam2_fast.py:647-698）。
   注意其内部按 batch 索引 masks[i] 后直接 F.interpolate，因此 masks 必须以
   list（每元素 4D (B, N, H, W)）形式传入；实测 4D 张量直接传会崩（3D 切片
   无法 bilinear 插值到 2 维 size）。
5. torch<2.3 兼容：transformers 4.56 的 fast image processor 在 resize 时调用
   torch.compiler.is_compiling()（image_processing_utils_fast.py:292），该 API
   torch 2.3 才引入；本环境 torch 2.2.1 会 AttributeError。SAM2 只有 fast
   processor（Sam2VideoProcessor.attributes 固定 image_processor_class=
   "Sam2ImageProcessorFast"，processing_sam2_video.py:56-58，无 slow 版本），
   因此 load_sam2 时做幂等 monkey patch（注入 is_compiling = lambda: False）。

自动主体提示（无人工点选）：3×3 网格（含中心点）共 9 个候选点。实测发现
transformers 4.56 的 Sam2VideoModel._single_frame_forward 在 num_objects>1（即
input_points 第二维 >1，本项目 9 个候选点即此形态）时存在内部形状 bug
（object_score_logits 返回 (B,1,N,1) 而 low_res_multimasks 为 (B,N,3,H,W)，
torch.where 广播在第 3 维冲突：'size of tensor a (9) must match the size of
tensor b (256)'），因此不能一次前向传 9 个对象。修复：视觉特征只编码一次
（get_image_embeddings，含 no_memory_embedding 注入，与 _single_frame_forward
内部一致），9 个候选点各自作为单对象（num_objects=1）逐点前向（每点仅
prompt decoder 开销，实测 9 点共约 0.3 秒），语义与"一次前向 9 点"等价。
过滤规则：
- object_score_logits > 0（模型认为该点处存在对象，否则 mask 被强制为全背景）
- mask 面积占比 ∈ [0.05, 0.9]（过滤全图 / 极小碎片，规格 §9）
取剩余候选中 iou_scores 最高者；全部被过滤则返回 None，由上层降级整图 mask（§27）。

显存预算：Sam2VideoModel 约 176M 参数，fp16 权重约 0.35GB；单帧 1024 推理峰值
激活约 1~2GB（RTX 4060 8GB 无压力）。
"""

import gc
import os
import sys

import numpy as np
from PIL import Image
import torch
import cv2

# 保证从任意位置直接运行本文件时都能 import 到项目根目录的模块
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402

# =========================
# 分割参数（规格 §9：过滤全图 / 极小碎片）
# =========================
MIN_MASK_AREA_RATIO = 0.05   # mask 面积占比下限（低于视为碎片）
MAX_MASK_AREA_RATIO = 0.9    # mask 面积占比上限（高于视为全图误分割）

# 自动主体提示：3×3 网格相对坐标（相对原图宽高的比例，含中心点 (0.5, 0.5)）
GRID_REL_POINTS = [
    (0.25, 0.25), (0.50, 0.25), (0.75, 0.25),
    (0.25, 0.50), (0.50, 0.50), (0.75, 0.50),
    (0.25, 0.75), (0.50, 0.75), (0.75, 0.75),
]


def _patch_torch_compiler_for_fast_processor() -> None:
    """幂等兼容补丁：torch<2.3 无 torch.compiler.is_compiling。

    transformers 4.56 的 fast image processor（SAM2 唯一 processor 类型）在 resize
    时调用 torch.compiler.is_compiling()（image_processing_utils_fast.py:292），该
    API 于 torch 2.3 引入；torch 2.2.1 下直接调用会 AttributeError。
    注入返回 False 的占位函数（默认语义：不在编译中），对运行无任何副作用。
    """
    import torch.compiler  # torch 2.2 起 torch.compiler 模块已存在
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]


def load_sam2(path=None):
    """加载本地 SAM2 视频模型（fp16 GPU）及其 processor。

    参数：
        path: 模型目录（默认取 config.SAM_PATH，即 models/sam2）

    返回：
        (model, processor) 元组；任何加载异常（路径不存在 / 权重损坏 /
        无 CUDA / ImportError 等）打印原因并返回 None，调用方据此降级
        （config.USE_SAM=False -> 整图 mask，规格 §9/§27）。
    """
    if path is None:
        path = config.SAM_PATH

    try:
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        # torch<2.3 兼容补丁（见文件头说明第 5 条），必须在 processor 预处理前生效
        _patch_torch_compiler_for_fast_processor()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用（SAM2 fp16 需 GPU 推理，本环境不允许 CPU 大模型推理）")

        processor = Sam2VideoProcessor.from_pretrained(path)
        model = Sam2VideoModel.from_pretrained(path, dtype=torch.float16)
        model.to("cuda")
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        vram_gb = num_params * 2 / (1024 ** 3)  # fp16 = 2 字节/参数
        print(f"[SAM2] 已加载: {path}")
        print(f"[SAM2] 参数 {num_params / 1e6:.1f}M，fp16 权重显存约 {vram_gb:.2f}GB，单帧推理峰值约 1~2GB")
        return model, processor
    except Exception as e:
        print(f"[SAM2] 加载失败（返回 None），原因: {type(e).__name__}: {e}")
        print("[SAM2] 降级建议: 在 config.py 中设置 USE_SAM=False，"
              "主体分割退化为整图 mask（规格 §9/§27，最小模式不受影响）")
        return None


def segment_subject_boxed(image, sam):
    """带主体框与背景反例点的单帧分割；用于"主体大致居中"的角色参考图。

    与 segment_subject 的区别：需要框住主体的矩形并给出背景反例点，因此结果在
    背景复杂时更稳定；实测两者互补，因此本项目只保留这一份 SAM 调用实现，
    由调用方按场景选择入口，而不是各自复制一份分割代码。
    """
    if sam is None:
        return None
    model, processor = sam
    width, height = image.size
    points = [[[[.50 * width, .35 * height], [.02 * width, .02 * height],
                [.98 * width, .02 * height], [.02 * width, .98 * height],
                [.98 * width, .98 * height]]]]
    labels = [[[1, 0, 0, 0, 0]]]
    inputs = processor(images=image, input_points=points, input_labels=labels,
                       input_boxes=[[[.03 * width, .015 * height, .97 * width, .985 * height]]],
                       return_tensors="pt")
    device = next(model.parameters()).device
    with torch.no_grad():
        output = model._single_frame_forward(
            pixel_values=inputs["pixel_values"].to(device, dtype=torch.float16),
            input_points=inputs["input_points"].to(device),
            input_labels=inputs["input_labels"].to(device),
            input_boxes=inputs["input_boxes"].to(device), multimask_output=False)
    post = processor.post_process_masks([output.high_res_masks.float()],
                                        inputs["original_sizes"].tolist(),
                                        mask_threshold=0.0, binarize=False)
    mask = post[0].reshape(-1, height, width)[0].cpu().numpy() > 0
    if not .02 < mask.mean() < .95:
        return None
    return Image.fromarray(mask.astype(np.uint8) * 255)


def largest_component(mask, keep_ratio=None):
    """保留主体自身的连通块，去掉零散噪声。

    实测教训：只保留"最大"一个连通块是错的——SAM 会把一个物体拆成几块
    （杯子案例：杯口圈 + 杯身 + 把手是分开的），只留最大块就只剩杯口圈（10.8%），
    杯身反而没被覆盖。因此这里保留的是"与最大块规模相当的所有连通块"，
    即把属于同一个物体的碎片合并回来，同时丢掉真正的零散噪声。
    """
    if mask is None:
        return None
    try:
        import cv2
    except ImportError:
        return mask
    array = (np.asarray(mask.convert("L")) > 128).astype(np.uint8)
    if array.sum() == 0:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.max())
    keep_ratio = config.SUBJECT_COMPONENT_KEEP_RATIO if keep_ratio is None else keep_ratio
    threshold = max(1, int(biggest * keep_ratio))
    keep = [index + 1 for index, area in enumerate(areas) if area >= threshold]
    merged = np.isin(labels, keep)
    return Image.fromarray(merged.astype(np.uint8) * 255, mode="L")


def plausible_subject_mask(mask, max_coverage=None, min_coverage=None):
    """判断掩版是否像"一个主体"：覆盖率既不能太小也不能太大。

    实测：杯子这类物体在框提示下会把整张桌面算进来（覆盖率过高），
    此时掩版不适合当作"物体表面"，调用方应退化为用框限定范围而不是硬用错掩版。
    返回 (是否可信, 说明)。
    """
    if mask is None:
        return False, "掩版为空"
    coverage = float((np.asarray(mask.convert("L")) > 128).mean())
    max_coverage = config.SUBJECT_MASK_MAX_COVERAGE if max_coverage is None else max_coverage
    min_coverage = config.SUBJECT_MASK_MIN_COVERAGE if min_coverage is None else min_coverage
    if coverage < min_coverage:
        return False, f"掩版覆盖率 {coverage:.1%} 过小（<{min_coverage:.0%}）"
    if coverage > max_coverage:
        return False, f"掩版覆盖率 {coverage:.1%} 过大（>{max_coverage:.0%}），疑把背景算入"
    return True, f"覆盖率 {coverage:.1%}"


def segment_object_surface(image, sam, box):
    """物体表面掩版：按框分割 + 只留最大连通块 + 覆盖率合理性检查。

    返回 (掩版 或 None, 说明)。掩版不可信时返回 None，由调用方退回"用框限定范围"。
    """
    if sam is None or not box:
        return None, "缺少 SAM 或物体框"
    width, height = image.size
    left, top, right, bottom = box
    pixel_box = [left * width, top * height, right * width, bottom * height]
    mask = sam_point_maskbox(image, sam, points=[], box=pixel_box)
    if mask is None:
        return None, "SAM 未给出物体掩版"
    mask = largest_component(mask)
    ok, reason = plausible_subject_mask(mask)
    if not ok:
        return None, reason
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    return mask, reason


def segment_subject(image, sam, rank="iou"):
    """SAM2 自动主体分割（无人工点选），返回单通道 255/0 的 PIL.Image 或 None。

    参数：
        image: PIL.Image（RGB，原图尺寸；坐标以原图为准，processor 负责归一化）
        sam  : load_sam2 返回的 (model, processor) 元组，或 None（直接返回 None）

    流程：
        3×3 网格 9 个候选提示点 -> 视觉特征只编码一次（get_image_embeddings）
        -> 每点作为单对象逐点前向（num_objects=1，规避 transformers 4.56 的
        num_objects>1 形状 bug，见文件头）-> 后处理到原图尺寸
        -> 过滤（对象分数 > 0 且面积占比 ∈ [0.05, 0.9]）-> 取 IoU 分数最高者
        -> 255/0 单通道 mask。

    返回：
        成功返回与 image 同尺寸的单通道 "L" 模式 PIL.Image（前景 255，背景 0）；
        任何失败（sam 为 None / 模型异常 / 全部候选被过滤）打印原因并返回 None，
        由 get_subject_mask_or_full 降级为整图 mask（规格 §27）。

    rank 参数（新增，默认保持旧行为）：
        "iou"    取 IoU 最高的候选（旧行为，兼容既有调用方）；
        "center" 先要求 mask 覆盖图像中心，再按质心距中心最近、IoU 最高排序。
                 实测（input/cup.jpg）：网格点 (0.75,0.25) 落在桌面背景上，SAM 给出
                 IoU=0.992 但内容是"桌面+杯把"；仅按 IoU 选会把背景当成主体。
    """
    if sam is None:
        print("[SAM2] sam 为 None，无法分割")
        return None
    model, processor = sam
    if not isinstance(image, Image.Image):
        raise TypeError(f"[SAM2] image 必须是 PIL.Image，收到: {type(image)}")

    try:
        w, h = image.size
        # 1. 候选提示点：3×3 网格（含中心点），原图像素坐标（x=列, y=行）
        #    格式 [image][object][point][x, y]，每个 object 一个前景点（label=1）
        points = [[[[rx * (w - 1), ry * (h - 1)]] for rx, ry in GRID_REL_POINTS]]
        labels = [[[1] for _ in GRID_REL_POINTS]]

        # 2. 预处理：缩放到 1024 空间 + 坐标归一化（processing_sam2_video.py:67-195）
        inputs = processor(images=image, input_points=points, input_labels=labels, return_tensors="pt")
        try:
            device = next(model.parameters()).device  # 正常取模型所在设备（cuda）
        except (StopIteration, AttributeError):
            # 防御：异常时回退到 CUDA（模型加载时就已 .to("cuda")）
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)

        # 3. 视觉特征只编码一次（get_image_embeddings 内部已加 no_memory_embedding，
        #    与 _single_frame_forward 传 pixel_values 的路径一致）
        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(pixel_values)

        # 4. 逐点前向：每个网格点作为单对象（num_objects=1）调用
        #    _single_frame_forward（modeling_sam2_video.py:1854），规避 transformers
        #    4.56 在 num_objects>1 时的 torch.where 广播 bug（见文件头说明）。
        #    multimask_output=True：每点预测 3 个 mask，模型内部自动取 IoU 最高者。
        #    inputs["input_points"] 形状 (1, 9, 1, 2)，按点切片复用坐标归一化结果。
        all_pts = inputs["input_points"].to(device)
        all_lbs = inputs["input_labels"].to(device)
        high_res_list = []
        iou_list = []
        obj_list = []
        with torch.no_grad():
            for i in range(all_pts.shape[1]):
                outputs = model._single_frame_forward(
                    pixel_values=None,
                    image_embeddings=image_embeddings,
                    input_points=all_pts[:, i:i + 1],
                    input_labels=all_lbs[:, i:i + 1],
                    multimask_output=True,
                )
                high_res_list.append(outputs.high_res_masks.float())
                iou_list.append(outputs.iou_scores[0])            # (1, 3)
                obj_list.append(outputs.object_score_logits[0, :, 0])  # (1,)

        # 5. 后处理：逐点 high_res_masks (1, 1, 1024, 1024) -> 原图尺寸 (1, 1, H, W)
        #    必须包一层 list（fast image processor 按 batch 索引后直接 interpolate）
        #    .float() 避免 fp16 interpolate 的精度/兼容问题
        # 注意：original_sizes 必须转为普通 list（元素为 [H, W]），
        # 传 tensor 会原样进入 F.interpolate 的 size 参数导致类型错误
        orig_sizes = inputs["original_sizes"].tolist()
        post = processor.post_process_masks(
            high_res_list, [orig_sizes[0]] * len(high_res_list),
            mask_threshold=0.0, binarize=False,
        )
        masks = torch.cat([p[0] for p in post], dim=0)   # (N, H, W) float logits
        iou_scores = torch.cat(iou_list, dim=0)          # (N, 3) 每点 3 个 mask 的 IoU
        obj_scores = torch.cat(obj_list, dim=0)          # (N,) 对象出现分数
        best_iou = iou_scores.max(dim=-1).values         # (N,) 每点最佳 mask 的 IoU

        # 6. 过滤 + 选择：对象分数 > 0（模型认为该处有对象）且面积占比在 [0.05, 0.9]
        candidates = []
        for i in range(masks.shape[0]):
            if obj_scores[i].item() <= 0:
                continue
            frac = (masks[i] > 0).float().mean().item()
            if MIN_MASK_AREA_RATIO <= frac <= MAX_MASK_AREA_RATIO:
                candidates.append((best_iou[i].item(), i, frac))

        if not candidates:
            print("[SAM2] 全部候选提示点未通过过滤（无对象或面积占比异常），"
                  "返回 None，上层将降级为整图 mask")
            return None

        if rank == "center":
            # 主体通常包含画面中心；用"是否覆盖中心 + 质心距中心"替代"只看 IoU"，
            # 避免把覆盖大面积背景的候选当成主体。
            cx, cy = (image.width - 1) / 2.0, (image.height - 1) / 2.0
            scored = []
            for iou, i, frac in candidates:
                binary = masks[i] > 0
                contains = bool(binary[int(round(cy)), int(round(cx))].item())
                ys, xs = torch.nonzero(binary, as_tuple=True)
                mx, my = xs.float().mean().item(), ys.float().mean().item()
                dist = (((mx - cx) / max(image.width, 1)) ** 2
                        + ((my - cy) / max(image.height, 1)) ** 2) ** 0.5
                scored.append({"contains_center": contains, "dist": dist, "iou": iou,
                               "index": i, "frac": frac})
            best = max(scored, key=lambda c: (c["contains_center"], -c["dist"], c["iou"]))
            best_idx, best_frac, best_score = best["index"], best["frac"], best["iou"]
            print(f"[SAM2] center 排序选中候选 #{best_idx}"
                  f"（覆盖中心={best['contains_center']}, 质心距={best['dist']:.3f}, "
                  f"IoU={best_score:.3f}）")
        else:
            best_score, best_idx, best_frac = max(candidates, key=lambda t: t[0])
        mask_np = (masks[best_idx] > 0).float().cpu().numpy()
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        print(f"[SAM2] 主体 mask 已生成: 网格点 #{best_idx}（{GRID_REL_POINTS[best_idx]}），"
              f"IoU={best_score:.3f}，面积占比={best_frac:.2f}，尺寸={mask_img.size}")
        return mask_img
    except Exception as e:
        print(f"[SAM2] 分割失败，返回 None，上层将降级为整图 mask。"
              f"原因: {type(e).__name__}: {e}")
        return None


def subject_mask_from_regions(image, sam, masks, pad_ratio=0.03):
    """用已知区域掩版的**质心**作提示点，向 SAM 要一张完整的"人/主体"掩版。

    为什么需要它：人物样例的局部重绘只覆盖"头发 + 衣服"两块，脸、脖子、手臂仍来自源图。
    合成后就会出现"身体是角色、头和皮肤还是原照片"的**媒介拼接**（写实脸 + 动漫衣服），
    这正是用户说的"分界没做好、风格与原图不一致"。要整幅一致，就必须有一张覆盖
    整个人物的有机轮廓掩版，把整块主体替换成生成结果（背景仍逐像素保留）。

    只用掩版的质心做正例点（不依赖文字框、不依赖主体类别假设），
    再按最小外接框稍作外扩作为框提示；结果做闭运算补洞 + 只保留主连通块。
    """
    if sam is None or not masks:
        return None
    valid = [m for m in masks.values() if m is not None]
    if not valid:
        return None
    width, height = image.size
    union = np.zeros((height, width), dtype=bool)
    for mask in valid:
        layer = mask.convert("L")
        if layer.size != image.size:
            layer = layer.resize(image.size, Image.NEAREST)
        union |= np.asarray(layer) > 128
    if union.sum() < 64:
        return None
    ys, xs = np.nonzero(union)
    centre = (float(xs.mean()), float(ys.mean()))
    left, right = float(xs.min()), float(xs.max())
    top, bottom = float(ys.min()), float(ys.max())
    pad_x, pad_y = (right - left) * 0.08, (bottom - top) * 0.08
    box = (max(0.0, left - pad_x), max(0.0, top - pad_y),
           min(float(width - 1), right + pad_x), min(float(height - 1), bottom + pad_y))
    mask = sam_point_maskbox(image, sam, points=[centre], labels=[1], box=box)
    if mask is None:
        return None
    import cv2
    binary = (np.asarray(mask.convert("L")) > 128).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == largest).astype(np.uint8)
    coverage = float(binary.mean())
    if coverage < config.SUBJECT_MASK_MIN_COVERAGE or coverage > config.SUBJECT_MASK_MAX_COVERAGE:
        print(f"[SAM2] 主体轮廓掩版覆盖率 {coverage:.1%} 不合理，放弃使用")
        return None
    print(f"[SAM2] 主体轮廓掩版已生成: 覆盖率 {coverage:.1%}（提示点来自区域质心）")
    return Image.fromarray(binary * 255, mode="L")


def segment_subject_edges(image, sam):
    """按**边缘密度**挑主体候选：适合"物体在纹理背景上"的场合（如杯子在撒满咖啡豆的桌面）。

    为什么需要它：`segment_subject` 按 IoU 取候选，实测杯子那张图上网格点
    (0.75, 0.25) 落在桌面、IoU 高达 0.992，于是主体被当成整张桌面；
    而视觉模型给的物体框又会漂移（同一张图有时框住杯子、有时框到右下角）。
    两类输入都不可靠，因此改用**图像自身**的判别：物体（杯口、把手、高光）的
    边缘密度显著高于平滑的台面，取"边缘密度最高且面积合理"的候选即可。

    返回 (掩版 或 None, 说明)。
    """
    if sam is None:
        return None, "缺少 SAM"
    model, processor = sam
    try:
        w, h = image.size
        points = [[[[rx * (w - 1), ry * (h - 1)]] for rx, ry in GRID_REL_POINTS]]
        labels = [[[1] for _ in GRID_REL_POINTS]]
        inputs = processor(images=image, input_points=points, input_labels=labels,
                           return_tensors="pt")
        device = next(model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)
        with torch.no_grad():
            embeddings = model.get_image_embeddings(pixel_values)
            all_pts = inputs["input_points"].to(device)
            all_lbs = inputs["input_labels"].to(device)
            high_res, ious, objs = [], [], []
            for i in range(all_pts.shape[1]):
                out = model._single_frame_forward(
                    pixel_values=None, image_embeddings=embeddings,
                    input_points=all_pts[:, i:i + 1], input_labels=all_lbs[:, i:i + 1],
                    multimask_output=True)
                high_res.append(out.high_res_masks.float())
                ious.append(out.iou_scores[0])
                objs.append(out.object_score_logits[0, :, 0])
        orig = inputs["original_sizes"].tolist()
        post = processor.post_process_masks(high_res, [orig[0]] * len(high_res),
                                            mask_threshold=0.0, binarize=False)
        masks = torch.cat([p[0] for p in post], dim=0)
        iou_best = torch.cat(ious, dim=0).max(dim=-1).values
        obj = torch.cat(objs, dim=0)
        grey = np.asarray(image.convert("L")).astype(np.float32)
        edges = np.abs(cv2.Laplacian(grey, cv2.CV_32F))
        scored = []
        for i in range(masks.shape[0]):
            if obj[i].item() <= 0:
                continue
            binary = (masks[i] > 0).cpu().numpy()
            frac = float(binary.mean())
            if not MIN_MASK_AREA_RATIO <= frac <= MAX_MASK_AREA_RATIO:
                continue
            scored.append((float(edges[binary].mean()), float(iou_best[i].item()), i, frac))
        if not scored:
            return None, "没有候选通过过滤"
        scored.sort(key=lambda item: -item[0])
        density, iou, index, frac = scored[0]
        binary = (masks[index] > 0).cpu().numpy()
        print(f"[SAM2] 边缘密度选主体: 点 #{index}（{GRID_REL_POINTS[index]}）密度={density:.1f} "
              f"IoU={iou:.3f} 面积占比={frac:.2f}")
        return Image.fromarray((binary * 255).astype(np.uint8), mode="L"), \
            f"边缘密度 {density:.1f}（IoU {iou:.2f}）"
    except Exception as exc:
        print(f"[SAM2] 边缘密度选主体失败: {type(exc).__name__}: {exc}")
        return None, f"失败: {type(exc).__name__}"


def segment_object_by_contour(image, box=None, min_ratio=0.02, max_ratio=0.75):
    """用**连续轮廓**在指定框内找出物体表面，返回 (掩版 或 None, 说明)。

    为什么不能只用 SAM 的自动分割：实测杯子那张图里，SAM 无论按 IoU 还是按边缘密度，
    给出的都是**整幅图**（覆盖率 63%，bbox 覆盖全画面）——因为灰色杯身与灰色石纹台面
    在颜色与纹理上都很接近，"物体 vs 背景"的语义提示不足以分开它们。

    但物体自己有可用的信号：**闭合且连续的轮廓**（杯口、杯身轮廓、杯把）。
    做法（只用图像自身的边缘）：
        1. 框内做 Canny → 闭运算把断开的边缘接起来；
        2. 取外轮廓并填充 → 得到候选块；
        3. 保留面积落在合理区间的候选（太小是噪声，太大是把背景连成一片）；
        4. 多个候选取面积最大的那个，并只保留与它重叠的轮廓（避免把另一件物体也算进来）。
    判据里没有"杯子""把手"这类题材假设，换任意物体同样成立。
    """
    import numpy as np
    import cv2
    from PIL import Image

    if image is None:
        return None, "没有图像"
    width, height = image.size
    grey = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if box:
        left = max(0, int(box[0] * width))
        top = max(0, int(box[1] * height))
        right = min(width, int(box[2] * width))
        bottom = min(height, int(box[3] * height))
    else:
        left, top, right, bottom = 0, 0, width, height
    if right - left < 16 or bottom - top < 16:
        return None, "框太小"
    roi = grey[top:bottom, left:right]
    edges = cv2.Canny(roi, max(10, int(config.CANNY_LOW * 0.6)),
                      max(30, int(config.CANNY_HIGH * 0.9)))
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = float(roi.shape[0] * roi.shape[1])
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if roi_area * min_ratio <= area <= roi_area * max_ratio:
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((area, contour, (x, y, w, h)))
    if not candidates:
        return None, "框内没有闭合轮廓落在合理面积区间"
    candidates.sort(key=lambda item: -item[0])
    area, contour, bbox = candidates[0]
    filled = np.zeros_like(edges)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = filled
    coverage = float((mask > 0).mean())
    if coverage < config.SUBJECT_MASK_MIN_COVERAGE or coverage > config.OBJECT_SURFACE_MAX_COVERAGE:
        return None, f"轮廓面积占比 {coverage:.1%} 超出可用区间"
    note = (f"闭合轮廓 面积占比 {coverage:.1%} 框内占比 {area / roi_area:.1%} "
            f"外接框 {bbox}")
    print(f"[SAM2] {note}")
    return Image.fromarray(mask, mode="L"), note


def get_subject_mask_or_full(image, sam):
    """主体 mask 获取入口（规格 §9/§27 降级语义）。

    segment_subject 成功 -> 返回分割 mask；
    失败 / SAM 不可用 -> 返回与原图同尺寸的全白整图 mask（255），
    并打印“已降级为整图 mask”（整图 ControlNet 条件，保证流程不断）。
    """
    mask = segment_subject(image, sam)
    if mask is None:
        w, h = image.size
        full = Image.new("L", (w, h), 255)
        print("[SAM2] 已降级为整图 mask（规格 §9/§27：SAM 不可用或分割失败 -> 整图 ControlNet）")
        return full
    return mask


def sam_point_mask(image, sam, point):
    """单个正提示点的 SAM 分割（复用 segment_subject 已验证的前向与后处理路径）。

    用途：像"头发"这种**有固定解剖位置、但视觉模型经常拒答**的区域，可以用"主体顶部
    中心"这样一个确定性提示点来分割，不依赖 VLM 的文字定位。
    """
    model, processor = sam
    try:
        inputs = processor(images=image, input_points=[[[list(point)]]],
                           input_labels=[[[1]]], return_tensors="pt")
        device = next(model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)
        with torch.no_grad():
            embeddings = model.get_image_embeddings(pixel_values)
            output = model._single_frame_forward(
                pixel_values=None, image_embeddings=embeddings,
                input_points=inputs["input_points"].to(device),
                input_labels=inputs["input_labels"].to(device),
                multimask_output=True)
        post = processor.post_process_masks([output.high_res_masks.float()],
                                            inputs["original_sizes"].tolist(),
                                            mask_threshold=0.0, binarize=False)
        masks = post[0]
        # 单点提示时 _single_frame_forward 只返回一张 high_res mask，但 iou_scores
        # 仍是 3 个分数（对应多掩码候选），维度不能按"3 张掩膜"去索引。
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        scores = output.iou_scores.reshape(-1).float()
        best = int(scores.argmax().item()) if (masks.shape[0] > 1 and scores.numel() == masks.shape[0]) else 0
        arr = (masks[best].cpu().numpy() > 0)
        if not arr.any():
            return None
        return Image.fromarray(arr.astype(np.uint8) * 255, mode="L")
    except Exception as exc:
        print(f"[SAM2] 单点分割失败: {type(exc).__name__}: {exc}")
        return None


def sam_point_maskbox(image, sam, points, labels=None, box=None, return_all=False):
    """点提示（可含正/负例）＋可选框的 SAM 分割，返回 255/0 的 "L" 掩版或 None。

    用途：像"头发""衣服"这种**有固定解剖位置、但视觉模型经常定位不准**的区域，
    用确定性提示点（正例点在目标上，负例点在其邻近的干扰物上）配合框来分割，
    不依赖视觉模型的文字定位精度。

    points: [(x, y), ...] 像素坐标；labels: 1=正例, 0=负例（默认全为正例）
    box   : (left, top, right, bottom) 像素坐标，可选；给出后会把搜索限制在框内

    `return_all=True` 时返回**全部候选** `[(mask, iou_score), ...]`，
    由调用方按自己的判据挑一个。为什么需要（实测）：单点提示下
    SAM 的三个候选常常分别是"一小块碎片 / 合适的一块 / 整幅画面"，
    而最高 IoU 分并不总落在"合适"那个上——实测出现覆盖 0.2%（碎片）
    与 100%（整幅）两种极端，用现成的最佳候选挑不出标志本体。
    调用方知道"标志应当占裁剪的多大比例"，因此把选择权交给它更可靠。

    实现与 segment_subject 保持同一路径（get_image_embeddings 只编码一次 +
    单帧前向）；同一帧的多个点必须**一起**送入 input_points，
    逐点单独调用会让单帧前向在 transformers 路径上抛
    "num_objects referenced before assignment"。
    """
    if sam is None:
        return None
    model, processor = sam
    if not points and box is None:
        return None
    labels = list(labels) if labels else [1] * len(points)
    if len(labels) != len(points):
        raise ValueError("points 与 labels 数量必须一致")
    try:
        kwargs = {
            "images": image,
            # 形状必须是 [batch][object][point][2]：同一对象的多个点放在最内层，
            # 少一层就会让单帧前向把点当成多个对象而抛 num_objects 未赋值。
            "input_points": [[[list(p) for p in points]]] if points else None,
            "input_labels": [[labels]] if points else None,
            "return_tensors": "pt",
        }
        if box is not None:
            kwargs["input_boxes"] = [[list(box)]]
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        inputs = processor(**kwargs)
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pixel_values = inputs["pixel_values"].to(device, dtype=torch.float16)
        base = {"multimask_output": True}
        for key in ("input_points", "input_labels", "input_boxes"):
            if key in inputs:
                base[key] = inputs[key].to(device)
        with torch.no_grad():
            embeddings = model.get_image_embeddings(pixel_values)
            output = self_forward(model, embeddings, base)
        orig_sizes = inputs["original_sizes"].tolist()
        post = processor.post_process_masks([output.high_res_masks.float()],
                                            [orig_sizes[0]], mask_threshold=0.0,
                                            binarize=False)
        masks = post[0][0]                      # (3, H, W) 单对象的 3 个候选
        scores = output.iou_scores.reshape(-1).float()
        if return_all:
            candidates = []
            for index in range(masks.shape[0]):
                array = masks[index].cpu().numpy() > 0
                if array.any():
                    candidates.append((array, float(scores[index].item())))
            return candidates or None
        best = int(scores.argmax().item()) if masks.shape[0] > 1 else 0
        array = masks[best].cpu().numpy() > 0
        if not array.any():
            return None
        return Image.fromarray(array.astype(np.uint8) * 255, mode="L")
    except Exception as exc:
        print(f"[SAM2] 点提示分割失败: {type(exc).__name__}: {exc}")
        return None


def self_forward(model, embeddings, payload):
    """单帧前向的兼容包装。

    transformers 4.57 的 _single_frame_forward 在同一对象上**不能同时**接收
    input_points 与 input_boxes（源码里两支互斥，两支都命中时 num_objects 未赋值，
    实测直接抛 UnboundLocalError）。因此这里按"点优先、退化到框"的顺序尝试，
    保证提示信息尽可能保留，而不是整段功能不可用。
    """
    attempts = [dict(payload)]
    if "input_points" in payload and "input_boxes" in payload:
        without_box = {k: v for k, v in payload.items() if k != "input_boxes"}
        attempts.append(without_box)
    last = None
    for attempt in attempts:
        try:
            with torch.no_grad():
                return model._single_frame_forward(
                    pixel_values=None, image_embeddings=embeddings, **attempt)
        except UnboundLocalError as exc:
            last = exc
            continue
    raise last


def unload_sam(sam, processor=None):
    """释放 SAM2 模型与 processor，清空 CUDA 缓存（显存腾挪用）。

    兼容两种调用：unload_sam(sam)（sam 为 load_sam2 返回的元组）
    或 unload_sam(model, processor)。
    """
    if processor is not None:
        model = sam  # 分开传参形式
    elif isinstance(sam, tuple) and len(sam) == 2:
        model, processor = sam
    else:
        model = sam
        processor = None

    for obj in (model, processor):
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[SAM2] 已卸载并清空 CUDA 缓存")


if __name__ == "__main__":
    """自检块（PyCharm 直接运行，不加载任何模型，纯 CPU）。"""
    print("=" * 60)
    print("vision/segmentation.py 自检（不加载模型）")
    print("=" * 60)

    from PIL import Image as _PILImage

    # 1. 降级路径：sam=None -> get_subject_mask_or_full 返回全白整图 mask（§27）
    img = _PILImage.new("RGB", (80, 60), (100, 100, 100))
    full = get_subject_mask_or_full(img, None)
    arr = np.array(full)
    ok1 = (full.size == img.size and full.mode == "L"
           and arr.shape == (60, 80) and arr.min() == 255 and arr.max() == 255)
    print(f"[1] sam=None 降级整图 mask: size={full.size} mode={full.mode} 全白={arr.min()==255}  {'OK' if ok1 else 'FAIL'}")

    # 2. segment_subject(img, None) 直接返回 None
    r = segment_subject(img, None)
    ok2 = (r is None)
    print(f"[2] segment_subject(img, None) -> {r}  {'OK' if ok2 else 'FAIL'}")

    # 3. 模型异常路径：假 sam 对象（_single_frame_forward 抛异常）-> 返回 None
    class _FakeBrokenModel:
        def _single_frame_forward(self, **kwargs):
            raise RuntimeError("模拟模型推理崩溃")

    class _FakeProcessor:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("不该走到预处理")

    fake_sam = (_FakeBrokenModel(), _FakeProcessor())
    r3 = segment_subject(img, fake_sam)
    ok3 = (r3 is None)
    print(f"[3] 模型异常 -> segment_subject 返回 {r3}（不崩溃）  {'OK' if ok3 else 'FAIL'}")

    # 4. unload_sam 兼容两种调用形式（不触发 CUDA 清理逻辑的报错）
    unload_sam(None)
    unload_sam(fake_sam)
    unload_sam(_FakeBrokenModel(), _FakeProcessor())
    ok4 = True
    print(f"[4] unload_sam 三种调用形式均不抛异常  {'OK' if ok4 else 'FAIL'}")

    # 5. 配置联动提示
    print(f"[5] config.USE_SAM={config.USE_SAM} config.SAM_PATH={config.SAM_PATH}")
    print(f"    SAM2 fp16 权重显存预算: 约 {176e6 * 2 / 1024**3:.2f}GB（model.safetensors 约 176M 参数）")

    print("=" * 60)
    print("全部通过" if (ok1 and ok2 and ok3 and ok4) else "存在 FAIL")
