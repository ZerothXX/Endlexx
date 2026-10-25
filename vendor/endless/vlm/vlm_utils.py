# -*- coding: utf-8 -*-
"""vlm/vlm_utils.py —— 本地视觉语言模型封装（唯一实现）

架构无关：使用 AutoProcessor + AutoModelForImageTextToText 自动识别模型类型，
因此同一份代码既能加载 Qwen2.5-VL，也能加载磁盘上实际存在的 Qwen3-VL-4B
（旧实现硬编码 Qwen2_5_VLForConditionalGeneration，遇到 Qwen3-VL 必然失败）。

职责（方案 §2.3）：自动标注、角色特征观察、输入主体理解、辅助规划。
不负责：训练、决定效果是否合格、替代权重。

加载链：4bit 量化 GPU -> fp16 GPU -> CPU(float32)；全部失败则 available=False，
由调用方决定是明确报错还是降级（本项目默认明确报错，不静默换实现）。
"""
import json
import os
import re

import torch

import config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# 提示词（唯一一份定义，其他模块引用这里而不是各写一份）
# =========================
CHARACTER_PROMPT = (
    "You are analysing reference images of ONE character design. Return ONLY JSON with these keys:\n"
    '{"character_identity":"4 to 10 words",'
    '"visual_style":"the rendering medium, e.g. anime illustration / pixel art / photograph",'
    '"hair":{"color":"visible hair colour","style":"visible hairstyle or \\"\\""},'
    '"face":{"eye_color":"visible eye colour","face_features":"brief visible features"},'
    '"clothing":{"description":"garment structure as seen: layers, hem, sleeves, trims"},'
    '"color_palette":["colour 1","colour 2","colour 3"],'
    '"visual_features":[{"name":"short name","description":"what is visible","design_cue":"pattern, line or trim detail",'
    '"region":"head|torso|arms|waist|legs|feet","kind":"pattern|garment|accessory|material","importance":"high|medium|low"}],'
    '"signature_accessories":[{"name":"short name","shape":"visible geometry","color":"visible colours",'
    '"location":"where it sits on the character","transfer_role":"emblem|wearable|prop","semantic_importance":"high|medium|low"}]}\n'
    "Use only what is visible. Use an empty string or empty list when something is not visible. "
    "Do not describe the background, the pose, or anything outside the character design."
)

SUBJECT_PROMPT = (
    "Identify the main subject of this image. Return ONLY JSON:\n"
    '{"subject_type":"human|animal|object|unknown",'
    '"description":"4 to 10 words",'
    '"style":"the rendering medium: photograph, anime, illustration, pixel art, line art, 3d render",'
    '"framing":"full body|upper body|close-up|wide",'
    '"pose":"brief visible pose or orientation",'
    '"visible_regions":["head","torso","arms","legs"],'
    '"held_items":["item"],'
    '"wearing":"brief clothing description"}'
)

ACCESSORY_PROMPT = (
    "Describe the isolated object or graphic in this image using only visible evidence. Return ONLY JSON:\n"
    '{"name":"short object name","shape":"visible geometry","color":"visible colours",'
    '"description":"brief visible design","transfer_role":"emblem|wearable|prop"}'
)

EVAL_PROMPT = (
    "Compare image 1 (source) and image 2 (result). Return ONLY JSON:\n"
    '{"character_features_present":true|false,"structure_preserved":true|false,'
    '"style_preserved":true|false,"issues":["short issue"]}\n'
    "Judge only what is visible. This is evidence for a human reviewer, not a final verdict."
)

CAPTION_PROMPT = (
    "Classify the framing of this image. Return ONLY JSON:\n"
    '{"view":"front view|side view|back view|three-quarter view|unknown",'
    '"framing":"full body shot|upper body|face close-up|clothing close-up|feet close-up|accessory close-up|pattern close-up"}\n'
    "Judge only from the image borders. Do not imagine anything outside the picture."
)

# 标志证据（docs/统一角色标志与自动证据提取_技术调整说明_20261004.md §3、§5）：
# 用户给的是**中文大白话**，系统内部一律使用结构化字段 + 规范化英文短名。
# 结构化与英文化都由这里完成；VLM 不可用时保留原文并记缺口，不臆造。
MARK_STRUCTURE_PROMPT = (
    "You are given one character mark (an emblem, ornament, pattern, material or the "
    "character's colour scheme), plus the user's plain-language note about it.\n"
    "Describe that ONE mark and return ONLY this JSON object, with every field filled in:\n"
    '{"name_zh":"规范中文短名",'
    '"name_en":"short english name",'
    '"entity_mode":"locatable|describable",'
    '"outline":"visible outer silhouette in a few words",'
    '"internal_topology":"how the inside is divided: symmetry, spokes, petals, layers",'
    '"color_relation":"which colours are where, and their relative area",'
    '"repeat_unit":"the repeating unit if it is a continuous pattern, else \\"\\"",'
    '"material":"visible material or finish",'
    '"relative_scale":"how large this mark is relative to the part that carries it",'
    '"mark_class":"palette|emblem|pattern|trim|wearable|material|detail",'
    '"transfer_role":"emblem|wearable|prop|pattern|palette",'
    '"description_en":"one short english sentence usable as a training caption, '
    'colours first, geometry second"}\n'
    "Rules:\n"
    "- Fill in every field from what the user's note states plus what this image actually "
    "shows. Only use an empty string when the image really gives you nothing for that "
    "field; an all-empty answer is never acceptable.\n"
    "- The user's note is the primary evidence: it is a human observation of this exact "
    "mark. Translate its meaning into the fields even when the crop is small.\n"
    "- entity_mode is \"locatable\" when this mark has a real boundary you could draw a box "
    "around, and \"describable\" when it is a surface property with no single boundary "
    "(an overall colour scheme, hair colour, hairstyle, eye colour).\n"
    "- The image may contain much more than this mark. Describe ONLY the mark named by the "
    "user's note; if the note does not name it clearly, describe only the most prominent "
    "mark in the image.\n"
    "- Never invent details that neither the note nor the image supports."
)

MARK_QUERY_PROMPT = (
    "Look at this image carefully. Decide whether the following character mark is "
    "actually visible in it: {target}\n"
    "Return ONLY JSON:\n"
    '{{"bbox_2d":[left, top, right, bottom] or null,'
    '"visibility":"visible|partial|occluded|out_of_view|absent|unknown",'
    '"visible_description":"what is actually visible of it here, or empty"}}\n'
    "bbox_2d must be absolute pixel coordinates on this {width} by {height} image and must "
    "tightly bound the mark itself. Use \"visible\" only when the whole mark is clearly "
    "shown, \"partial\" when only part of it is in frame or covered, \"occluded\" when "
    "something blocks it, \"out_of_view\" when the carrying body part or object side is "
    "not in this picture, \"absent\" when the surface where it belongs is visible but the "
    "mark is not there, and \"unknown\" when you cannot tell. "
    "Never guess a box for a mark you cannot see. "
    "Write visible_description in ENGLISH only, even when the mark description above is "
    "in Chinese, and keep it under 25 words."
)

# 无边界的外观属性（配色、发色、瞳色、发型剪裁）：能看出"在不在这张图里"、
# 也能描述"这里看到的是什么颜色/什么形状"，但**没有唯一的位置框**。
# 逼模型给出框只会得到"整幅上半部"这类退化框，进而污染裁剪样本（实测踩过）。
MARK_TRAIT_PROMPT = (
    "Look at this image carefully. The following character appearance property may be "
    "visible in it: {target}\n"
    "Return ONLY JSON:\n"
    '{{"visibility":"visible|partial|occluded|out_of_view|absent|unknown",'
    '"visible_description":"what this property actually looks like in THIS image, '
    'colours first"}}\n'
    "Use \"visible\" only when the image clearly shows this property, \"partial\" when it "
    "is partly shown or partly covered, \"occluded\" when something blocks it, "
    "\"out_of_view\" when the carrying body part is not in this picture, \"absent\" when "
    "the body part is visible but the property is not as described, and \"unknown\" when "
    "you cannot tell. Do not return any box. "
    "Write visible_description in ENGLISH only, even when the property above is described "
    "in Chinese, and keep it under 25 words."
)

FRAMINGS = ("full body shot", "upper body", "face close-up", "clothing close-up",
            "feet close-up", "accessory close-up", "pattern close-up")
VIEWS = ("front view", "side view", "back view", "three-quarter view")
MARK_VISIBILITIES = ("visible", "partial", "occluded", "out_of_view", "absent", "unknown")


# =========================
# 输出解析
# =========================
def parse_json(text) -> dict:
    """从模型输出中提取第一个 JSON 对象；失败返回 {}（不猜内容）。"""
    if not isinstance(text, str) or not text.strip():
        return {}
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        return {}
    depth, in_string, escape = 0, False, False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(candidate[start:index + 1])
                except ValueError:
                    return {}
                return value if isinstance(value, dict) else {}
    return {}


# =========================
# 客户端
# =========================
class VLMClient:
    """本地 VLM 封装（只推理，不参与训练）。"""

    def __init__(self, model_path: str = None, dtype: str = None,
                 load_in_4bit: bool = None, offload_cpu: bool = None):
        self.model_path = model_path or config.VLM_PATH
        self.model = None
        self.processor = None
        self.device = None
        self.available = False
        self.load_error = None
        self.load_mode = None
        dtype = dtype or config.VLM_DTYPE
        self._dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype, torch.float32)
        load_in_4bit = config.VLM_LOAD_IN_4BIT if load_in_4bit is None else load_in_4bit
        offload_cpu = config.VLM_OFFLOAD_CPU if offload_cpu is None else offload_cpu
        self._load(load_in_4bit=bool(load_in_4bit), offload_cpu=bool(offload_cpu))

    # ---------- 加载 ----------
    def _load(self, load_in_4bit: bool, offload_cpu: bool) -> None:
        if not os.path.isdir(self.model_path):
            self.load_error = f"模型目录不存在: {self.model_path}"
            print(f"[VLM] 告警: {self.load_error}")
            return
        if not self._load_processor():
            return
        attempts = []
        if offload_cpu or not torch.cuda.is_available():
            attempts = [("cpu", self._load_cpu)]
        elif load_in_4bit:
            attempts = [("4bit", self._load_4bit), ("cpu", self._load_cpu)]
        else:
            attempts = [("fp16", self._load_fp16), ("4bit", self._load_4bit),
                        ("cpu", self._load_cpu)]
        for label, loader in attempts:
            if loader():
                self.load_mode = label
                return
        print(f"[VLM] 告警: 全部加载方式失败，VLM 不可用。最后一次错误: {self.load_error}")

    def _load_processor(self) -> bool:
        try:
            from transformers import AutoProcessor
            use_fast = not _torch_below((2, 3))   # 旧 torch 不支持 fast image processor
            self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=use_fast)
            return True
        except Exception as exc:
            self.load_error = f"processor 加载失败: {type(exc).__name__}: {exc}"
            print(f"[VLM] 降级: {self.load_error}")
            return False

    def _from_pretrained(self, **kwargs):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(self.model_path, **kwargs)

    def _finish(self, model, device, label):
        model.eval()
        self.model, self.device, self.available = model, device, True
        print(f"[VLM] 已加载（{label}）: {self.model_path} "
              f"架构={type(model).__name__}", flush=True)
        return True

    def _load_fp16(self) -> bool:
        try:
            model = self._from_pretrained(dtype=self._dtype, device_map="cuda:0")
            return self._finish(model, torch.device("cuda:0"), "fp16/bf16 GPU")
        except Exception as exc:
            self._cleanup()
            self.load_error = f"GPU 半精度加载失败: {type(exc).__name__}: {exc}"
            print(f"[VLM] 降级: {self.load_error}")
            return False

    def _load_4bit(self) -> bool:
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16)
            model = self._from_pretrained(quantization_config=quant, device_map="cuda:0")
            return self._finish(model, torch.device("cuda:0"), "4bit GPU")
        except Exception as exc:
            self._cleanup()
            self.load_error = f"4bit 加载失败: {type(exc).__name__}: {exc}"
            print(f"[VLM] 降级: {self.load_error}")
            return False

    def _load_cpu(self) -> bool:
        try:
            model = self._from_pretrained(dtype=torch.float32)
            return self._finish(model, torch.device("cpu"), "CPU/fp32（很慢）")
        except Exception as exc:
            self._cleanup()
            self.load_error = f"CPU 加载失败: {type(exc).__name__}: {exc}"
            print(f"[VLM] 降级: {self.load_error}")
            return False

    def _cleanup(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- 推理 ----------
    def _generate_text(self, image, prompt: str, max_new_tokens=None,
                       temperature=None) -> str:
        if not self.available:
            print("[VLM] 告警: VLM 不可用，无法推理")
            return ""
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert("RGB")
        max_new_tokens = int(config.VLM_MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens)
        temperature = float(config.VLM_TEMPERATURE if temperature is None else temperature)
        try:
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
            messages = [{"role": "user", "content": content}]
            try:
                # 部分模型（如 Qwen3-VL）默认开启思考模式，会输出额外推理文本；
                # 这里只需要结构化答案，因此显式关闭；不支持的处理器会忽略该参数。
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            new_tokens = generated[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        except Exception as exc:
            print(f"[VLM] 告警: 推理失败，返回空文本: {type(exc).__name__}: {exc}")
            return ""

    def analyze_image(self, image, prompt: str, max_new_tokens=None,
                      temperature=None) -> dict:
        """按提示词分析单图，返回结构化 dict；解析失败返回 {}。"""
        raw = self._generate_text(image, prompt, max_new_tokens, temperature)
        if not raw:
            return {}
        result = parse_json(raw)
        if not result:
            print(f"[VLM] 告警: 输出无法解析为 JSON: {raw[:150]!r}")
        return result

    def locate(self, image, target: str, max_new_tokens: int = 160):
        """定位图中某个可见区域，返回归一化框 (left, top, right, bottom) 或 None。

        坐标先在实际送入模型的画布上取得，再除以该画布尺寸归一化，
        因此输入长宽比与 patch 对齐方式不会造成坐标错位。
        区域不可见 / 输出不合法时返回 None（调用方应跳过局部处理而不是猜一个框）。
        """
        from PIL import Image
        if not self.available:
            return None
        try:
            processor = self.processor.image_processor
            factor = int(getattr(processor, "patch_size", 14)) * int(
                getattr(processor, "merge_size", 2))
            width = max(factor, round(image.width / factor) * factor)
            height = max(factor, round(image.height / factor) * factor)
            canvas = image.resize((width, height), Image.Resampling.BICUBIC)
            prompt = (
                f'Locate {target}. Return ONLY JSON {{"bbox_2d": [left, top, right, bottom]}} '
                f'in absolute pixel coordinates on this {width} by {height} image. '
                'Tightly bound only the named visible region. '
                'Return {"bbox_2d": null} if it is not visible.')
            raw = self._generate_text(canvas, prompt, max_new_tokens=max_new_tokens,
                                      temperature=0)
            value = parse_json(raw).get("bbox_2d")
            print(f"[VLM] 定位 {target!r}: {value}", flush=True)
            if not isinstance(value, (list, tuple)) or len(value) != 4:
                return None
            numbers = [float(v) for v in value]
        except Exception as exc:
            print(f"[VLM] 定位失败（{target}）: {type(exc).__name__}: {exc}")
            return None
        left, top, right, bottom = (numbers[0] / width, numbers[1] / height,
                                   numbers[2] / width, numbers[3] / height)
        left, right = sorted((max(0.0, min(1.0, left)), max(0.0, min(1.0, right))))
        top, bottom = sorted((max(0.0, min(1.0, top)), max(0.0, min(1.0, bottom))))
        if right - left < 0.02 or bottom - top < 0.02:
            return None
        return (left, top, right, bottom)

    def locate_mark(self, image, target: str, region_hint: str = "") -> dict:
        """在图中定位一个**角色标志**并给出逐图可见状态（规格 §5 自动识别流程）。

        与 `locate()` 的区别：`locate()` 问的是"人体/物体的某个部位在哪"，
        问法固定、答案稳定；本方法问的是"这个具体标志在不在这张图里、在哪"，
        需要同时得到框与可见状态，并且**允许回答"看不到"**——拿不到框不是失败，
        而是一条有意义的事实（该图不提供这个标志的外观证据）。

        送入模型的画布会先锐化（规格 §5.3：放大只帮助阅读已有像素，
        不声称恢复本来不存在的纹样）：小标志在 768 短边的图上只有几十像素，
        不锐化时模型看不出内部结构，会给出"大概在这一带"的宽框。

        返回 {"box": (l,t,r,b) 归一化, "visibility": str, "description": str,
              "canvas": (w,h)}；无法判定时返回 {}（调用方记缺口，不猜）。
        """
        from PIL import Image, ImageFilter
        if not self.available:
            return {}
        try:
            processor = self.processor.image_processor
            factor = int(getattr(processor, "patch_size", 14)) * int(
                getattr(processor, "merge_size", 2))
            width = max(factor, round(image.width / factor) * factor)
            height = max(factor, round(image.height / factor) * factor)
            canvas = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            if config.MARK_LOCATE_UPSCALE > 1:
                canvas = canvas.resize((int(width * config.MARK_LOCATE_UPSCALE),
                                        int(height * config.MARK_LOCATE_UPSCALE)),
                                       Image.Resampling.LANCZOS)
                width, height = canvas.size
            if config.MARK_LOCATE_SHARPEN > 0:
                canvas = canvas.filter(ImageFilter.UnsharpMask(
                    radius=2, percent=int(config.MARK_LOCATE_SHARPEN * 100), threshold=2))
            hint = f" Its usual place on the character is: {region_hint}." if region_hint else ""
            prompt = MARK_QUERY_PROMPT.format(target=target, width=width, height=height) + hint
            raw = self._generate_text(canvas, prompt, max_new_tokens=256, temperature=0)
            result = parse_json(raw)
            if not result:
                print(f"[VLM] 标志定位输出无法解析: {target!r} -> {raw[:120]!r}")
                return {"visibility": "unknown", "description": "",
                        "canvas": (width, height), "reason": "输出无法解析"}
            visibility = str(result.get("visibility") or "unknown").strip().lower()
            if visibility not in MARK_VISIBILITIES:
                visibility = "unknown"
            description = str(result.get("visible_description") or "").strip()
            box = result.get("bbox_2d")
            if visibility in ("absent", "out_of_view", "occluded", "unknown"):
                # 模型明确说看不到：框一律丢弃（只保留可见状态与描述）。
                print(f"[VLM] 标志 {target!r}: {visibility}", flush=True)
                return {"box": None, "visibility": visibility,
                        "description": description, "canvas": (width, height)}
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                print(f"[VLM] 标志 {target!r}: 声称 {visibility} 但没有可用框", flush=True)
                return {"box": None, "visibility": "unknown",
                        "description": description, "canvas": (width, height),
                        "reason": "有可见声明但缺少框"}
            numbers = [float(v) for v in box]
        except Exception as exc:
            print(f"[VLM] 标志定位失败（{target}）: {type(exc).__name__}: {exc}")
            return {}
        left, top, right, bottom = (numbers[0] / width, numbers[1] / height,
                                    numbers[2] / width, numbers[3] / height)
        left, right = sorted((max(0.0, min(1.0, left)), max(0.0, min(1.0, right))))
        top, bottom = sorted((max(0.0, min(1.0, top)), max(0.0, min(1.0, bottom))))
        if right - left < config.MARK_LOCATE_MIN_SIDE or \
                bottom - top < config.MARK_LOCATE_MIN_SIDE:
            print(f"[VLM] 标志 {target!r}: 框过小已丢弃 "
                  f"({right - left:.3f}×{bottom - top:.3f})", flush=True)
            return {"box": None, "visibility": "unknown",
                    "description": description, "canvas": (width, height),
                    "reason": "框小于最小边长"}
        print(f"[VLM] 标志 {target!r}: {visibility} "
              f"{tuple(round(v, 3) for v in (left, top, right, bottom))}", flush=True)
        return {"box": (left, top, right, bottom), "visibility": visibility,
                "description": description, "canvas": (width, height)}

    def generate_with_images(self, images, prompt: str, max_new_tokens=None,
                             temperature=None) -> str:
        """多图提问（Qwen-VL 的 content 支持多张 image 块）。

        用途：标志识别的**留出对照**——给它一张参考裁剪 + 一张整图，
        问"这个标志在不在这张整图里"。单图接口做不到这件事。
        """
        if not self.available:
            print("[VLM] 告警: VLM 不可用，无法推理")
            return ""
        images = [image for image in (images or []) if image is not None]
        if not images:
            return ""
        max_new_tokens = int(config.VLM_MAX_NEW_TOKENS if max_new_tokens is None
                             else max_new_tokens)
        temperature = float(config.VLM_TEMPERATURE if temperature is None else temperature)
        try:
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=images, return_tensors="pt")
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs, do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    max_new_tokens=max_new_tokens, use_cache=True)
            new_tokens = generated[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        except Exception as exc:
            print(f"[VLM] 告警: 多图推理失败，返回空文本: {type(exc).__name__}: {exc}")
            return ""

    def observe_trait(self, image, target: str, region_hint: str = "") -> dict:
        """查询一个**无边界的外观属性**在某图里的可见状态与可见描述。

        规格 §4：逐图状态至少包括 visible / partial / occluded / out_of_view /
        absent / unknown；`absent` 与 `occluded` 的区别是"该在的地方看得到但没有"
        还是"被挡住了"。这两者都不是负样本，只是"这里没有可学的正向像素"。
        """
        from PIL import Image
        if not self.available:
            return {}
        try:
            processor = self.processor.image_processor
            factor = int(getattr(processor, "patch_size", 14)) * int(
                getattr(processor, "merge_size", 2))
            width = max(factor, round(image.width / factor) * factor)
            height = max(factor, round(image.height / factor) * factor)
            canvas = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            hint = f" It belongs to: {region_hint}." if region_hint else ""
            prompt = MARK_TRAIT_PROMPT.format(target=target) + hint
            raw = self._generate_text(canvas, prompt, max_new_tokens=160, temperature=0)
            result = parse_json(raw)
        except Exception as exc:
            print(f"[VLM] 属性观察失败（{target}）: {type(exc).__name__}: {exc}")
            return {}
        if not result:
            return {}
        visibility = str(result.get("visibility") or "unknown").strip().lower()
        if visibility not in MARK_VISIBILITIES:
            visibility = "unknown"
        return {"box": None, "visibility": visibility,
                "description": str(result.get("visible_description") or "").strip()}

    def generate_tags(self, image_path: str, system_prompt: str = None) -> list:
        """返回 2~4 个视角/构图英文短标签，供 captions.txt 使用。

        约定：返回 list[str]，无法判定时返回 []（调用方不得据此编造标注）。
        """
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((640, 640))
        prompt = CAPTION_PROMPT
        if system_prompt and system_prompt != CAPTION_PROMPT:
            prompt = prompt + "\nAdditional context: " + system_prompt
        raw = self._generate_text(image, prompt, max_new_tokens=96, temperature=0)
        if not raw:
            return []
        result = parse_json(raw)
        framing = re.sub(r"\s*\([^)]*\)\s*$", "", str(result.get("framing", "")).lower().strip())
        view = str(result.get("view", "")).lower().strip()
        if view in {"face close-up", "feet close-up"} and framing in {"clothing close-up", "accessory close-up"}:
            framing, view = view, "unknown"
        if framing not in FRAMINGS:
            print(f"[VLM] 构图分类无效，丢弃该图标注: {raw[:200]!r}")
            return []
        if framing in {"accessory close-up", "pattern close-up"}:
            evidence = self.analyze_image(
                image, 'Is an actual human face visible in this image? Return ONLY JSON '
                       '{"face_visible":true} or {"face_visible":false}. Ignore designs on objects.',
                temperature=0, max_new_tokens=48)
            if type(evidence.get("face_visible")) is not bool:
                return []          # 无法核实就不把这张图绑到"配饰"类别上
            if evidence["face_visible"]:
                framing = "face close-up"
        return ([view] if view in VIEWS else []) + [framing]

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self.available = False
        self.device = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[VLM] 已卸载，显存已释放")


def _torch_below(version) -> bool:
    try:
        parts = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    except ValueError:
        return False
    return parts < version


def get_default_memory(folder: str) -> dict:
    """VLM 不可用时的最小结构：字段齐全但为空，绝不编造特征。"""
    return {"folder": folder, "identity": "", "appearance": {}, "color_palette": {},
            "clothing": {}, "accessories": [], "signature_accessories": [],
            "visual_features": [], "visual_style": "", "source": "offline",
            "note": "VLM 不可用，未生成任何观察；请人工补充 config.CHARACTER_PROFILES"}
