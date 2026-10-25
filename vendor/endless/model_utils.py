# -*- coding: utf-8 -*-
"""model_utils.py —— 单一底模的全部模型操作（唯一实现）

本项目只使用一份 SDXL 底模：角色身份与视觉概念的学习、以及正式推理都在它上面进行。
因此这里不提供"按主体类型切换底模"的入口；需要保留源图画风时，靠结构条件、
重绘强度与主体蒙版控制，而不是换底模（换底模会让已训练权重失效）。

对外接口：
    load_components     加载底模 / VAE / ControlNet（按需），只装一次
    build_text2img      文生图管线
    build_img2img       图生图管线（可挂 ControlNet：结构条件保留源图形态）
    verify_model_paths  启动前路径检查
    build_train_pipeline 训练用管线（冻结 VAE 与文本编码器）
    prepare_lora        在 UNet 上注入 LoRA
    save_lora           以 diffusers 格式保存 LoRA 单文件
    load_adapter        把训练所得权重挂到推理管线上（含强度设置）
    unload_adapters     卸载全部适配器（角色切换时防止残留）
    adapter_manifest    权重文件的自描述元数据
"""
import json
import os

import torch

import config

ADAPTERS = ("default",)   # 当前只有一个角色分支；多分支时这里列出全部适配器名


# ---------------------------------------------------------------------------
# 组件加载
# ---------------------------------------------------------------------------
def verify_model_paths() -> dict:
    """检查模型路径存在性，返回 {名称: {exists, path, required}}。"""
    entries = {
        "BASE_MODEL_PATH": (config.BASE_MODEL_PATH, "file", True),
        "BASE_VAE_PATH": (config.BASE_VAE_PATH, "file", True),
        "CONTROLNET_PATH": (config.CONTROLNET_PATH, "dir", bool(config.USE_CONTROLNET)),
        "SAM_PATH": (config.SAM_PATH, "dir", bool(config.USE_SAM)),
        "VLM_PATH": (config.VLM_PATH, "dir", bool(config.USE_VLM)),
    }
    status = {}
    for name, (path, kind, required) in entries.items():
        exists = os.path.isfile(path) if kind == "file" else os.path.isdir(path)
        status[name] = {"exists": exists, "path": path, "required": required}
    return status


def missing_required(status: dict) -> list:
    return [name for name, info in status.items() if info["required"] and not info["exists"]]


def load_components(device: str = "cuda", dtype=torch.float16):
    """加载底模与 VAE（训练与推理共用这一份实现），返回 dict。

    使用 from_single_file 加载单文件 checkpoint；文本编码器与 VAE 的配置从本地
    HuggingFace 缓存解析，因此需要 config.HF_OFFLINE 或可用的镜像端点。
    """
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    if not os.path.isfile(config.BASE_MODEL_PATH):
        raise FileNotFoundError(f"底模文件不存在: {config.BASE_MODEL_PATH}")
    if not os.path.isfile(config.BASE_VAE_PATH):
        raise FileNotFoundError(f"VAE 文件不存在: {config.BASE_VAE_PATH}")
    if config.HF_OFFLINE:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)

    print(f"[model] 加载底模: {config.BASE_MODEL_PATH}（dtype={dtype}）", flush=True)
    pipe = StableDiffusionXLPipeline.from_single_file(config.BASE_MODEL_PATH, torch_dtype=dtype)
    print(f"[model] 加载 VAE: {config.BASE_VAE_PATH}", flush=True)
    pipe.vae = AutoencoderKL.from_single_file(config.BASE_VAE_PATH, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    return {"pipe": pipe, "device": device, "dtype": dtype}


def load_controlnet(dtype=torch.float16):
    """加载结构条件模型；缺失或失败时抛错，由调用方决定是否降级（不静默换实现）。"""
    from diffusers import ControlNetModel

    if not os.path.isdir(config.CONTROLNET_PATH):
        raise FileNotFoundError(f"ControlNet 目录不存在: {config.CONTROLNET_PATH}")
    controlnet = ControlNetModel.from_pretrained(
        config.CONTROLNET_PATH, torch_dtype=dtype, variant="fp16")
    print(f"[model] 已加载 ControlNet(canny, fp16): {config.CONTROLNET_PATH}", flush=True)
    return controlnet


# ---------------------------------------------------------------------------
# 管线装配
# ---------------------------------------------------------------------------
def build_text2img(components, with_controlnet: bool = False):
    """装配**真正的**文生图管线（例如"让模型画一张平面纹样"）。

    ## 必须显式新建管线，不能复用传入的那个（实测踩过，值得记）

    早期实现在 `with_controlnet=False` 时直接 `return pipe`——它假设
    `components["pipe"]` 是原始的 `StableDiffusionXLPipeline`。
    但调用方常常先把管线换成了 **img2img**：
      * `test.py` 的主链路用 `build_img2img(..., with_controlnet=True)`，
        拿到的是 `StableDiffusionXLControlNetImg2ImgPipeline`；
      * 对它调用文生图会直接抛
        `TypeError: image must be passed and be one of PIL image, ...`
        （img2img 管线没有 `image` 就不能工作）。
    这个错误在单独探测时**看不出来**：`build_train_pipeline` 返回的是
    纯文生图管线，所以同一条代码在那里能跑通（实测对比过）。
    因此这里改为**始终新建**：从 `components["pipe"].components` 装配
    `StableDiffusionXLPipeline`，与 `build_img2img` 的做法保持一致。
    """
    from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline

    source = components["pipe"]
    # **必须把 `controlnet` 从 components 里剔除**：调用方的管线可能是
    # ControlNet 变体，它的 `.components` 会带上 `controlnet` 键，
    # 而 `StableDiffusionXLPipeline.__init__` 不接受这个参数——
    # 实测直接抛 `TypeError: __init__() got an unexpected keyword argument 'controlnet'`。
    base_components = {key: value for key, value in source.components.items()
                       if key != "controlnet"}
    base = StableDiffusionXLPipeline(**base_components)
    if not with_controlnet:
        base._generation_mode = "txt2img"
        _place(base, components)
        return base, False
    controlnet = load_controlnet(dtype=components["dtype"])
    new_pipe = StableDiffusionXLControlNetPipeline(
        **base_components, controlnet=controlnet)
    new_pipe._generation_mode = "txt2img+controlnet"
    _place(new_pipe, components)
    return new_pipe, True


def build_img2img(components, with_controlnet: bool = None):
    """装配图生图管线，返回 (pipe, use_controlnet)。

    带 ControlNet 时为 StableDiffusionXLControlNetImg2ImgPipeline：源图作为
    img2img 初值锚定姿态与构图，ControlNet 以较低权重保持轮廓清晰。
    """
    from diffusers import (
        StableDiffusionXLControlNetImg2ImgPipeline,
        StableDiffusionXLImg2ImgPipeline,
    )

    if with_controlnet is None:
        with_controlnet = bool(config.USE_CONTROLNET)
    pipe = components["pipe"]
    if with_controlnet:
        controlnet = load_controlnet(dtype=components["dtype"])
        new_pipe = StableDiffusionXLControlNetImg2ImgPipeline(
            **pipe.components, controlnet=controlnet)
        new_pipe._generation_mode = "img2img+controlnet"
        _place(new_pipe, components)
        return new_pipe, True
    new_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    new_pipe._generation_mode = "img2img"
    _place(new_pipe, components)
    return new_pipe, False


def _place(pipe, components):
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    if getattr(config, "ATTENTION_SLICING", False):
        try:
            pipe.enable_attention_slicing()
        except Exception as exc:
            print(f"[model] 注意力分片不可用，跳过: {type(exc).__name__}")
    pipe.to(components["device"])
    pipe.set_progress_bar_config(disable=True)


def build_train_pipeline(device: str = "cuda", dtype=torch.float16):
    """训练用管线：底模 + VAE 放到设备上，文本编码器留给训练循环编码后移到 CPU。

    训练阶段不加载 ControlNet / SAM / VLM（显存必须全部留给反向传播）。
    """
    components = load_components(device=device, dtype=dtype)
    pipe = components["pipe"]
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.to(device)
    for name in ("unet", "vae", "text_encoder", "text_encoder_2"):
        module = getattr(pipe, name)
        count = sum(p.numel() for p in module.parameters())
        print(f"[model]   {name:14s} dtype={str(module.dtype):9s} 参数 {count:,}（{count / 1e6:.1f}M）")
    return pipe


# ---------------------------------------------------------------------------
# LoRA 注入与保存
# ---------------------------------------------------------------------------
def resolve_lora_targets(unet) -> list:
    """LoRA 注入的目标模块名；config 显式配置优先，否则扫描 UNet 注意力投影层。"""
    explicit = getattr(config, "LORA_TARGET_MODULES", None)
    if explicit:
        return list(explicit)
    patterns = ["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "out_proj"]
    found = {pattern for name, module in unet.named_modules()
             if isinstance(module, torch.nn.Linear)
             for pattern in patterns if name.endswith(pattern)}
    if not found:
        raise RuntimeError("UNet 中未找到注意力投影层，无法确定 LoRA 目标模块")
    return sorted(found)


def prepare_lora(pipe, rank: int = None, alpha: int = None, target_modules=None):
    """在 UNet 上注入 LoRA：底模全部冻结，只有 LoRA 参数参与反向传播。"""
    from peft import LoraConfig, get_peft_model

    rank = int(config.LORA_RANK if rank is None else rank)
    alpha = int(config.LORA_ALPHA if alpha is None else alpha)
    unet = pipe.unet
    unet.requires_grad_(False)
    for module in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        module.requires_grad_(False)
    targets = list(target_modules) if target_modules else resolve_lora_targets(unet)
    lora_config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets,
                             lora_dropout=0.0, bias="none")
    unet_lora = get_peft_model(unet, lora_config)
    total = sum(p.numel() for p in unet_lora.parameters())
    trainable = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
    print(f"[lora] 注入完成 rank={rank} alpha={alpha} targets={targets}")
    print(f"[lora] 可训练参数 {trainable:,}（{trainable / 1e6:.2f}M）"
          f"，占 UNet 的 {trainable / max(1, total) * 100:.3f}%")
    return unet_lora


def _relative_or_self(path, start):
    """尽量给出相对路径；跨盘符时退回绝对路径（不让记录因路径分隔失败）。"""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return path


def adapter_manifest(role, entry, rank, alpha, target_modules, steps, dataset_id=None,
                     base_model=None, notes=None) -> dict:
    """角色包 manifest 的单个权重条目（方案 §6：权重必须自带用途与适用底模）。"""
    import hashlib
    digest = hashlib.sha256()
    with open(entry, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "role": role,
        "file": _relative_or_self(entry, config.CHARACTERS_DIR).replace("\\", "/"),
        "bytes": os.path.getsize(entry),
        "sha256": digest.hexdigest(),
        "format": "diffusers-unet-lora-safetensors",
        "base_model": base_model or _relative_or_self(config.BASE_MODEL_PATH,
                                                      config.PROJECT_ROOT).replace("\\", "/"),
        "rank": int(rank),
        "alpha": int(alpha),
        "target_modules": list(target_modules),
        "trained_steps": int(steps),
        "dataset_id": dataset_id,
        "notes": notes or "",
    }


def save_lora(pipe, unet_lora, save_path: str) -> str:
    """保存为可被 load_lora_weights 直接加载的单文件 safetensors。

    peft 的 state_dict 键带 "base_model.model." 前缀与 adapter 名，而 diffusers
    加载端期望无前缀的键；这里用 get_peft_model_state_dict 生成正确键格式。
    """
    from peft import get_peft_model_state_dict

    layers = get_peft_model_state_dict(
        unet_lora, state_dict=unet_lora.get_base_model().state_dict())
    layers = {k: v.detach().to("cpu") for k, v in layers.items()}
    out_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(out_dir, exist_ok=True)
    pipe.save_lora_weights(out_dir, unet_lora_layers=layers,
                           weight_name=os.path.basename(save_path))
    print(f"[lora] 权重已保存: {save_path}（{len(layers)} 个张量）", flush=True)
    return save_path


# ---------------------------------------------------------------------------
# 推理侧适配器管理
# ---------------------------------------------------------------------------
def load_adapter(pipe, path, scale=None, adapter_name="default"):
    """把训练所得权重挂到推理管线上；文件缺失即报错，不允许静默用纯底模。

    返回审计 dict，供生成记录区分"声明加载了什么"与"实际加载了什么"。

    注意：新加载一个适配器时，diffusers 会把**之前激活的适配器停用**（实测：
    加载 concept 之后 active 只剩 ['concept']）。因此这里加载完成后立刻把
    登记表里的全部适配器连同各自强度一起重新提交，保证多适配器同时生效。
    """
    path = os.path.abspath(str(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"角色权重不存在: {path}\n"
            "请先运行 python train.py 训练该角色；本项目不会静默改用纯底模或参考编辑。")
    weight = float(config.LORA_SCALE if scale is None else scale)
    # 判断"是否已经加载过"必须问管线本身，不能问本模块的登记表：
    # 登记表是模块级全局，set_adapter_scale() 在没有适配器时也会写进去，
    # 于是"登记表里有名字"≠"管线上真的挂着这个适配器"。早期版本据此跳过加载，
    # 结果是**权重从未被加载**，而日志显示"已存在，仅更新强度"（实测踩过）。
    if adapter_name in _loaded_adapter_names(pipe):
        print(f"[model] 适配器 {adapter_name} 已存在，仅更新强度为 {weight}")
        _ADAPTER_SCALES[adapter_name] = weight
        set_adapter_scale(pipe, weight, adapter_name)
        return {"loaded": True, "adapter_name": adapter_name, "path": path, "scale": weight}
    pipe.load_lora_weights(os.path.dirname(path), weight_name=os.path.basename(path),
                           adapter_name=adapter_name, local_files_only=True)
    _ADAPTER_SCALES[adapter_name] = weight
    _apply_adapter_scales(pipe)
    print(f"[model] 已加载角色权重: {os.path.basename(path)}（adapter={adapter_name}, "
          f"scale={weight}）", flush=True)
    return {"loaded": True, "adapter_name": adapter_name, "path": path, "scale": weight}


# 已加载适配器的强度登记表。必须自己记：diffusers 没有稳定的公开接口能读回单个适配器
# 的当前强度（实测 pipe._adapter_weights / get_active_adapters 都拿不到数值），
# 而多适配器同时生效又要求每次提交全部强度。
_ADAPTER_SCALES = {}


def _active_adapter_scales(pipe):
    """当前管线上已加载的适配器及其强度（以登记表为准，缺失时按激活列表补齐）。"""
    scales = {name: _ADAPTER_SCALES.get(name, float(config.LORA_SCALE))
              for name in (pipe.get_active_adapters() or [])}
    for name, value in _ADAPTER_SCALES.items():
        scales.setdefault(name, value)
    return scales


def _loaded_adapter_names(pipe) -> set:
    """**管线本身**记录着的适配器名（text_encoder / unet 的 peft_config）。

    与 `_active_adapter_scales` 的区别很关键：后者读的是本模块的登记表，
    里面有名字不代表管线上真的加载过权重。判断"要不要加载"只能用这个函数，
    否则会在登记表被写过之后错误地跳过真正的加载（实测踩过）。
    """
    names = set()
    for component in ("unet", "text_encoder", "text_encoder_2", "transformer"):
        module = getattr(pipe, component, None)
        config_names = getattr(module, "peft_config", None)
        if config_names:
            names.update(str(k) for k in config_names)
    try:
        for value in (pipe.get_list_adapters() or {}).values():
            names.update(str(v) for v in value or [])
    except Exception:      # 管线不支持该接口时不影响主判据
        pass
    return names


def _apply_adapter_scales(pipe):
    """把登记表里的全部适配器与强度一次性提交（缺一个都会让那个适配器被停用）。

    实测踩过的坑（本项目最贵的一个 bug）：早期这里写的是
        pipe.set_adapters(names, adapter_weights=[...])
    Pipeline 的 `set_adapters` 参数名是 `adapter_weights`，但它把权重转发给
    `unet.set_adapters(adapter_names, weights=...)` 时用的是**位置参数还是关键字**取决于
    版本；实测在 diffusers 0.31 + peft 0.17 下权限不足时**不会报错**，
    于是强度从未真正生效——所有"强度对照实验"其实都在同一个强度下跑的，
    这就是多轮"改了没效果"的根源之一（另一个是陈旧的 __pycache__）。

    现在改为只走 diffusers 明确支持的写法，并在提交后**回读** peft 层的 scaling
    做校验：拿不到就打印警告，绝不假装生效。
    """
    scales = _active_adapter_scales(pipe)
    if not scales:
        return
    names = list(scales)
    weights = [float(scales[name]) for name in names]
    # 直接对 UNet 提交：diffusers 的 pipeline 层会把权重转发给 unet，
    # 但转发路径在不同版本里对参数名/顺序的处理不一致（实测会出现"不报错也不生效"）。
    # 这里只走被 peft 明确定义的那一层（UNet.set_adapters(adapter_names, weights)），
    # 提交后回读 peft 层的 scaling 校验，确保强度真的落下去了。
    unet = getattr(pipe, "unet", pipe)
    unet.set_adapters(names, weights=weights)
    for component in ("text_encoder", "text_encoder_2"):
        module = getattr(pipe, component, None)
        if module is not None and getattr(module, "peft_config", None):
            try:
                module.set_adapters(names, weights=weights)
            except (TypeError, ValueError) as exc:
                print(f"[model] 提示: {component} 的适配器强度未提交（{type(exc).__name__}）")
    actual = adapter_scales_readback(pipe)
    if actual and any(abs(actual.get(n, -1.0) - float(scales[n])) > 1e-3 for n in names):
        print(f"[model] 警告: 适配器强度未被接受，请求={scales} 实际={actual}", flush=True)
    print(f"[model] 适配器强度: " + ", ".join(f"{n}={scales[n]:.2f}" for n in names)
          + (f"（回读 {actual}）" if actual else "（无法回读校验）"), flush=True)


def adapter_scales_readback(pipe) -> dict:
    """回读 peft 层上真正的 scaling 值，用于校验强度是否真的生效。

    只读第一层即可：diffusers 的 set_weights_and_activate_adapters 会对所有
    BaseTunerLayer 统一设置，因此样本层能代表整体。拿不到时返回空 dict。
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        return {}
    unet = getattr(pipe, "unet", pipe)
    for _, module in unet.named_modules():
        if isinstance(module, BaseTunerLayer):
            return {str(k): round(float(v), 4) for k, v in module.scaling.items()}
    return {}


def set_adapter_scale(pipe, scale, adapter_name="default"):
    """按图调整某个适配器的强度，**保持其余适配器不变**。

    注意（实测踩过的坑）：diffusers 的 `pipe.set_adapters(单个名字, weights=[...])`
    会把**未列出的适配器一并置零并停用**。本项目在非人物上同时挂
    身份权重（强度 0）与概念权重（强度 0.35），若每次只设置一个，
    另一个会被悄悄关掉——结果与预期不符却没有任何报错。
    """
    target = float(scale)
    if adapter_name not in _active_adapter_scales(pipe):
        # 还没加载任何适配器时只登记强度，不去管线里设置：
        # 对空的适配器集合调用 set_adapters 会抛 ValueError（实测）。
        _ADAPTER_SCALES[adapter_name] = target
        return target
    _ADAPTER_SCALES[adapter_name] = target
    _apply_adapter_scales(pipe)
    return target


def unload_adapters(pipe) -> None:
    """卸载全部适配器：角色切换时必须调用，防止上一角色的条件残留。"""
    _ADAPTER_SCALES.clear()
    try:
        pipe.unload_lora_weights()
        print("[model] 已卸载全部 LoRA 适配器")
    except Exception as exc:   # 未加载过适配器时 diffusers 会抛错，属正常情况
        print(f"[model] 卸载适配器跳过（{type(exc).__name__}）：当前管线没有已加载的适配器")


def restore_base_unet(pipe, unet_lora) -> None:
    """把 pipe.unet 从 peft / accelerate 包装里还原成裸 UNet。

    用途：同一进程内训练第二个分支（身份之后训练概念）时必须先拆掉上一个 LoRA 包装，
    否则注入新 LoRA 时找不到注意力线性层——实测报"UNet 中未找到注意力投影层"。
    只影响进程内对象，不动磁盘上的权重文件。
    """
    import torch.nn as nn
    model = unet_lora
    if hasattr(model, "get_base_model"):            # peft 包装
        model = model.get_base_model()
    inner = getattr(model, "module", None)          # accelerate / DDP 包装
    if inner is not None:
        model = inner.get_base_model() if hasattr(inner, "get_base_model") else inner
    # peft 的 base_model 是 LoraModel，其 .model 才是真正的 UNet
    candidate = getattr(model, "model", None)
    if isinstance(candidate, nn.Module) and hasattr(candidate, "down_blocks"):
        model = candidate
    if not hasattr(model, "down_blocks"):
        raise RuntimeError("无法还原裸 UNet：对象结构与预期不符")
    pipe.unet = model
    for parameter in pipe.unet.parameters():
        parameter.requires_grad_(False)
    print("[model] 已把 UNet 还原为裸底模（用于训练下一个分支）")


# ---------------------------------------------------------------------------
# 受控参考增强（可选，方案 §8）
# ---------------------------------------------------------------------------
def reference_available() -> bool:
    path = os.path.join(config.IP_ADAPTER_PATH, config.REFERENCE_ADAPTER_SUBFOLDER,
                        config.REFERENCE_ADAPTER_WEIGHT)
    encoder = os.path.join(config.IP_ADAPTER_PATH, config.REFERENCE_IMAGE_ENCODER,
                           "model.safetensors")
    return os.path.isfile(path) and os.path.isfile(encoder)


def load_reference_adapter(pipe, scales=None) -> dict:
    """把 SDXL IP-Adapter 挂到当前管线上，返回审计 dict。

    与角色 LoRA 并存：两者都挂在 UNet 上，作用强度分别控制。
    参考增强只用于补细节，不能替代权重，因此各槽强度默认取小值。
    """
    if not reference_available():
        raise FileNotFoundError(
            "IP-Adapter 权重不存在: "
            f"{os.path.join(config.IP_ADAPTER_PATH, config.REFERENCE_ADAPTER_SUBFOLDER, config.REFERENCE_ADAPTER_WEIGHT)}")
    scales = scales or [config.REFERENCE_FACE_SCALE, config.REFERENCE_COSTUME_SCALE]
    errors = []
    for weight in (config.REFERENCE_ADAPTER_WEIGHT, config.REFERENCE_ADAPTER_FALLBACK):
        if not weight:
            continue
        try:
            pipe.load_ip_adapter(config.IP_ADAPTER_PATH,
                                 subfolder=config.REFERENCE_ADAPTER_SUBFOLDER,
                                 weight_name=weight,
                                 image_encoder_folder=config.REFERENCE_IMAGE_ENCODER)
            pipe.set_ip_adapter_scale(list(scales))
            print(f"[model] 已加载参考增强（IP-Adapter）：{weight} scale={list(scales)}")
            return {"loaded": True, "scales": list(scales), "weight": weight,
                    "fallbacks_tried": errors}
        except Exception as exc:
            errors.append(f"{weight}: {type(exc).__name__}: {exc}")
            print(f"[model] 参考增强权重 {weight} 加载失败: {type(exc).__name__}: {exc}")
    raise RuntimeError("参考增强的全部候选权重都无法加载：" + " | ".join(errors))


def set_reference_scales(pipe, scales) -> list:
    pipe.set_ip_adapter_scale(list(scales))
    return list(scales)


def unload_reference_adapter(pipe) -> None:
    try:
        pipe.unload_ip_adapter()
        print("[model] 已卸载参考增强（IP-Adapter）")
    except Exception as exc:
        print(f"[model] 卸载参考增强跳过（{type(exc).__name__}）")


def write_adapter_audit(path, audit) -> None:
    """把实际加载的权重与强度写入结果旁的审计文件（方案 §26）。"""
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    print(f"[model] 权重审计已保存: {path}")
