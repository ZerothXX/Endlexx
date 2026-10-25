# -*- coding: utf-8 -*-
"""
vlm —— VLM 封装子包（规格 §2 目录结构）。

核心实现集中在 vlm/vlm_utils.py：
    VLMClient    : Qwen2.5-VL 统一封装（加载链 / analyze_image / generate_tags / unload）
    parse_json   : 容错 JSON 解析
    repair_json  : 启发式 JSON 修复
    get_default_memory : 无 VLM 时的最小语义记忆兜底
    CHARACTER_PROMPT / SUBJECT_PROMPT / ACCESSORY_PROMPT / EVAL_PROMPT / CAPTION_PROMPT

本 __init__.py 保持为空实现（仅文档），避免 import vlm 时立即拉起
torch/transformers（数据预处理等轻量场景可能只需要 CAPTION_PROMPT 常量）。
"""
