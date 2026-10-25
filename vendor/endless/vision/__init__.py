# -*- coding: utf-8 -*-
"""
vision/ —— 视觉模块（规格 §2 目录结构）

对外导出（供 test.py 等上层调用）：
- subject_analyzer.analyze_subject / get_style_token   输入主体理解（§7 / §17）
- segmentation.load_sam2 / segment_subject / get_subject_mask_or_full / unload_sam
                                                       SAM2 主体分割（§9 / §27）
"""

from importlib import import_module

_EXPORTS={
    'analyze_subject':'subject_analyzer','get_style_token':'subject_analyzer',
    'load_sam2':'segmentation','segment_subject':'segmentation',
    'get_subject_mask_or_full':'segmentation','unload_sam':'segmentation',
}

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value=getattr(import_module(f'{__name__}.{_EXPORTS[name]}'),name)
    globals()[name]=value
    return value

__all__ = [
    "analyze_subject",
    "get_style_token",
    "load_sam2",
    "segment_subject",
    "get_subject_mask_or_full",
    "unload_sam",
]
