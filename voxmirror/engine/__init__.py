# -*- coding: utf-8 -*-
"""推理后端包。导入即可用：from voxmirror.engine import Qwen3TTSBackend, BACKENDS"""
from .base import TTSBackend, BACKENDS, register  # noqa: F401

try:                                    # 后端按需加载，避免没装 qwen-tts/torch 时整包崩溃
    from .qwen3tts import Qwen3TTSBackend  # noqa: F401
except Exception:                       # pragma: no cover
    Qwen3TTSBackend = None
