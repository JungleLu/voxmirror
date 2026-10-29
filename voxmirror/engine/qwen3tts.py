# -*- coding: utf-8 -*-
"""Qwen3-TTS 后端（默认）。

零样本克隆：给参考音 + 参考文稿 → create_voice_clone_prompt 抽声线；
同一份声线 prompt 复用到多句台词，保证整批同一个人的声音不漂移。
模型本体 Apache-2.0，来自 https://github.com/QwenLM/Qwen3-TTS

依赖（可选）：
    pip install torch
    pip install qwen-tts        # 或按官方仓库 README 安装
"""
from .base import TTSBackend, register


@register("qwen3tts")
class Qwen3TTSBackend(TTSBackend):
    def __init__(self, model_path, device="cuda:0", dtype="bfloat16"):
        import torch
        from qwen_tts import Qwen3TTSModel

        if dtype == "bfloat16":
            dt = torch.bfloat16
        elif dtype == "float16":
            dt = torch.float16
        else:
            dt = torch.float32

        self.device = device
        self.model = Qwen3TTSModel.from_pretrained(
            model_path, device_map=device, dtype=dt,
            attn_implementation="sdpa")

    def clone(self, ref_wav, ref_text, x_vector_only_mode=False):
        return self.model.create_voice_clone_prompt(
            ref_wav, ref_text, x_vector_only_mode=x_vector_only_mode)[0]

    def synth(self, texts, languages, prompts, temperature=0.9, top_p=1.0, top_k=50):
        out, sr = self.model.generate_voice_clone(
            text=texts, language=languages, voice_clone_prompt=prompts,
            temperature=temperature, top_p=top_p, top_k=top_k)
        import numpy as np
        return [np.asarray(o, dtype=np.float32).reshape(-1) for o in out], sr
