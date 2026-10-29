# -*- coding: utf-8 -*-
"""TTS 后端抽象层。

把"克隆音色 + 文本合成"这件事从具体模型里解耦出来：
voxmirror 默认用 Qwen3-TTS，但你完全可以写一个 `CoquiBackend` / `XVoiceBackend`
继承这个基类，注册到 `BACKENDS` 里，整个克隆→校验→生成流程一行都不用改。
"""
from abc import ABC, abstractmethod


class TTSBackend(ABC):
    """语音合成后端的统一接口。"""

    @abstractmethod
    def clone(self, ref_wav, ref_text, x_vector_only_mode=False):
        """用参考音 + 参考文稿生成一条可复用的"声线 prompt"，返回后端原生对象。"""
        raise NotImplementedError

    @abstractmethod
    def synth(self, texts, languages, prompts, temperature=0.9, top_p=1.0, top_k=50):
        """合成一组句子。

        texts      : List[str]   每条要念的文本
        languages  : List[str]   每条文本的语言（"Chinese"/"English"/...）
        prompts    : List[Any]   与 texts 等长的声线 prompt（来自 clone）
        返回 (List[np.ndarray], int sr)，每条音频为一维 float32。
        """
        raise NotImplementedError


# 名称 -> 后端类（CLI 用 --backend qwen3tts 选择）
BACKENDS = {}


def register(name):
    def deco(cls):
        BACKENDS[name] = cls
        return cls
    return deco
