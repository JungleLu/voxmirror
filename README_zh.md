# 声镜 · VoxMirror

> 🌐 **[English](README.md)** · 镜像任意音色，逐句自动质检。

克隆音色 → 逐句质检 → 生成音频。一个极简的开源语音克隆工具，**只做三件事**：

1. **克隆音色** —— 给一段参考音 + 它的文稿，抽出一个可复用的"声线"。
2. **校验** —— 对每一句生成结果做纯信号质检（音高 / 响度 / 音色），抓出"飘了、偏亮、偏闷、离群"的句子。
3. **生成音频** —— 把文稿逐句合成成该声线，自动把离群句换种子重采，直到通过或取到最不离群的一版。

底层默认用 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（Apache-2.0，3 秒零样本克隆），但后端是可插拔的：换 `CoquiBackend` / `XVoiceBackend` 继承同一个接口即可，整条流程一行不用改。

## 安装

```bash
pip install -e .                 # 基础（质检/克隆逻辑，无需显卡）
pip install -e ".[qwen3tts]"     # 加上 Qwen3-TTS 生成后端（需要 torch）
# 注意：qwen-tts 若 PyPI 名不同，请按 QwenLM/Qwen3-TTS 官方 README 安装
```

模型本体首次运行会自动从 Hugging Face 下载（或用 `--model` 指向本地目录）。
国内可设 `export HF_ENDPOINT=https://hf-mirror.com`。

## 快速开始

```bash
# 1) 注册一条声线：参考音 + 参考文稿必须成对，文稿与参考音逐字一致（含标点）
voxmirror clone --name me --ref me.wav --text "Good morning. I like to start my day with a short walk."

# 2) 把文稿合成成"我"的声音（自动逐句质检 + 离群重采）
voxmirror gen --voice me --text lines.txt --out out/

# 3) 还有几句飘着？只重采这几句（原版备份到 out/parts_orig/）
voxmirror rescan --out out/
```

|                                    |                             |
| ---------------------------------- | --------------------------- |
| `lines.txt` 每行一句；想用多人声线，在行内写 `声线名\|要念的文本`，其余行用 `--voice` 当默认： |

```
今天天气真好。
me|This is a cloned voice.
friend|你好，我是另一个声线。
```

输出（`out/`）：

| 文件                          | 说明                        |
| --------------------------- | ------------------------- |
| `parts/001_me.wav …`        | 逐句音频（质检/重采的对象）            |
| `output.wav` / `output.mp3` | 拼接后的整段（句间 0.4s 静音）        |
| `lines.json`                | 句子→声线→文本映射（rescan 重采时复用）  |
| `report.txt`                | 生成报告（含每句重采轮次）             |
| `guard_report.txt`          | 纯信号质检明细（每句 F0 / RMS / 音色） |

## 它怎么判定"飘了"？

完全不跑 ASR、不加载任何模型，只看声学指标（详见 `voxmirror/guard.py`）：

- **G1 响度飙**：比参考音响 > +3dB
- **G2 起调过高**：句首/参考 > 2.0 且 句首/正文 > 1.5
- **G3 正文塌陷**：正文/参考 < 0.70
- **G4 正文偏高**：正文/参考 > 1.25
- **G5 音色偏亮 / G6 音色偏闷**：频谱质心相对本批中位 > 1.20 / < 0.86（或高频比偏 > 1.8dB）
- **G7 音高离群**：body F0 稳健 z > 3 且 相对中位偏离 > 12%

阈值都能用环境变量临时覆盖（如 `SG_RMS_GAIN_DB`），方便调参。

## 合规提醒

- 只克隆**你有权使用的声音**。不要拿别人的声音做冒犯性或欺诈性内容。
- Qwen3-TTS 本身 Apache-2.0 可商用；但**参考音的版权归你/授权方**，仓库不内置任何真人声音。
- 本项目示例里的 `examples/ref_example.wav` 是一段合成测试音，仅用于跑通流程，**不能**当真实声线用。

## License

Apache-2.0
