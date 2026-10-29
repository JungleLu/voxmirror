---
name: voxmirror
description: "Clone a voice from a short reference audio + its transcript, then synthesize any script in that cloned voice with automatic per-sentence pitch/timbre QC and outlier re-sampling. 镜像任意音色，逐句自动质检，生成音频. Use when the user wants to clone a voice, generate spoken audio in a cloned voice, batch-produce audiobook or podcast narration, or fix cloned sentences that sound 'drifting', too bright, too muffled, or off-pitch. Triggers EN: clone my voice, generate audio in X's voice, voxmirror, re-sample the off lines, voice cloning, zero-shot TTS. Triggers ZH: 克隆音色, 克隆我的声音, 用我的声音生成, 声音飘了, 重采, 声镜, 语音克隆."
agent_created: true
---

# VoxMirror — Voice cloning with automatic QC

Mirror any voice, verify every line. VoxMirror is a minimal open-source CLI that
does exactly three things: **clone** a timbre, **verify** every generated line,
and **generate** audio. It deliberately excludes music beds, subtitles, video,
PDF, and Excel — only clean audio comes out.

## When to use

- The user supplies (or wants to create) a reference voice and asks to speak new text in it.
- Batch narration tasks (audiobooks, language-learning podcasts, dubbing) where consistent timbre across many sentences matters.
- A previously cloned sentence "drifted" (pitch/timbre out of distribution) and needs re-sampling.

## Install / locate

VoxMirror is a Python package exposing a `voxmirror` CLI.

```bash
# From PyPI (once published)
pip install voxmirror

# Or from a local clone of the repo
pip install -e .                 # QC / clone logic only (CPU, no GPU)
pip install -e ".[qwen3tts]"     # + Qwen3-TTS generation backend (needs torch + GPU)
```

- Default backend: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0, 3-second zero-shot cloning). The engine is pluggable — any backend implementing the `TTSBackend` interface works unchanged.
- Model weights auto-download from Hugging Face on first `gen`/`rescan`, or pass `--model /path/to/local`. Set `HF_ENDPOINT=https://hf-mirror.com` in China.
- Voice registration and QC run on CPU; only `gen`/`rescan` need a GPU.

## Workflow

### 1. Clone a voice (reference audio + transcript, paired)

The reference transcript must be **word-for-word identical** to what the reference audio actually says (punctuation included). This anchors the timbre.

```bash
voxmirror clone --name me --ref me.wav \
  --text "Good morning. I like to start my day with a short walk."
```

This stores `ref.wav` + `ref.txt` under the voice library (`~/.voxmirror/voices/me/` by default; override with `--data-dir`).

### 2. Generate a script in that voice

Write the target script as plain text, one sentence per line. For multiple voices, prefix a line with `voice_name|text`; other lines use `--voice` as default.

```bash
voxmirror gen --voice me --text lines.txt --out out/
```

Every sentence is synthesized, then passed through the signal guard. Outliers are automatically re-sampled with a different seed until they pass (or the best non-outlier take is kept). Outputs in `out/`:

| File | Meaning |
| --- | --- |
| `parts/001_me.wav …` | per-sentence audio (the QC / re-sample unit) |
| `output.wav` / `output.mp3` | concatenated whole (0.4s gaps) |
| `lines.json` | sentence→voice→text map (reused by `rescan`) |
| `report.txt` | generation report (re-sample rounds per line) |
| `guard_report.txt` | per-line signal QC details (F0 / RMS / timbre) |

### 3. Re-sample only the lines that still drift

`rescan` re-checks existing `parts`, re-synthesizes only the flagged sentences (originals backed up to `out/parts_orig/`), then re-concatenates.

```bash
voxmirror rescan --out out/
```

## How "drift" is detected (the differentiator)

No ASR, no model — pure acoustics (`voxmirror/guard.py`). A line is flagged when:

- **G1** loudness spike: RMS > +3 dB vs reference
- **G2** high onset: head/ref > 2.0 **and** head/body > 1.5
- **G3** body collapse: body/ref < 0.70
- **G4** body too high: body/ref > 1.25
- **G5/G6** timbre too bright / too muffled: spectral centroid vs batch median > 1.20 / < 0.86 (or HF ratio > +1.8 dB)
- **G7** pitch outlier: robust z > 3 **and** relative deviation > 12%

All thresholds are overridable via environment variables (e.g. `SG_RMS_GAIN_DB`).

## Compliance

- Clone only voices the user is authorized to use. Do not produce deceptive or fraudulent audio of others.
- The repo ships **no real human voices**; `examples/ref_example.wav` is a synthetic test tone and must not be used as a real voice.

## References

- `README.md` / `README_zh.md` — full bilingual docs.
- `voxmirror/guard.py` — the QC implementation and threshold constants.
- `examples/` — a runnable demo (`ref_example.wav` + sample scripts).

## Tips

- List registered voices with `voxmirror voices`.
- Use `--data-dir` to isolate voice libraries per project.
- When a generation run is interrupted, simply re-run the same `gen` command — existing `parts` are reused and only missing sentences are synthesized.
