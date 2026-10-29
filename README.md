# VoxMirror

> 🌐 **[简体中文](README_zh.md)** · Mirror any voice. Verify every line.

Clone a timbre → verify every line → generate audio. A minimal open-source voice-cloning CLI that does **exactly three things**:

1. **Clone** — give a reference audio + its transcript, get a reusable voice.
2. **Verify** — run a pure-signal QC on every generated sentence (pitch / loudness / timbre) to catch lines that drift, go bright, go muffled, or fall out of distribution.
3. **Generate** — synthesize a script in that voice, auto re-sampling outliers until they pass (or until the best non-outlier version is kept).

Backend defaults to [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0, 3-second zero-shot cloning), but the engine is pluggable: implement a `CoquiBackend` / `XVoiceBackend` against the same interface and the whole pipeline is unchanged.

## Install

```bash
pip install -e .                 # base (QC / clone logic, no GPU needed)
pip install -e ".[qwen3tts]"     # + Qwen3-TTS generation backend (needs torch)
```

The model weights download from Hugging Face on first run (or point `--model` at a local dir). Mirror: `export HF_ENDPOINT=https://hf-mirror.com`.

## Quick start

```bash
# 1) Register a voice: reference audio + its transcript, word-for-word (punctuation included)
voxmirror clone --name me --ref me.wav --text "Good morning. I like to start my day with a short walk."

# 2) Synthesize a script in that voice (auto QC + outlier re-sampling)
voxmirror gen --voice me --text lines.txt --out out/

# 3) A few lines still off? Re-sample only those (originals backed up to out/parts_orig/)
voxmirror rescan --out out/
```

|                                                                                            |                                                   |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| `lines.txt` is one sentence per line. For multiple voices, prefix a line with \`voice_name | text`; remaining lines use `--voice\` as default. |

Outputs in `out/`: `parts/###_voice.wav` (per sentence), `output.wav` / `output.mp3` (concatenated, 0.4s gaps), `lines.json` (mapping, reused by rescan), `report.txt`, `guard_report.txt`.

## How does it decide a line "drifted"?

No ASR, no model — pure acoustics (see `voxmirror/guard.py`):

- G1 loudness spike: RMS > +3dB vs reference
- G2 high onset: head/ref > 2.0 **and** head/body > 1.5
- G3 body collapse: body/ref < 0.70
- G4 body too high: body/ref > 1.25
- G5 too bright / G6 too muffled: spectral centroid vs batch median > 1.20 / < 0.86 (or HF ratio > +1.8 dB)
- G7 pitch outlier: robust z > 3 **and** relative deviation > 12%

Every threshold is overridable via env vars (e.g. `SG_RMS_GAIN_DB`).

## License

Apache-2.0. Clone only voices you are authorized to use.
