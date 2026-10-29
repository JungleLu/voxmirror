# -*- coding: utf-8 -*-
"""voxmirror 的核心编排：注册声线、逐句生成（带质检）、离群句重采。

只做三件事：克隆音色、校验、生成音频。不碰垫乐、字幕、视频、PDF、Excel。
"""
import json
import os
import shutil
import subprocess
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf

import voxmirror.guard as sg
from voxmirror.engine import BACKENDS

DEFAULT_HOME = os.environ.get("VOXMIRROR_HOME") or os.path.expanduser("~/.voxmirror")
GAP = 0.4  # 拼接时句间静音秒数


def data_dir():
    return DEFAULT_HOME


def voice_path(name, data_dir=None):
    return os.path.join(data_dir or DEFAULT_HOME, "voices", name)


def list_voices(data_dir=None):
    base = os.path.join(data_dir or DEFAULT_HOME, "voices")
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base)
                 if os.path.isfile(os.path.join(base, d, "ref.wav")))


def register_voice(name, ref_wav, ref_text, force=False, data_dir=None):
    """注册一条声线：把参考音与参考文稿存到本地，并做一次基准体检。"""
    dest = voice_path(name, data_dir)
    if os.path.isdir(dest) and not force:
        raise SystemExit("声线 %r 已存在；用 --force 覆盖" % name)
    os.makedirs(dest, exist_ok=True)

    y, sr = sf.read(ref_wav, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    sf.write(os.path.join(dest, "ref.wav"), y.astype(np.float32), sr)
    with open(os.path.join(dest, "ref.txt"), "w", encoding="utf-8") as f:
        f.write(ref_text.strip() + "\n")

    base = sg.baseline(os.path.join(dest, "ref.wav"))
    if base["f0"] <= 0 or base["dur"] < 3.0:
        print("  ⚠ 参考音 %.2fs、参考 F0 %.1fHz：偏短或清音过多，克隆质量可能不稳"
              % (base["dur"], base["f0"]))
    meta = dict(name=name, sr=int(sr), ref_f0=base["f0"], dur=base["dur"])
    with open(os.path.join(dest, "voice.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return dest


def load_voice(name, data_dir=None):
    dest = voice_path(name, data_dir)
    if not os.path.isfile(os.path.join(dest, "ref.wav")):
        raise SystemExit("声线 %r 不存在；先 voxmirror clone 注册" % name)
    with open(os.path.join(dest, "ref.txt"), encoding="utf-8") as f:
        ref_text = f.read().strip()
    return dict(name=name, ref_wav=os.path.join(dest, "ref.wav"), ref_text=ref_text)


def cjk(s):
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)


def detect_lang(text):
    return "Chinese" if cjk(text) else "English"


def _build_backend(model, backend_name, device, dtype):
    cls = BACKENDS.get(backend_name)
    if cls is None:
        raise SystemExit("后端 %r 不可用；可选：%s" % (backend_name, ", ".join(BACKENDS) or "（均未安装）"))
    return cls(model, device=device, dtype=dtype)


def _synth_sentence(backend, prompt, text, lang, base, seed, idx, tries, temp, top_p,
                    guard, seed_offset=0):
    """对单句做「合成 + 质检 + 重采」，返回 (y, sr, m, used, passed, hist)。"""
    import torch  # 仅生成路径需要，且后端已懒加载了 qwen-tts
    y, m, hist, best, passed = None, None, [], None, False
    for t in range(tries):
        if seed is not None:
            torch.manual_seed(seed + idx + seed_offset + 1000 * t)
        out, sr = backend.synth([text], [lang], [prompt],
                                temperature=temp, top_p=top_p, top_k=40)
        y = out[0]
        if not guard:
            return y, sr, None, 1, True, []
        m = sg.measure(y, sr)
        bad, why, sev = sg.check(m, base)
        if not bad:
            return y, sr, m, t + 1, True, []
        hist.append(dict(round=t + 1, why=list(why), sev=round(sev, 3)))
        if best is None or sev < best[0]:
            best = (sev, y, m)
    _, y, m = best
    return y, sr, m, tries, False, hist


def _concat(wavs, sr, gap=GAP):
    if not wavs:
        return np.zeros(1, dtype=np.float32)
    pad = np.zeros(int(sr * gap), dtype=np.float32)
    pieces = []
    for y in wavs:
        pieces.append(y)
        pieces.append(pad)
    return np.concatenate(pieces)


def _maybe_mp3(wav_path, sr, whole):
    mp3 = wav_path[:-4] + ".mp3"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "f32le",
                        "-ar", str(sr), "-ac", "1", "-i", "pipe:0", "-b:a", "160k", mp3],
                       input=np.ascontiguousarray(whole, dtype="<f4").tobytes(), check=True)
        return mp3
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _write_reports(out_dir, lines, gstats, base_by_voice, seed, guard, lang_label):
    lines_out = [
        "voxmirror 生成报告",
        "种子 %s（每句 seed+idx；重采 seed+idx+1000t）" % (seed if seed is not None else "随机"),
        "句子数 %d" % len(lines),
        "质检 %s（阈值 %s）" % ("开" if guard else "关", sg.threshold_desc()),
    ]
    if guard and gstats:
        n_retry = sum(1 for g in gstats if g["used"] > 1)
        lines_out.append("")
        lines_out.append("重采 %d 句 / 共 %d 句" % (n_retry, len(gstats)))
    lines_out.append("")
    for g in gstats:
        lines_out.append("  [%03d] %-8s 轮次 %d  %s" % (g["idx"], g["voice"], g["used"], g["verdict"]))
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    if not guard:
        return
    with open(os.path.join(out_dir, "guard_report.txt"), "w", encoding="utf-8") as f:
        f.write("纯信号离群守卫报告  %d 句\n" % len(gstats))
        f.write("阈值 %s\n\n" % sg.threshold_desc())
        for name, b in base_by_voice.items():
            f.write("基准 %-8s 参考 F0 %.1fHz  RMS %.2fdBFS\n" % (name, b["f0"], b["rms"]))
        f.write("\n" + "-" * 110 + "\n")
        f.write("%-6s%-8s%7s%6s  %s  判定\n" % ("句", "声线", "时长", "轮次", "指标"))
        f.write("-" * 110 + "\n")
        for g in gstats:
            bb = base_by_voice[g["voice"]]
            v = g["verdict"]
            f.write("%03d   %-8s%7.2f%6d  %s  %s\n" % (
                g["idx"], g["voice"], g["dur"], g["used"],
                sg.describe(g["m"], bb), v))


def generate(backend_name, model, text_file, out_dir, default_voice, lang, seed, tries,
             temp, top_p, guard, device="cuda:0", dtype="bfloat16", data_dir=None):
    """读目标文稿 → 逐句克隆声线合成 → 质检重采 → 出音频 + 报告。"""
    # 解析文稿
    with open(text_file, encoding="utf-8") as f:
        raw = [ln.rstrip("\n") for ln in f]
    lines = []  # (idx, voice, text)
    for ln in raw:
        s = ln.strip()
        if not s:
            continue
        if "|" in s and not s.startswith("|"):
            name, text = s.split("|", 1)
            name, text = name.strip(), text.strip()
        else:
            name, text = default_voice, s
        if not name:
            raise SystemExit("某行没有指定声线，且未给 --voice 默认值")
        lines.append((len(lines) + 1, name, text))

    backend = _build_backend(model, backend_name, device, dtype)

    # 收集出现过的声线，建 prompt
    names = sorted({n for _, n, _ in lines})
    profiles, prompts, bases = {}, {}, {}
    for n in names:
        p = load_voice(n, data_dir)
        profiles[n] = p
        prompts[n] = backend.clone(p["ref_wav"], p["ref_text"])
        bases[n] = sg.baseline(p["ref_wav"])

    pdir = os.path.join(out_dir, "parts")
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(out_dir, "lines.json"), "w", encoding="utf-8") as f:
        json.dump([{"idx": i, "voice": v, "text": t} for i, v, t in lines],
                  f, ensure_ascii=False, indent=2)

    wavs, gstats = [], []
    t_all = time.time()
    for idx, name, text in lines:
        lg = lang if lang != "auto" else detect_lang(text)
        y, sr, m, used, passed, hist = _synth_sentence(
            backend, prompts[name], text, lg, bases[name], seed, idx, tries, temp, top_p, guard)
        sf.write(os.path.join(pdir, "%03d_%s.wav" % (idx, name)), y, sr)
        wavs.append(y)
        verdict = "通过" if used == 1 else \
            ("重采到第 %d 轮通过" % used if passed else "仍未通过（取最不离群）")
        if hist:
            verdict += "（首轮：%s）" % hist[0]["why"][0]
        gstats.append(dict(idx=idx, voice=name, dur=len(y) / sr, used=used, verdict=verdict, m=m))
        print("  [%3d/%d] %5.2fs %-8s %s" % (idx, len(lines), len(y) / sr, name, verdict))

    whole = _concat(wavs, sr)
    wav_path = os.path.join(out_dir, "output.wav")
    sf.write(wav_path, whole, sr)
    mp3 = _maybe_mp3(wav_path, sr, whole)
    _write_reports(out_dir, lines, gstats, bases, seed, guard, lang)
    print("\n→ %s%s" % (wav_path, ("  + %s" % mp3) if mp3 else "  （无 ffmpeg，未出 mp3）"))
    if guard:
        print("→ %s（质检详情）" % os.path.join(out_dir, "guard_report.txt"))
    return wav_path


def rescan(backend_name, model, out_dir, default_voice, tries, temp, top_p, seed,
           device="cuda:0", dtype="bfloat16", data_dir=None):
    """对已有 parts 重新质检，把离群句换种子重采（不重跑整批）。"""
    pdir = os.path.join(out_dir, "parts")
    lj = os.path.join(out_dir, "lines.json")
    if not os.path.isdir(pdir) or not os.path.isfile(lj):
        raise SystemExit("找不到 %s / lines.json；先跑过 voxmirror gen" % pdir)
    with open(lj, encoding="utf-8") as f:
        lines = json.load(f)

    backend = _build_backend(model, backend_name, device, dtype)

    names = sorted({d["voice"] for d in lines})
    profiles, prompts, bases = {}, {}, {}
    for n in names:
        p = load_voice(n, data_dir)
        profiles[n] = p
        prompts[n] = backend.clone(p["ref_wav"], p["ref_text"])
        bases[n] = sg.baseline(p["ref_wav"])

    # 现状体检
    recs = []
    for d in lines:
        idx, name, text = d["idx"], d["voice"], d["text"]
        p = os.path.join(pdir, "%03d_%s.wav" % (idx, name))
        y, sr = sf.read(p, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        m = sg.measure(y, sr)
        bad, why, sev = sg.check(m, bases[name])
        recs.append(dict(idx=idx, voice=name, text=text, path=p, y=y, sr=sr,
                         m=m, bad=bad, why=list(why), sev=sev))

    # 批次内基准（G5~G7 用）：同声线 body/ref 中位 + 音色中位 + 稳健 z
    target, tb, f0z = {}, {}, {}
    for n in names:
        sel = [r for r in recs if r["voice"] == n and r["m"]["body"] > 0]
        if not sel:
            continue
        rat = [r["m"]["body"] / bases[n]["f0"] for r in sel]
        target[n] = float(np.median(rat))
        med, scale = sg.robust_scale([r["m"]["body"] for r in sel])
        f0z[n] = (med, scale)
        tb[n] = dict(cent=float(np.median([r["m"]["cent"] for r in sel])),
                     hf=float(np.median([r["m"]["hf"] for r in sel])))

    print("批次内 body/ref 中位：%s" %
          "  ".join("%s %.2f" % (k, v) for k, v in target.items()))

    for r in recs:
        extra = []
        n = r["voice"]
        if n in tb:
            bt, whyt, _ = sg.check_timbre(r["m"], tb[n])
            extra += whyt
            if n in f0z:
                bf, whyf, _, _ = sg.outlier_f0(r["m"], *f0z[n])
                extra += whyf
        r["extra"] = extra
        if extra and not r["bad"]:
            r["bad"] = True
            r["why"] = r["why"] + extra

    todo = [r for r in recs if r["bad"]]
    print("待处理 %d 句 / 共 %d 句" % (len(todo), len(recs)))
    for r in todo:
        print("  %03d_%-8s %s" % (r["idx"], r["voice"], "；".join(r["why"])))
    if not todo:
        print("全部通过，无需重采。")
        return

    orig_dir = os.path.join(out_dir, "parts_orig")
    os.makedirs(orig_dir, exist_ok=True)

    summary = []
    for r in todo:
        n, i = r["voice"], r["idx"]
        b = bases[n]
        tgt = target.get(n, 1.0)
        best = None
        import torch  # 仅重采路径需要
        for t in range(1, tries + 1):
            if seed is not None:
                torch.manual_seed(seed + 5000 + i + 1000 * t)  # +5000 与 gen 错开，探索不同样本
            y, sr = backend.synth([r["text"]], [detect_lang(r["text"])],
                                  [prompts[n]], temperature=temp, top_p=top_p, top_k=40)
            y = y[0]
            m = sg.measure(y, sr)
            bad, why, sev = sg.check(m, b)
            rr = m["body"] / b["f0"] if (m["body"] and b["f0"]) else 0.0
            dist = abs(rr - tgt)
            tbw = tb.get(n, {})
            med, scale = f0z.get(n, (0.0, 1.0))
            ok = sg.voice_ok(m)
            crt = abs(m["cent"] / tbw["cent"] - 1) / 0.20 if (tbw and ok) else 0.0
            hft = abs(m["hf"] - tbw["hf"]) / sg.TIMBRE_HF_DB if (tbw and ok) else 0.0
            f0t = abs((m["body"] - med) / scale) / sg.BODY_Z if (med and ok) else 0.0
            cost = (0 if not bad else 100) + dist + crt + hft + f0t
            print("    %03d t%d  body/ref %.2f (目标 %.2f)  %s  %s" % (
                i, t, rr, tgt, sg.describe(m, b),
                "通过" if not bad else "离群 " + "；".join(why)))
            if best is None or cost < best[0]:
                best = (cost, y, m, t, not bad)
            if not bad and dist <= 0.08 and max(crt, hft, f0t) < 1.0:
                break
        _c, y, m, t, ok = best
        old = os.path.join(orig_dir, "%03d_%s.wav" % (i, n))
        if not os.path.exists(old):
            shutil.copyfile(r["path"], old)
        sf.write(r["path"], y, r["sr"])
        summary.append((i, n, r["m"]["body"] / b["f0"] if b["f0"] else 0.0,
                        m["body"] / b["f0"] if b["f0"] else 0.0, t, ok))

    # 重拼 + 重新出质检报告（基于重采后的 parts 完整重新体检）
    wavs, srs = [], set()
    for d in lines:
        y, sr = sf.read(os.path.join(pdir, "%03d_%s.wav" % (d["idx"], d["voice"])),
                        dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        wavs.append(y.reshape(-1))
        srs.add(sr)
    sr = srs.pop() if len(srs) == 1 else max(srs)
    whole = _concat(wavs, sr)
    wav_path = os.path.join(out_dir, "output.wav")
    sf.write(wav_path, whole, sr)
    mp3 = _maybe_mp3(wav_path, sr, whole)
    print("\n→ %s%s" % (wav_path, ("  + %s" % mp3) if mp3 else ""))

    # 重新体检所有 parts（重采后的最终状态）并写 guard_report.txt
    recs2 = []
    for d in lines:
        y, sr = sf.read(os.path.join(pdir, "%03d_%s.wav" % (d["idx"], d["voice"])),
                        dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        m = sg.measure(y.reshape(-1), sr)
        bad, why, _ = sg.check(m, bases[d["voice"]])
        extra = []
        n = d["voice"]
        if n in tb:
            bt, whyt, _ = sg.check_timbre(m, tb[n])
            extra += whyt
            if n in f0z:
                bf, whyf, _, _ = sg.outlier_f0(m, *f0z[n])
                extra += whyf
        recs2.append(dict(idx=d["idx"], voice=n, m=m, dur=len(y) / sr,
                          bad=bad, why=why + extra))

    with open(os.path.join(out_dir, "guard_report.txt"), "w", encoding="utf-8") as f:
        f.write("重采后质检报告  %d 句\n" % len(recs2))
        f.write("阈值 %s\n\n" % sg.threshold_desc())
        for name, b in bases.items():
            f.write("基准 %-8s 参考 F0 %.1fHz  RMS %.2fdBFS\n" % (name, b["f0"], b["rms"]))
        f.write("\n" + "-" * 110 + "\n")
        f.write("%-6s%-8s%7s  %s  判定\n" % ("句", "声线", "时长", "指标"))
        f.write("-" * 110 + "\n")
        for r in recs2:
            bb = bases[r["voice"]]
            v = "通过" if not r["bad"] else "离群：" + "；".join(r["why"])
            f.write("%03d   %-8s%7.2f  %s  %s\n" % (
                r["idx"], r["voice"], r["dur"], sg.describe(r["m"], bb), v))

    print("→ %s（最终质检详情）" % os.path.join(out_dir, "guard_report.txt"))
    print("原版备份在 %s" % orig_dir)
    print("本批重采 %d 句，其中 %d 句仍未通过（取最不离群）。"
          % (len(todo), sum(1 for *_x, ok in summary if not ok)))
