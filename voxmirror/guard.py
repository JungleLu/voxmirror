# -*- coding: utf-8 -*-
"""纯信号离群守卫：不跑 ASR、不加载任何大模型，只看 F0 与响度/音色。

设计要点（基于 50 句 / 多参考音频的实测标定）：
  · 男女统一用「相对参考音频的比值」做基础判据（G1~G4），因为零样本克隆会搬运
    参考的音色、音域与电平。
  · 音高单看任何一个维度都会误伤：正常语调的 head/body 比值完全可以到 2.1，
    所以「起调过高」必须用 head/ref 与 head/body 的**联合**条件。
  · 音色类判据（G5~G7）比「本批次同说话人中位」而不是比参考音频：
    参考音频的频谱质心往往比模型实际输出亮很多，拿它当标尺会把系统性差异当成正常，
    句内漂移（尤其变亮/气声）就再也看不出来。
  · 短句（<1.2s 或浊音帧不足）测不准 F0/音色，G5~G7 一律跳过，否则片头片尾会误报。

判据（任一命中 ⇒ 该句离群，建议换种子重采）：
  G1 响度飙：   rms − ref_rms            > +3.0 dB
  G2 起调过高： head/ref                 > 2.0   且  head/body > 1.5
  G3 正文塌陷： body/ref                 < 0.70
  G4 正文偏高： body/ref                 > 1.25
  G5 音色偏亮： cent/批次中位            > 1.20  或  hf − 批次中位 > +1.8 dB
  G6 音色偏闷： cent/批次中位            < 0.86
  G7 音高离群： body 的稳健 z            > 3.0   且  |body/中位 − 1| > 12%

阈值都可用环境变量临时覆盖（SG_RMS_GAIN_DB 等），便于调参。
"""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf

# ---------- 判据阈值（要调只改这里；也可用环境变量临时覆盖，便于测试） ----------
RMS_GAIN_DB = float(os.environ.get("SG_RMS_GAIN_DB", 3.0))   # G1：比参考响多少 dB 算异常
HEAD_RATIO = float(os.environ.get("SG_HEAD_RATIO", 2.0))     # G2：句首音高 / 参考音高
HEAD_BODY = float(os.environ.get("SG_HEAD_BODY", 1.5))       # G2：句首 / 正文 的比值
BODY_RATIO = float(os.environ.get("SG_BODY_RATIO", 0.70))    # G3：正文音高 / 参考音高（下界）
BODY_HIGH = float(os.environ.get("SG_BODY_HIGH", 1.25))      # G4：正文音高 / 参考音高（上界）

# 只标记、不重采的"接近上限"线（写进报告供人工复查）
NEAR_RMS_DB = float(os.environ.get("SG_NEAR_RMS_DB", 2.0))

# G5/G6 音色判据（相对本批次同说话人中位）
TIMBRE_CENT_HIGH = float(os.environ.get("SG_TIMBRE_CENT_HIGH", 1.20))
TIMBRE_CENT_LOW = float(os.environ.get("SG_TIMBRE_CENT_LOW", 0.86))
TIMBRE_HF_DB = float(os.environ.get("SG_TIMBRE_HF_DB", 1.8))
BODY_Z = float(os.environ.get("SG_BODY_Z", 3.0))   # G7：body F0 相对本批次中位的稳健 z
BODY_REL = float(os.environ.get("SG_BODY_REL", 0.12))    # G7 第二条件：相对幅度（12%≈2 个半音）
MAD_FLOOR = float(os.environ.get("SG_MAD_FLOOR", 0.05))  # 稳健 z 的离散度下限（× 中位）
MIN_DUR = float(os.environ.get("SG_MIN_DUR", 1.2))       # 短于此时长不测音高/音色（测不准）

# ---------- 分帧参数 ----------
FRAME = 0.04           # 40ms 帧
HOP = 0.02             # 20ms 跳
FMIN, FMAX = 60, 400   # 音高搜索范围（Hz）
CLARITY = 0.35         # 自相关清晰度门限
HEAD_T0, HEAD_T1 = 0.05, 0.65   # 句首窗口
MIN_VOICED = 4         # 至少这么多浊音帧才信这个统计量


def _frames(x, sr):
    fl, hop = int(sr * FRAME), int(sr * HOP)
    for i in range(0, len(x) - fl, hop):
        yield i, x[i:i + fl]


def f0_track(y, sr, t0=None, t1=None):
    """返回该区间内每帧的 F0（Hz），无浊音/不清晰的帧丢弃。"""
    x = np.asarray(y, dtype=np.float32).reshape(-1)
    if t0 is not None:
        x = x[int(t0 * sr):int(t1 * sr) if t1 is not None else len(x)]
    vals = []
    for _, s in _frames(x, sr):
        if np.sqrt((s ** 2).mean()) < 1e-3:
            continue
        s = s - s.mean()
        c = np.correlate(s, s, "full")[len(s) - 1:]
        c = c / (c[0] + 1e-12)
        lo, hi = int(sr / FMAX), min(int(sr / FMIN), len(c) - 1)
        if hi <= lo:
            continue
        k = int(np.argmax(c[lo:hi])) + lo
        if c[k] < CLARITY:
            continue
        vals.append(sr / k)
    return np.array(vals)


def measure(y, sr):
    """一句话的信号体检。head / body 在浊音帧不足时记 0（调用方跳过相应判据）。"""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    hv = f0_track(y, sr, 0.0, HEAD_T1)
    hw = f0_track(y, sr, HEAD_T0, HEAD_T1)
    bv = f0_track(y, sr, HEAD_T1, None)
    rms = 20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)
    cent, hf = timbre(y, sr)
    return dict(
        dur=len(y) / sr,
        rms=float(rms),
        peak=float(np.abs(y).max()),
        head=float(np.median(hw)) if len(hw) >= MIN_VOICED else 0.0,
        head_n=len(hw),
        headmax=float(hv.max()) if len(hv) else 0.0,
        body=float(np.median(bv)) if len(bv) >= MIN_VOICED else 0.0,
        body_n=len(bv),
        cent=cent,
        hf=hf,
    )


def timbre(y, sr):
    """浊音帧的谱质心(Hz) 与 高频比(3-8kHz / 0.2-3kHz, dB)。

    为什么加这两个：F0 中位正常、只有「音色变亮/气声变多」的句子，原有三条判据全部放行。
    实测某批第 31 句：body/ref 正常，但质心 +30%、高频比 +2.1dB —— 听感就是"这句突然飘起来"。
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    fl = int(sr * FRAME)
    win = np.hanning(fl).astype(np.float32)
    # 只按能量门限取帧，不做浊音筛选：气声/齿音正是"飘"的来源，滤掉就看不见了
    rms_all = np.array([np.sqrt((y[i:i + fl] ** 2).mean())
                        for i in range(0, max(1, len(y) - fl), int(sr * HOP))])
    if not len(rms_all):
        return 0.0, 0.0
    gate = max(rms_all.max() * 0.06, 10 ** (-50 / 20))
    cents, hfs = [], []
    for k, (i, s) in enumerate(_frames(y, sr)):
        if k >= len(rms_all) or rms_all[k] < gate:
            continue
        s = (s - s.mean()) * win
        mag = np.abs(np.fft.rfft(s)) + 1e-9
        fr = np.fft.rfftfreq(fl, 1 / sr)
        band = (fr >= 80) & (fr <= 8000)
        m, f = mag[band], fr[band]
        cents.append(float((m * f).sum() / m.sum()))
        hfs.append(float(10 * np.log10((m[f >= 3000].sum() + 1e-9) /
                                       (m[f < 3000].sum() + 1e-9))))
    if not cents:
        return 0.0, 0.0
    return float(np.median(cents)), float(np.median(hfs))


def voice_ok(m):
    """这句话是否长到足以测音高/音色。片尾短句只有 0.6s，测什么都是噪声。"""
    return bool(m.get("body", 0) > 0 and m.get("body_n", 0) >= MIN_VOICED
                and m.get("dur", 0) >= MIN_DUR)


def robust_scale(vals):
    """本批次 body 的中位与稳健离散度（MAD 过小给下限，避免把自然波动误判为离群）。"""
    v = np.asarray([x for x in vals if x and x > 0], dtype=float)
    if not len(v):
        return 0.0, 0.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, max(1.4826 * mad, MAD_FLOOR * med)


def outlier_f0(m, med, scale):
    """G7：body 相对本批次中位的偏离。z 与相对幅度**同时**越界才算。"""
    if not voice_ok(m) or not med or not scale:
        return False, [], 0.0, 0.0
    z = (m["body"] - med) / scale
    rel = m["body"] / med - 1
    if abs(z) > BODY_Z and abs(rel) > BODY_REL:
        return True, ["音高离群 z=%+.1f (%+.0f%%)" % (z, rel * 100)], z, rel
    return False, [], z, rel


def check_timbre(m, tbase):
    """相对本批次同说话人中位音色的偏离（tbase 由 rescan/gen 统计）。"""
    bad, why, sev = False, [], 0.0
    if not tbase or not m.get("cent") or not tbase.get("cent"):
        return bad, why, sev
    if not voice_ok(m):
        return bad, why, sev
    cr = m["cent"] / tbase["cent"]
    dh = m["hf"] - tbase["hf"]
    if cr > TIMBRE_CENT_HIGH or dh > TIMBRE_HF_DB:
        bad = True
        why.append("音色偏亮 质心/中位 %.2f 高频%+.2f dB" % (cr, dh))
        sev += max(0.0, cr - TIMBRE_CENT_HIGH) * 3 + max(0.0, dh - TIMBRE_HF_DB)
    if cr < TIMBRE_CENT_LOW:
        bad = True
        why.append("音色偏闷 质心/中位 %.2f" % cr)
        sev += (TIMBRE_CENT_LOW - cr) * 3
    return bad, why, float(sev)


def baseline(ref_path, cache=None):
    """参考音频的基准（F0 中位 / RMS）。"""
    if cache is None:
        cache = {}
    if ref_path not in cache:
        y, sr = sf.read(ref_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        v = f0_track(y, sr)
        cache[ref_path] = dict(
            f0=float(np.median(v)) if len(v) else 0.0,
            rms=float(20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)),
            dur=len(y) / sr)
    return cache[ref_path]


def check(m, base):
    """返回 (是否离群, 命中的判据说明列表, severity 分数)。severity 越小越好。"""
    bad, why, sev = False, [], 0.0
    drms = m["rms"] - base["rms"]
    if drms > RMS_GAIN_DB:
        bad = True
        why.append("响度飙 +%.2fdB" % drms)
        sev += (drms - RMS_GAIN_DB)
    if m["head"] > 0 and m["body"] > 0 and base["f0"] > 0:
        hr = m["head"] / base["f0"]
        hb = m["head"] / m["body"]
        br = m["body"] / base["f0"]
        if hr > HEAD_RATIO and hb > HEAD_BODY:
            bad = True
            why.append("起调过高 head/ref %.2f 且 head/body %.2f" % (hr, hb))
            sev += 3.0 * ((hr / HEAD_RATIO - 1.0) + (hb / HEAD_BODY - 1.0))
        if br < BODY_RATIO:
            bad = True
            why.append("正文塌陷 body/ref %.2f" % br)
            sev += 3.0 * (BODY_RATIO - br) / BODY_RATIO
        if br > BODY_HIGH:
            bad = True
            why.append("正文偏高 body/ref %.2f" % br)
            sev += 3.0 * (br / BODY_HIGH - 1.0)
    return bad, why, float(sev)


def describe(m, base):
    """一行式的指标摘要（写进 report）。"""
    hr = m["head"] / base["f0"] if (m["head"] and base["f0"]) else 0.0
    br = m["body"] / base["f0"] if (m["body"] and base["f0"]) else 0.0
    return ("head %6.1fHz(n=%2d, ref%.2f)  body %6.1fHz(ref%.2f)  "
            "rms %7.2f (ref%+.2f)  peak %.3f"
            % (m["head"], m["head_n"], hr, m["body"], br,
               m["rms"], m["rms"] - base["rms"], m["peak"]))


def threshold_desc():
    return (f"rms-ref > +{RMS_GAIN_DB}dB | "
            f"head/ref > {HEAD_RATIO} 且 head/body > {HEAD_BODY} | "
            f"body/ref 不在 [{BODY_RATIO}, {BODY_HIGH}] | "
            f"音色偏亮/偏闷 或 音高离群（相对本批中位）")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python -m voxmirror.guard --ref ref.wav sample1.wav [sample2.wav ...]")
        raise SystemExit(1)
    ref = sys.argv[2]
    b = baseline(ref)
    print("参考基准：F0 %.1fHz  RMS %.2fdBFS  %.2fs" % (b["f0"], b["rms"], b["dur"]))
    for t in sys.argv[3:]:
        y, sr = sf.read(t, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        m = measure(y, sr)
        bad, why, _ = check(m, b)
        print("  %-50s %s" % (t, describe(m, b)))
        print("      → %s%s" % ("离群！" if bad else "通过",
                                ("  " + "；".join(why)) if why else ""))
