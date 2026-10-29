# -*- coding: utf-8 -*-
"""命令行入口：voxmirror clone / gen / rescan。

示例
----
  # 1) 注册一条声线（参考音 + 参考文稿必须成对，文稿与参考音逐字一致）
  voxmirror clone --name me --ref me.wav --text "Good morning. I like ..."

  # 2) 把一段文稿合成成"我"的声音，自动逐句质检 + 离群重采
  voxmirror gen --voice me --text lines.txt --out out/

  # 3) 哪几句还是飘了？只重采这几句（备份原版到 parts_orig/）
  voxmirror rescan --out out/

  # 多声线：文稿里用  "声线名|要念的文本"  区分，其余行默认 --voice
"""
import argparse
import os
import sys

import voxmirror.core as core
from voxmirror.engine import BACKENDS

MODEL_DEFAULT = os.environ.get("VOXMIRROR_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")


def _add_common(p):
    p.add_argument("--model", default=MODEL_DEFAULT,
                   help="TTS 模型：本地目录 / HF 仓库名 / 别名 0.6b|1.7b（默认 %s）" % MODEL_DEFAULT)
    p.add_argument("--backend", default="qwen3tts", choices=list(BACKENDS) or ["qwen3tts"],
                   help="推理后端（默认 qwen3tts）")
    p.add_argument("--device", default="cuda:0", help="推理设备（默认 cuda:0）")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--data-dir", default=None,
                   help="声线库目录（默认 $VOXMIRROR_HOME 或 ~/.voxmirror）")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="voxmirror", description="克隆音色 → 质检 → 生成音频")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clone", help="注册一条声线（参考音 + 参考文稿）")
    c.add_argument("--name", required=True, help="声线名（之后 gen/rescan 用它引用）")
    c.add_argument("--ref", required=True, help="参考音 wav（5~15s 清晰人声）")
    c.add_argument("--text", required=True, help="参考音里实际说出的文字，逐字一致（含标点）")
    c.add_argument("--force", action="store_true", help="覆盖已存在的同名声线")
    _add_common(c)

    g = sub.add_parser("gen", help="把文稿合成成克隆声线，逐句质检 + 重采")
    g.add_argument("--voice", help="默认声线名（文稿里用 名|文本 覆盖单行）")
    g.add_argument("--text", required=True, help="目标文稿 .txt（每行一句；可选 声线|文本）")
    g.add_argument("--out", required=True, help="输出目录")
    g.add_argument("--lang", default="auto", help="语言：auto|Chinese|English（默认 auto）")
    g.add_argument("--seed", default="20261016",
                   help="固定种子（每句 seed+idx；重采 seed+idx+1000t）；random 表示不固定")
    g.add_argument("--tries", type=int, default=4, help="单句重采上限（默认 4）")
    g.add_argument("--temp", type=float, default=0.9, help="采样温度（默认 0.9）")
    g.add_argument("--top-p", type=float, default=1.0, dest="top_p", help="top_p（默认 1.0）")
    g.add_argument("--no-guard", dest="guard", action="store_false", default=True,
                   help="关闭质检（默认开）")
    _add_common(g)

    r = sub.add_parser("rescan", help="对已有 parts 重新质检，只重采离群句")
    r.add_argument("--out", required=True, help="之前 gen 的输出目录")
    r.add_argument("--voice", help="默认声线名（一般可省略，从 lines.json 读取）")
    r.add_argument("--tries", type=int, default=5, help="每句最多试几个种子（默认 5）")
    r.add_argument("--temp", type=float, default=0.7, help="重采温度（默认 0.7）")
    r.add_argument("--top-p", type=float, default=0.9, dest="top_p", help="重采 top_p（默认 0.9）")
    r.add_argument("--seed", default="20261016", help="与 gen 同源的种子（重采时自动偏移避免重复）")
    _add_common(r)

    l = sub.add_parser("voices", help="列出已注册声线")
    _add_common(l)

    args = ap.parse_args(argv)

    if args.cmd == "voices":
        for n in core.list_voices(args.data_dir):
            print(n)
        return

    if args.cmd == "clone":
        d = core.register_voice(args.name, args.ref, args.text, args.force, args.data_dir)
        print("已注册声线 %r -> %s" % (args.name, d))
        return

    seed = None if str(args.seed).lower() in ("random", "-1", "none") else int(args.seed)

    if args.cmd == "gen":
        if args.backend == "qwen3tts" and not args.voice and _needs_default_voice(args.text):
            sys.exit("文稿里不是每行都带 '声线|文本'，请加 --voice 指定默认声线")
        core.generate(args.backend, args.model, args.text, args.out, args.voice,
                      args.lang, seed, args.tries, args.temp, args.top_p, args.guard,
                      args.device, args.dtype, args.data_dir)
        return

    if args.cmd == "rescan":
        core.rescan(args.backend, args.model, args.out, args.voice, args.tries,
                    args.temp, args.top_p, seed, args.device, args.dtype, args.data_dir)
        return


def _needs_default_voice(text_file):
    with open(text_file, encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            if "|" in s and not s.startswith("|"):
                continue
            return True
    return False


if __name__ == "__main__":
    main()
