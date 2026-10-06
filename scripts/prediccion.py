#!/usr/bin/env python3
"""Predice la generación según lo lleno que esté el contexto y la compara con lo medido por scripts/bench.sh.

  tok/s(n) ≈ ancho_de_banda_efectivo / (bytes_pesos + bytes_KV(n))
  ancho_de_banda_efectivo ≈ tok/s con el contexto vacío × bytes_pesos   (aproximación)

Pesos: los que están en la GPU según el log de scripts/llamacpp_cache.sh («model buffer size»), o --pesos-mib.
Si no hay log, usa el tamaño del GGUF (algo mayor: incluye los embeddings, que se quedan en CPU).

  python3 scripts/prediccion.py resultados/bench_fp16.jsonl [--tipo f16]
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util  # noqa: E402


def pesos_gpu_mib():
    for log in sorted(glob.glob(os.path.join(util.RESULTADOS, "logs", "llamacpp_*.log")), key=os.path.getmtime, reverse=True):
        m = re.search(r"(?:MTL0|CUDA0)(?:_Mapped)? model buffer size =\s+([\d.]+) MiB", open(log, errors="replace").read())
        if m:
            return float(m[1]), f"log {os.path.basename(log)}"
    return os.path.getsize(util.gguf()) / 2**20, "tamaño del GGUF (aproximado)"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench")
    p.add_argument("--tipo", default="f16", choices=list(util.BYTES_POR_VALOR))
    p.add_argument("--pesos-mib", type=float)
    a = p.parse_args()
    filas = [json.loads(l) for l in open(a.bench) if l.strip()]
    med = lambda f: sorted(f["samples_ts"])[len(f["samples_ts"]) // 2]
    tg = {f["n_depth"]: med(f) for f in filas if f["n_gen"] > 0}
    if 0 not in tg:
        sys.exit("Hace falta la profundidad 0 en el benchmark (-d 0,…).")
    pesos, fuente = (a.pesos_mib, "--pesos-mib") if a.pesos_mib else pesos_gpu_mib()
    kv_mib = util.kv_bytes_por_token(util.metadatos(), a.tipo) / 2**20
    ancho = tg[0] * pesos
    print(f"pesos en GPU: {util.es(pesos)} MiB ({fuente}) · caché {a.tipo}: {util.es(kv_mib * 1024)} KiB/token")
    print(f"ancho de banda efectivo ≈ {util.es(tg[0], 2)} tok/s × {util.es(pesos)} MiB = {util.es(ancho / 1024)} GiB/s "
          f"({util.es(ancho * 2**20 / 1e9)} GB/s)\n")
    print(f"  {'profundidad':>11} {'predicho':>9} {'medido':>8} {'error':>7}")
    for d in sorted(tg):
        pred = ancho / (pesos + d * kv_mib)
        print(f"  {d:>11} {util.es(pred, 2):>9} {util.es(tg[d], 2):>8} {util.es((tg[d] - pred) / pred * 100):>6} %")
    print("\n(con caché cuantizada la predicción falla: descuantizar cuesta cómputo y la generación no sube)")


if __name__ == "__main__":
    main()
