#!/usr/bin/env python3
"""Carga el modelo en Ollama con num_ctx = N y un prompt mínimo, y cuenta qué ha reservado.

Lee /api/ps (size, size_vram, contexto), la columna PROCESSOR de `ollama ps` y, del log del servidor,
la caché KV, el búfer de cómputo y las capas en GPU. Descarga el modelo antes y después (así suelta también
la caché de prompts que el llama-server de Ollama guarda en RAM, hasta 8 GiB).

Vigila el swap durante la carga y mata el runner si crece más de --limite-swap MB.

  python3 scripts/ollama_carga.py 4096 16384 32768 65536
  python3 scripts/ollama_carga.py 131072 --solo-decision   # qué capas pondría en GPU, sin cargar pesos ni caché
  python3 scripts/ollama_carga.py 98304 --sin-seguridad    # salta la regla del 85 % (búsquedas cerca del límite)

Resultados en resultados/ollama_carga.csv.
"""
import argparse
import csv
import os
import random
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util  # noqa: E402


def lineas_nuevas(log, desde):
    if not log:
        return ""
    with open(log, errors="replace") as f:
        return "".join(f.readlines()[desde:])


def num_lineas(log):
    if not log:
        return 0
    with open(log, errors="replace") as f:
        return sum(1 for _ in f)


def cargar(n, a, log):
    util.ollama_descargar()
    time.sleep(2)
    l0 = num_lineas(log)
    s0, p0 = util.swap()
    smax = [s0]
    res = {}

    def peticion():
        try:
            util.api("/api/generate", {"model": util.MODELO, "prompt": f"[{random.randint(0, 1 << 30)}] hola",
                                       "stream": False, "options": {"num_ctx": n, "num_predict": 1}}, timeout=900)
        except Exception as e:  # el runner muerto devuelve 500
            res["error"] = str(e)[:120]

    t = threading.Thread(target=peticion, daemon=True)
    t.start()
    estado = "cargado"
    while t.is_alive():
        s, _ = util.swap()
        smax[0] = max(smax[0], s)
        nuevo = lineas_nuevas(log, l0)
        if a.solo_decision and re.search(r"fitting params to free memory took|offloaded \d+/\d+ layers|KV buffer size", nuevo):
            util.matar_runner_ollama()
            estado = "solo decisión"
            break
        if s - s0 > a.limite_swap:
            util.matar_runner_ollama()
            estado = f"CORTADO swap +{s - s0:.0f} MB"
            break
        time.sleep(0.25)
    t.join(timeout=60)
    ps = util.api("/api/ps").get("models", []) if estado == "cargado" else []
    cli = util.ejecutar(["ollama", "ps"]).stdout if estado == "cargado" else ""
    s1, p1 = util.swap()
    nuevo = lineas_nuevas(log, l0)

    def g(p):
        m = re.search(p, nuevo)
        return m[1] if m else ""

    fila = {
        "num_ctx": n, "estado": estado, "error": res.get("error", ""),
        "size": ps[0].get("size", "") if ps else "", "size_vram": ps[0].get("size_vram", "") if ps else "",
        "context_length": ps[0].get("context_length", "") if ps else "",
        "processor": " ".join(cli.splitlines()[1].split()[4:6]) if len(cli.splitlines()) > 1 else "",
        "capas_gpu": g(r"offloaded (\d+/\d+) layers") or g(r"\(.*\): +(\d+) layers,") or
                     ("todas" if "entire model can be fit" in nuevo or "no changes needed" in nuevo else ""),
        "kv_gpu_MiB": g(r"(?:MTL0|CUDA0) KV buffer size =\s+([\d.]+)"), "kv_cpu_MiB": g(r"CPU KV buffer size =\s+([\d.]+)"),
        "computo_gpu_MiB": g(r"(?:MTL0|CUDA0) compute buffer size =\s+([\d.]+)"),
        "pesos_gpu_MiB": g(r"(?:MTL0|CUDA0)(?:_Mapped)? model buffer size =\s+([\d.]+)"),
        "fa": g(r"Flash Attention (enabled|disabled)"), "tipo_cache": g(r"K \((\w+)\)"),
        "proyectado_MiB": g(r"projected to use (\d+) MiB"),
        "swap_inicio_MB": round(s0, 1), "swap_max_MB": round(smax[0], 1), "swap_crece_MB": round(smax[0] - s0, 1),
        "paginas_expulsadas": p1 - p0,
    }
    if cli:
        print(cli.rstrip())
    util.ollama_descargar()
    return fila


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("contextos", nargs="+", type=int)
    p.add_argument("--solo-decision", action="store_true")
    p.add_argument("--limite-swap", type=float, default=256)
    p.add_argument("--sin-seguridad", action="store_true")
    p.add_argument("--csv", default=util.resultados("ollama_carga.csv"))
    a = p.parse_args()
    log = util.log_ollama()
    if not log:
        print("  (sin log del servidor: arráncalo con scripts/servidor_ollama.sh para ver caché y capas)", file=sys.stderr)
    for n in a.contextos:
        if not (a.solo_decision or a.sin_seguridad):
            util.comprobar_seguridad(n)
        fila = cargar(n, a, log)
        nuevo = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fila))
            if nuevo:
                w.writeheader()
            w.writerow(fila)
        print("  " + " · ".join(f"{k}={v}" for k, v in fila.items() if v not in ("", None)))


if __name__ == "__main__":
    main()
