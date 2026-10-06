#!/usr/bin/env python3
"""Qué decide `--fit` de llama.cpp cuando no le das ni -c ni -ngl.

Lanza llama-server con el GGUF y todo por defecto, lee del log la decisión (contexto, capas en GPU)
y lo mata (SIGKILL) en cuanto `--fit` termina su cálculo, ANTES de cargar los pesos y la caché. No llena la memoria.

  python3 scripts/fit_decision.py              # solo la decisión (seguro)
  python3 scripts/fit_decision.py --completo   # deja que reserve la caché, vigilando el swap (puede mandar el
                                               # equipo al swap: se corta si crece más de --limite-swap MB)
"""
import argparse
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--completo", action="store_true")
    p.add_argument("--limite-swap", type=float, default=256)
    p.add_argument("--puerto", default=os.environ.get("PUERTO", "18080"))
    a = p.parse_args()
    if util.ollama_cargado():
        sys.exit("Ollama tiene un modelo cargado. Descárgalo antes: ollama stop " + util.MODELO)
    log = util.resultados("logs", "fit_decision.log")
    s0, _ = util.swap()
    with open(log, "w") as f:
        proc = subprocess.Popen(["llama-server", "-m", util.gguf(), "-lv", "4", "--port", a.puerto], stdout=f, stderr=f)
    estado, t0 = "decisión leída", time.time()
    try:
        while proc.poll() is None and time.time() - t0 < 300:
            texto = open(log).read()
            if not a.completo and re.search(r"fitting params to free memory took|offloaded \d+/\d+ layers", texto):
                break  # --fit ya ha decidido: cortar antes de cargar pesos y caché
            s, _ = util.swap()
            if not a.completo and s - s0 > 128:  # algo no ha ido bien: cortar ya
                estado = f"CORTADO: swap +{s - s0:.0f} MB"
                break
            if a.completo:
                if s - s0 > a.limite_swap:
                    estado = f"CORTADO: swap +{s - s0:.0f} MB"
                    break
                if "listening on" in texto:
                    estado = "cargado"
                    break
            time.sleep(0.05)
    finally:
        proc.kill()  # SIGKILL: con SIGTERM, llama-server termina de reservar la caché antes de salir
        proc.wait(timeout=30)
    texto = open(log).read()
    for patron in (r"common_params_fit_impl: (context size unset[^\n]*)", r"common_params_fit_impl: (projected to use[^\n]*)",
                   r"common_params_fit_impl: (cannot meet[^\n]*)", r"common_params_fit_impl: (will leave[^\n]*)",
                   r"common_params_fit_impl: (context size reduced[^\n]*)", r"common_params_fit_impl: (entire model can be fit[^\n]*)",
                   r"common_params_fit_impl: +- (\w+ \(.*\): +\d+ layers[^\n]*)", r"load_tensors: (offloaded[^\n]*)",
                   r"llama_kv_cache: +(\w+ KV buffer size[^\n]*)"):
        for m in re.findall(patron, texto):
            print("  " + m.split(" -> need")[0])
    s1, _ = util.swap()
    print(f"\n  {estado} · swap {s0:.0f} → {s1:.0f} MB · log: {log}")


if __name__ == "__main__":
    main()
