#!/usr/bin/env python3
"""Espera hasta la primera palabra (TTFT) y velocidad de generación en Ollama, por streaming.

Cada pasada: `ollama stop` (suelta también la caché de prompts que el llama-server de Ollama guarda en RAM),
precarga con «hola» sin cronometrar y después el prompt largo con un prefijo único (la caché de prefijos
no puede reutilizar nada). La pasada 0 es de calentamiento y se descarta. Vigila el swap durante cada pasada.

  python3 scripts/espera.py --num-ctx 20480 --tokens 15000            # ~15.000 tokens, 3 repeticiones
  python3 scripts/espera.py --num-ctx 36864 --tokens 32000 --repeticiones 3
  python3 scripts/espera.py --num-ctx 16384 --tokens 12000 --texto mi_texto.txt

El texto por defecto son los manuales locales de zsh y bash (man zshall, man bash). Los tokens se estiman
con 1,31 tokens por palabra (medido con el tokenizador de Llama 3.1 sobre ese texto); el recuento real es
prompt_eval_count. Resultados en resultados/espera.csv.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util  # noqa: E402

TOKENS_POR_PALABRA = 1.31


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-ctx", type=int, required=True)
    p.add_argument("--tokens", type=int, required=True, help="tamaño aproximado del prompt")
    p.add_argument("--repeticiones", type=int, default=3)
    p.add_argument("--num-predict", type=int, default=128)
    p.add_argument("--texto", help="fichero de texto propio en vez de las páginas de man")
    p.add_argument("--etiqueta", default="")
    p.add_argument("--csv", default=util.resultados("espera.csv"))
    a = p.parse_args()
    if a.tokens + a.num_predict >= a.num_ctx:
        sys.exit("El prompt más la respuesta no caben en num_ctx: Ollama truncaría el prompt a la mitad del contexto.")
    util.comprobar_seguridad(a.num_ctx)
    palabras = open(a.texto).read().split() if a.texto else util.texto_manual()
    n_pal = int(a.tokens / TOKENS_POR_PALABRA)
    if n_pal > len(palabras):
        sys.exit(f"El texto tiene {len(palabras)} palabras y harían falta {n_pal}. Usa --texto con uno más largo.")
    etiqueta = a.etiqueta or f"ctx{a.num_ctx}_tok{a.tokens}"

    def prompt(i):
        desde = (i * 7919) % max(1, len(palabras) - n_pal)  # cada pasada empieza en otro punto del texto
        return (f"Sesión {random.randint(10**8, 10**9)}-{i}. Lee este manual y después resume en una frase de qué trata.\n\n"
                + " ".join(palabras[desde:desde + n_pal]) + "\n\nResume en una frase de qué trata el texto.")

    nuevo = not os.path.exists(a.csv)
    f = open(a.csv, "a", newline="")
    campos = ["etiqueta", "pasada", "num_ctx", "prompt_eval_count", "espera_s", "prefill_tok_s", "eval_count",
              "generacion_tok_s", "swap_crece_MB", "marcada", "primera_frase"]
    w = csv.DictWriter(f, fieldnames=campos)
    if nuevo:
        w.writeheader()
    medidas = []
    for i in range(a.repeticiones + 1):
        util.ollama_descargar()
        time.sleep(2)
        util.api("/api/generate", {"model": util.MODELO, "prompt": "hola", "stream": False,
                                   "options": {"num_ctx": a.num_ctx, "num_predict": 1}})
        s0, _ = util.swap()
        smax, parar = [s0], [False]

        def vigilar():
            while not parar[0]:
                smax[0] = max(smax[0], util.swap()[0])
                time.sleep(0.5)

        hilo = threading.Thread(target=vigilar, daemon=True)
        hilo.start()
        cuerpo = {"model": util.MODELO, "prompt": prompt(i), "stream": True,
                  "options": {"num_ctx": a.num_ctx, "num_predict": a.num_predict}}
        req = urllib.request.Request(util.OLLAMA + "/api/generate", json.dumps(cuerpo).encode(),
                                     {"Content-Type": "application/json"})
        t0, espera, texto, ultimo = time.time(), None, "", {}
        with urllib.request.urlopen(req, timeout=3600) as r:
            for linea in r:
                d = json.loads(linea)
                if espera is None and d.get("response", "").strip():
                    espera = time.time() - t0
                texto += d.get("response", "")
                ultimo = d
        parar[0] = True
        hilo.join()
        n = ultimo.get("prompt_eval_count", 0)
        fila = {"etiqueta": etiqueta, "pasada": "calentamiento" if i == 0 else i, "num_ctx": a.num_ctx,
                "prompt_eval_count": n, "espera_s": round(espera or 0, 2),
                "prefill_tok_s": round(n / (ultimo.get("prompt_eval_duration", 1) / 1e9), 1),
                "eval_count": ultimo.get("eval_count"),
                "generacion_tok_s": round(ultimo.get("eval_count", 0) / (ultimo.get("eval_duration", 1) / 1e9), 2),
                "swap_crece_MB": round(smax[0] - s0, 1), "marcada": "SWAP" if smax[0] - s0 > 128 else "",
                "primera_frase": re.split(r"(?<=[.!?])\s", texto.strip())[0][:160]}
        w.writerow(fila)
        f.flush()
        print("  " + " · ".join(f"{k}={v}" for k, v in fila.items() if k != "primera_frase"))
        if i:
            medidas.append(fila)
    util.ollama_descargar()
    med = lambda k: sorted(m[k] for m in medidas)[len(medidas) // 2]
    print(f"\nmediana: {util.es(med('prompt_eval_count'), 0)} tokens · {util.es(med('espera_s'))} s hasta la primera palabra · "
          f"{util.es(med('generacion_tok_s'))} tok/s generando")
    if any(m["marcada"] for m in medidas):
        print("⚠️  Alguna pasada tiró de swap: repítela con menos apps abiertas antes de dar el número por bueno.")


if __name__ == "__main__":
    main()
