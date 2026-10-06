#!/usr/bin/env python3
"""La fórmula de la caché KV, en papel.

  python3 scripts/kv.py modelo                    # bytes por token y total a 4K…128K del modelo de Ollama (MODELO)
  python3 scripts/kv.py hf REPO [REPO…]           # lo mismo desde el config.json de Hugging Face (se descarga)
  python3 scripts/kv.py seguridad N [--tipo q8_0] [--fa off]   # regla del 85 % antes de cargar N tokens

Fórmula: 2 × capas × cabezas KV × dimensión de cabeza × bytes por valor (f16 = 2, q8_0 = 1,0625, q4_0 = 0,5625).
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util  # noqa: E402

CONTEXTOS = (4096, 8192, 16384, 32768, 65536, 131072)


def tabla(md, titulo):
    print(f"\n{titulo}: {md['capas']} capas · {md['cabezas_kv']} cabezas KV · dimensión {md['dim_cabeza']}")
    print(f"  {'tipo':6} {'KiB/token':>10} " + " ".join(f"{n // 1024:>8}K" for n in CONTEXTOS) + "   (MiB)")
    for tipo in util.BYTES_POR_VALOR:
        b = util.kv_bytes_por_token(md, tipo)
        print(f"  {tipo:6} {util.es(b / 1024):>10} " + " ".join(f"{util.es(n * b / 2**20, 0):>9}" for n in CONTEXTOS))
    if md.get("cabezas") and md["cabezas"] != md["cabezas_kv"]:
        b = 2 * md["capas"] * md["cabezas"] * md["dim_cabeza"] * 2
        print(f"  (trampa: con las {md['cabezas']} cabezas de consulta en vez de las {md['cabezas_kv']} KV saldrían "
              f"{util.es(b / 1024, 0)} KiB/token)")


def desde_hf(repo):
    """Descarga config.json (cada modelo tiene su licencia; se guarda en resultados/configs/)."""
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    try:
        c = json.load(urllib.request.urlopen(url, timeout=30))
    except OSError as e:
        sys.exit(f"{repo}: no se puede descargar ({e}). Si el repo está restringido, usa un espejo público.")
    with open(util.resultados("configs", repo.replace("/", "_") + ".json"), "w") as f:
        json.dump(c, f, indent=1)
    t = c.get("text_config", c)
    cabezas = t["num_attention_heads"]
    md = {"capas": t["num_hidden_layers"], "cabezas": cabezas,
          "cabezas_kv": t.get("num_key_value_heads", cabezas),
          "dim_cabeza": t.get("head_dim") or t["hidden_size"] // cabezas}
    tabla(md, repo)
    if t.get("head_dim") and t["head_dim"] != t["hidden_size"] // cabezas:
        print(f"  (ojo: head_dim = {t['head_dim']}, pero hidden_size / cabezas daría {t['hidden_size'] // cabezas})")
    tipos = t.get("layer_types")
    if t.get("sliding_window") and (tipos or t.get("sliding_window_pattern")):
        if tipos:
            completas = sum(1 for x in tipos if x == "full_attention")
        else:
            completas = md["capas"] // t["sliding_window_pattern"]
        print(f"  ventana deslizante: {completas} de {md['capas']} capas con atención completa "
              f"(ventana de {t['sliding_window']} tokens en el resto) → a 128K, "
              f"{util.es(131072 * 2 * completas * md['cabezas_kv'] * md['dim_cabeza'] * 2 / 2**30)} GiB "
              f"en vez de {util.es(131072 * util.kv_bytes_por_token(md) / 2**30)} GiB")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="orden", required=True)
    sub.add_parser("modelo")
    h = sub.add_parser("hf")
    h.add_argument("repos", nargs="+")
    s = sub.add_parser("seguridad")
    s.add_argument("n_ctx", type=int)
    s.add_argument("--tipo", default="f16", choices=list(util.BYTES_POR_VALOR))
    s.add_argument("--fa", default="on", choices=["on", "off"])
    a = p.parse_args()
    if a.orden == "modelo":
        tabla(util.metadatos(), util.MODELO)
    elif a.orden == "hf":
        for r in a.repos:
            desde_hf(r)
    else:
        util.comprobar_seguridad(a.n_ctx, a.tipo, a.fa == "on")


if __name__ == "__main__":
    main()
