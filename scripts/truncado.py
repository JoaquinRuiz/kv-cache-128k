#!/usr/bin/env python3
"""¿Trunca Ollama en silencio? La prueba de la contraseña.

Genera un documento con una contraseña inventada en el primer párrafo del mensaje de usuario, texto natural
sin repeticiones (el manual de bash) y al final: «¿Cuál es la contraseña del principio? Responde solo con ella.»
Lo manda por /api/chat sin num_ctx y con un num_ctx suficiente, y compara prompt_eval_count y respuesta.

  python3 scripts/truncado.py                       # ~11.000 tokens, sin num_ctx y con num_ctx 16384
  python3 scripts/truncado.py --palabras 9000 --num-ctx 16384 --repeticiones 3
  python3 scripts/truncado.py --solo sin            # solo la pasada sin num_ctx (o --solo con)

El documento queda en resultados/truncado/documento.txt. El aviso de truncado aparece en el log del servidor
(«truncating input prompt»), no en la respuesta de la API.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util  # noqa: E402

CONTRASENA = "ALBATROS-7319-NARANJA"


def documento(palabras):
    ruta = util.resultados("truncado", "documento.txt")
    texto = util.texto_manual(("bash",))[:palabras]
    doc = (f"Antes de nada, apunta esto: la contraseña secreta de hoy es {CONTRASENA}. Guárdala, te la preguntaré al final.\n\n"
           + " ".join(texto) + "\n\n¿Cuál es la contraseña del principio? Responde solo con ella.")
    with open(ruta, "w") as f:
        f.write(doc)
    return ruta, doc


def pasada(doc, num_ctx=None):
    cuerpo = {"model": util.MODELO, "stream": False, "messages": [{"role": "user", "content": doc}]}
    if num_ctx:
        cuerpo["options"] = {"num_ctx": num_ctx}
    r = util.api("/api/chat", cuerpo)
    return r.get("prompt_eval_count"), r["message"]["content"].strip(), r.get("done_reason")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--palabras", type=int, default=9000)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--repeticiones", type=int, default=1)
    p.add_argument("--solo", choices=["sin", "con"])
    p.add_argument("--json", action="store_true", help="una línea JSON por pasada")
    a = p.parse_args()
    ruta, doc = documento(a.palabras)
    pasadas = [("sin num_ctx", None), (f"num_ctx {a.num_ctx}", a.num_ctx)]
    if a.solo == "sin":
        pasadas = pasadas[:1]
    elif a.solo == "con":
        pasadas = pasadas[1:]
    if not a.json:
        print(f"documento: {ruta}")
    for etiqueta, n in pasadas:
        if n:
            util.comprobar_seguridad(n)
        for i in range(a.repeticiones):
            util.ollama_descargar()
            leidos, respuesta, motivo = pasada(doc, n)
            acierta = CONTRASENA in respuesta
            if a.json:
                print(json.dumps({"pasada": etiqueta, "rep": i + 1, "prompt_eval_count": leidos,
                                  "respuesta": respuesta, "acierta": acierta, "done_reason": motivo}, ensure_ascii=False))
            else:
                print(f"  {etiqueta:16} prompt_eval_count={leidos:>6}  acierta={'sí' if acierta else 'no'}  "
                      f"done_reason={motivo}  respuesta={respuesta[:60]!r}")
    util.ollama_descargar()
    log = util.log_ollama()
    if log and not a.json:
        avisos = [l.strip() for l in open(log, errors="replace") if "truncating input prompt" in l][-1:]
        print("\nlog del servidor: " + (avisos[0].split("msg=")[-1] if avisos else "(sin avisos de truncado)"))


if __name__ == "__main__":
    main()
