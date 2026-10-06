# Funciones comunes para los scripts en bash (se carga con `source`). Compatible con bash 3.2 (macOS).

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODELO="${MODELO:-llama3.1:8b}"
export RESULTADOS="${RESULTADOS:-$RAIZ/resultados}"
mkdir -p "$RESULTADOS/logs"

# Ruta del GGUF: variable GGUF o la línea FROM de `ollama show --modelfile`.
ruta_gguf() {
  if [ -n "${GGUF:-}" ]; then echo "$GGUF"; return; fi
  local r
  r="$(ollama show "$MODELO" --modelfile 2>/dev/null | awk '/^FROM /{print $2; exit}')"
  if [ -z "$r" ] || [ ! -f "$r" ]; then
    echo "No encuentro el GGUF de $MODELO: instálalo con 'ollama pull $MODELO' o define GGUF=/ruta/al.gguf" >&2
    return 1
  fi
  echo "$r"
}

# Falla si Ollama tiene algún modelo cargado (no hay que tener Ollama y llama.cpp cargados a la vez).
exigir_ollama_vacio() {
  if ollama ps 2>/dev/null | tail -n +2 | grep -q .; then
    echo "Ollama tiene un modelo cargado. Descárgalo antes: ollama stop $MODELO" >&2
    exit 1
  fi
}

# Regla del 85 %: estima la memoria antes de cargar y se para si pasa del umbral (salvo FORZAR=1).
# Uso: seguridad N_CTX [f16|q8_0|q4_0] [on|off]
seguridad() {
  python3 "$RAIZ/scripts/kv.py" seguridad "$1" --tipo "${2:-f16}" --fa "${3:-on}" || exit 1
}

# Swap usado (MB) y páginas expulsadas acumuladas: "MB PAGINAS"
swap_ahora() {
  python3 -c "import sys; sys.path.insert(0, '$RAIZ/scripts'); import util; s, p = util.swap(); print(f'{s:.1f} {p}')"
}
