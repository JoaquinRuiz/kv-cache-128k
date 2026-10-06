#!/usr/bin/env bash
# Perplejidad con la caché en f16, q8_0 y q4_0 (una señal de calidad, no un benchmark).
# Descarga WikiText-2 (raw) la primera vez en resultados/wikitext/ (4,7 MB; Salesforce Research, licencia CC BY-SA)
# y ejecuta llama-perplexity con 4.096 de contexto sobre 20 bloques del conjunto de test.
#
# Uso: scripts/perplejidad.sh [CHUNKS=20] [CTX=4096]
set -u
source "$(dirname "$0")/comun.sh"

CHUNKS="${1:-20}"
CTX="${2:-4096}"
DIR="$RESULTADOS/wikitext"
TEST="$DIR/wikitext-2-raw/wiki.test.raw"
URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

exigir_ollama_vacio
G="$(ruta_gguf)" || exit 1
if [ ! -f "$TEST" ]; then
  mkdir -p "$DIR"
  echo "Descargando WikiText-2 (4,7 MB, licencia CC BY-SA) de $URL"
  curl -fsSL -o "$DIR/wikitext-2-raw-v1.zip" "$URL" && unzip -o -q "$DIR/wikitext-2-raw-v1.zip" -d "$DIR" || exit 1
fi
seguridad "$CTX" f16 on

for T in f16 q8_0 q4_0; do
  LOG="$RESULTADOS/logs/perplejidad_$T.log"
  llama-perplexity -m "$G" -f "$TEST" -c "$CTX" --chunks "$CHUNKS" -ngl all -fa on -fit off \
    -ctk "$T" -ctv "$T" >"$LOG" 2>&1
  printf "  %-5s %s\n" "$T" "$(grep -E "Final estimate" "$LOG" | sed -E 's/^.*Final estimate: //')"
done
