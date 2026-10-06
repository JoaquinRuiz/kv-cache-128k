#!/usr/bin/env bash
# (Re)arranca `ollama serve` en condiciones controladas: un solo slot (OLLAMA_NUM_PARALLEL=1, porque Ollama
# multiplica el contexto por los slots), sin OLLAMA_CONTEXT_LENGTH y con el log en resultados/ollama_serve.log.
#
# Uso: scripts/servidor_ollama.sh            # por defecto (flash attention en modo auto, caché f16)
#      scripts/servidor_ollama.sh q8_0       # caché cuantizada: OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
#      scripts/servidor_ollama.sh parar
#
# macOS: cierra antes la app de Ollama (su deslizador de contexto le pasa OLLAMA_CONTEXT_LENGTH al servidor).
# Linux con systemd: para antes el servicio (sudo systemctl stop ollama).
set -u
source "$(dirname "$0")/comun.sh"

TIPO="${1:-}"
LOG="$RESULTADOS/ollama_serve.log"

if [ "$(uname)" = "Darwin" ] && pgrep -f "Ollama.app/Contents/MacOS/Ollama" >/dev/null; then
  echo "La app de Ollama está abierta: ciérrala (icono de la barra de menús › Quit Ollama) y repite." >&2
  exit 1
fi
pkill -f "ollama serve" 2>/dev/null && sleep 2
[ "$TIPO" = "parar" ] && exit 0

VARS="OLLAMA_NUM_PARALLEL=1"
if [ -n "$TIPO" ]; then
  VARS="$VARS OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=$TIPO"
fi
# shellcheck disable=SC2086
env -u OLLAMA_CONTEXT_LENGTH -u OLLAMA_FLASH_ATTENTION -u OLLAMA_KV_CACHE_TYPE $VARS \
  nohup ollama serve >>"$LOG" 2>&1 &

for _ in $(seq 1 60); do
  curl -s "${OLLAMA_URL:-http://localhost:11434}/api/version" >/dev/null && break
  sleep 0.5
done
grep "server config" "$LOG" | tail -1 \
  | grep -o -E "OLLAMA_(NUM_PARALLEL|CONTEXT_LENGTH|FLASH_ATTENTION|KV_CACHE_TYPE):[^ ]*" | tr '\n' ' '
echo
grep "default context" "$LOG" | tail -1 | sed -E 's/^.*msg=//'
echo "log: $LOG"
