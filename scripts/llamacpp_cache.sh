#!/usr/bin/env bash
# Memoria que reserva llama.cpp para N tokens de contexto.
# Arranca llama-server con el GGUF, todas las capas en GPU, sin --fit, espera a que cargue,
# guarda el log en resultados/logs/ y lo para. Enseña la caché KV (K y V), el búfer de cómputo y los pesos.
#
# Uso: scripts/llamacpp_cache.sh N [on|off] [f16|q8_0|q4_0]      (on/off = flash attention)
set -u
source "$(dirname "$0")/comun.sh"

N="${1:?Uso: $0 N [on|off] [f16|q8_0|q4_0]}"
FA="${2:-on}"
TIPO="${3:-f16}"
PUERTO="${PUERTO:-18080}"
LOG="$RESULTADOS/logs/llamacpp_c${N}_fa${FA}_${TIPO}.log"

exigir_ollama_vacio
seguridad "$N" "$TIPO" "$FA"
G="$(ruta_gguf)" || exit 1

llama-server -m "$G" -c "$N" -ngl all -fa "$FA" -fit off -np 1 -ctk "$TIPO" -ctv "$TIPO" \
  --port "$PUERTO" -lv 4 >"$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 480); do
  grep -q "listening on" "$LOG" && break
  kill -0 "$PID" 2>/dev/null || { echo "llama-server ha terminado antes de cargar (mira $LOG)"; break; }
  sleep 0.25
done
kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

grep -E "llama_kv_cache: size|MTL0 KV buffer size|CUDA0 KV buffer size|compute buffer size|model buffer size" "$LOG" \
  | sed -E 's/^[0-9.]+ [IWE] //'
echo "log: $LOG"
