#!/usr/bin/env bash
# Velocidad según lo lleno que está el contexto, con llama-bench: prefill (pp512) y generación (tg128)
# a varias profundidades, con todas las capas en GPU, flash attention, calentamiento y 3 repeticiones.
# Vigila el swap cada segundo: marca la medida si crece más de 128 MB y la corta si pasa de 512 MB.
#
# Uso: scripts/bench.sh [ETIQUETA] [args extra de llama-bench]
#      scripts/bench.sh fp16 -d 0,8192,32768
#      scripts/bench.sh q8 -d 32768 -ctk q8_0 -ctv q8_0
# Salida: resultados/bench_ETIQUETA.jsonl y resultados/logs/bench_ETIQUETA_swap.csv
set -u
source "$(dirname "$0")/comun.sh"

ETIQUETA="${1:-fp16}"
[ $# -gt 0 ] && shift
[ $# -eq 0 ] && set -- -d 0,8192,32768

exigir_ollama_vacio
G="$(ruta_gguf)" || exit 1
SALIDA="$RESULTADOS/bench_${ETIQUETA}.jsonl"
VIGILA="$RESULTADOS/logs/bench_${ETIQUETA}_swap.csv"

read -r S0 P0 <<<"$(swap_ahora)"
echo "segundo,swap_MB,paginas_expulsadas" >"$VIGILA"
llama-bench -m "$G" -ngl 99 -fa on -p 512 -n 128 -r 3 -o jsonl --progress "$@" \
  >"$SALIDA" 2>"$RESULTADOS/logs/bench_${ETIQUETA}.log" &
PID=$!
SMAX="$S0"; ESTADO="ok"; T=0
while kill -0 "$PID" 2>/dev/null; do
  read -r S P <<<"$(swap_ahora)"
  echo "$T,$S,$P" >>"$VIGILA"
  SMAX="$(python3 -c "print(max($SMAX, $S))")"
  if python3 -c "import sys; sys.exit(0 if $S - $S0 > 512 else 1)"; then
    kill "$PID"; ESTADO="CORTADO: el swap ha crecido más de 512 MB"; break
  fi
  sleep 1; T=$((T + 1))
done
wait "$PID" 2>/dev/null
if [ "$ESTADO" = "ok" ] && python3 -c "import sys; sys.exit(0 if $SMAX - $S0 > 128 else 1)"; then
  ESTADO="MARCADO: el swap creció durante la medida (repítela con menos apps abiertas)"
fi

python3 - "$SALIDA" <<'PY'
import json, sys
filas = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"  {'profundidad':>11} {'prefill tok/s':>14} {'generación tok/s':>17}")
for d in sorted({f["n_depth"] for f in filas}):
    pp = [f for f in filas if f["n_depth"] == d and f["n_prompt"] > 0]
    tg = [f for f in filas if f["n_depth"] == d and f["n_gen"] > 0]
    med = lambda f: sorted(f[0]["samples_ts"])[len(f[0]["samples_ts"]) // 2] if f else float("nan")
    print(f"  {d:>11} {med(pp):>14.1f} {med(tg):>17.2f}")
PY
echo "  $ESTADO · swap $S0 → máx. $SMAX MB · $SALIDA"
