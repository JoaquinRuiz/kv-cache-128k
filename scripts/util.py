"""Utilidades comunes: modelo y GGUF, memoria de la GPU, swap, metadatos del modelo y API de Ollama.

Solo biblioteca estándar de Python 3.8+. Probado en macOS con Apple Silicon; las partes de Linux
(/proc/meminfo, /proc/vmstat) están escritas pero sin probar.
"""
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS = os.environ.get("RESULTADOS", os.path.join(RAIZ, "resultados"))
MODELO = os.environ.get("MODELO", "llama3.1:8b")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ES_MAC = platform.system() == "Darwin"
UMBRAL = float(os.environ.get("UMBRAL", "0.85"))  # regla de seguridad: no pasar del 85 % de la memoria para la GPU

# bytes por valor de la caché según su tipo (llama.cpp: q8_0 = 34 bytes / 32 valores, q4_0 = 18 / 32)
BYTES_POR_VALOR = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}


def resultados(*partes):
    """Ruta dentro de resultados/ (se crea si no existe)."""
    ruta = os.path.join(RESULTADOS, *partes)
    os.makedirs(os.path.dirname(ruta) if os.path.splitext(ruta)[1] else ruta, exist_ok=True)
    return ruta


def ejecutar(cmd, **kw):
    env = dict(os.environ, LC_ALL="C")
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


# --- modelo y GGUF -----------------------------------------------------------------------------

def gguf(modelo=MODELO):
    """Ruta del GGUF: variable GGUF o la línea FROM de `ollama show --modelfile`."""
    if os.environ.get("GGUF"):
        return os.environ["GGUF"]
    r = ejecutar(["ollama", "show", modelo, "--modelfile"])
    for linea in r.stdout.splitlines():
        if linea.startswith("FROM ") and os.path.exists(linea[5:].strip()):
            return linea[5:].strip()
    sys.exit(f"No encuentro el GGUF de {modelo}. Instálalo con `ollama pull {modelo}` o define GGUF=/ruta/al/modelo.gguf")


def metadatos(modelo=MODELO):
    """Capas, cabezas y dimensión de cabeza desde `ollama show --verbose` (sobrescribibles por variables)."""
    r = ejecutar(["ollama", "show", modelo, "--verbose"])
    kv = {}
    for linea in r.stdout.splitlines():
        m = re.match(r"\s+([a-z0-9_.]+)\s+(\S+)\s*$", linea)
        if m:
            kv[m[1]] = m[2]

    def buscar(sufijo):
        for k, v in kv.items():
            if k.endswith(sufijo) and re.fullmatch(r"\d+", v):
                return int(v)
        return None

    capas = int(os.environ.get("CAPAS") or buscar(".block_count") or 0)
    cabezas = int(os.environ.get("CABEZAS") or buscar(".attention.head_count") or 0)
    cabezas_kv = int(os.environ.get("CABEZAS_KV") or buscar(".attention.head_count_kv") or cabezas)
    dim = int(os.environ.get("DIM_CABEZA") or buscar(".attention.key_length") or 0)
    if not dim and cabezas:
        dim = (buscar(".embedding_length") or 0) // cabezas
    if not (capas and cabezas_kv and dim):
        sys.exit("No he podido leer capas / cabezas KV / dimensión de cabeza. Defínelas con CAPAS, CABEZAS_KV y DIM_CABEZA.")
    return {"capas": capas, "cabezas": cabezas, "cabezas_kv": cabezas_kv, "dim_cabeza": dim,
            "contexto_max": buscar(".context_length")}


def kv_bytes_por_token(md, tipo="f16"):
    """2 × capas × cabezas KV × dimensión de cabeza × bytes por valor."""
    return 2 * md["capas"] * md["cabezas_kv"] * md["dim_cabeza"] * BYTES_POR_VALOR[tipo]


# --- memoria ------------------------------------------------------------------------------------

def memoria_gpu_mib():
    """Memoria que el sistema deja usar a la GPU, según `llama-server --list-devices`.
    En un Mac es la recommendedMaxWorkingSetSize de Metal (MTL0); con NVIDIA, la VRAM (CUDA0).
    Sobrescribible con GPU_MIB."""
    if os.environ.get("GPU_MIB"):
        return float(os.environ["GPU_MIB"])
    try:
        r = ejecutar(["llama-server", "--list-devices"], timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for linea in (r.stdout + r.stderr).splitlines():
        m = re.match(r"\s*(\w+): .*\((\d+) MiB, \d+ MiB free\)", linea)
        if m and m[1] not in ("BLAS", "CPU") and int(m[2]) > 0:
            return float(m[2])
    return None


def swap():
    """(swap usado en MB, páginas expulsadas acumuladas). macOS: vm.swapusage + vm_stat; Linux: /proc."""
    if ES_MAC:
        s = ejecutar(["sysctl", "-n", "vm.swapusage"]).stdout
        usado = float(re.search(r"used = ([\d.,]+)M", s)[1].replace(",", "."))
        v = ejecutar(["vm_stat"]).stdout
        expulsadas = int(re.search(r"Pageouts:\s+(\d+)", v)[1])
        return usado, expulsadas
    mi = dict(re.findall(r"(\w+):\s+(\d+)", open("/proc/meminfo").read()))
    usado = (int(mi.get("SwapTotal", 0)) - int(mi.get("SwapFree", 0))) / 1024
    vs = dict(re.findall(r"(\w+) (\d+)", open("/proc/vmstat").read()))
    return usado, int(vs.get("pswpout", 0))


def estimar_mib(n_ctx, tipo="f16", fa=True, modelo=MODELO):
    """Estimación para la regla de seguridad: pesos (tamaño del GGUF) + caché + búfer de cómputo.
    Búfer con flash attention: ~0,006 MiB por token (mínimo 256). Sin ella: n_ctx × 512 × cabezas × 4 bytes."""
    md = metadatos(modelo)
    pesos = os.path.getsize(gguf(modelo)) / 2**20
    cache = n_ctx * kv_bytes_por_token(md, tipo) / 2**20
    computo = max(256.0, n_ctx * 0.006) if fa else n_ctx * 512 * md["cabezas"] * 4 / 2**20
    return {"pesos": pesos, "cache": cache, "computo": computo, "total": pesos + cache + computo}


def comprobar_seguridad(n_ctx, tipo="f16", fa=True, modelo=MODELO):
    """Calcula antes de cargar. Si pasa del umbral, se para salvo FORZAR=1. Devuelve la estimación."""
    est = estimar_mib(n_ctx, tipo, fa, modelo)
    gpu = memoria_gpu_mib()
    if gpu is None:
        print(f"  (no conozco la memoria de la GPU; define GPU_MIB) estimado {es(est['total'], 0)} MiB", file=sys.stderr)
        return est
    pct = est["total"] / gpu
    print(f"  seguridad: {es(n_ctx, 0)} tokens ({tipo}, FA {'sí' if fa else 'no'}) ≈ {es(est['total'], 0)} MiB "
          f"= {es(pct * 100, 0)} % de {es(gpu, 0)} MiB para la GPU", file=sys.stderr)
    if pct > UMBRAL and os.environ.get("FORZAR") != "1":
        sys.exit(f"  ⚠️  Pasa del {es(UMBRAL * 100, 0)} %. Cierra apps y repite con FORZAR=1 si quieres arriesgarte (vigila el swap).")
    return est


# --- Ollama -------------------------------------------------------------------------------------

def api(ruta, cuerpo=None, timeout=1800):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(OLLAMA + ruta, datos, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ollama_descargar(modelo=MODELO):
    ejecutar(["ollama", "stop", modelo])


def ollama_cargado():
    try:
        return [m["name"] for m in api("/api/ps", timeout=10).get("models", [])]
    except OSError:
        return []


def log_ollama():
    """Log del servidor de Ollama: OLLAMA_LOG, el de la app de macOS si está abierta, o resultados/ollama_serve.log."""
    if os.environ.get("OLLAMA_LOG"):
        return os.environ["OLLAMA_LOG"]
    if ES_MAC and ejecutar(["pgrep", "-f", "Ollama.app/Contents/Resources/ollama serve"]).returncode == 0:
        return os.path.expanduser("~/.ollama/logs/server.log")
    ruta = os.path.join(RESULTADOS, "ollama_serve.log")
    return ruta if os.path.exists(ruta) else None


def matar_runner_ollama():
    """Para el llama-server que lanza Ollama (lleva --no-webui --offline en su línea de comandos)."""
    ejecutar(["pkill", "-9", "-f", "llama-server.*--no-webui"])  # SIGKILL: que no termine de reservar la caché


def texto_manual(paginas=("zshall", "bash")):
    """Texto natural y sin líneas repetidas a partir de páginas de man locales (para prompts largos)."""
    lineas, vistas = [], set()
    for p in paginas:
        r = subprocess.run(f"man {p} 2>/dev/null | col -b", shell=True, capture_output=True, text=True)
        for l in r.stdout.splitlines():
            l = l.strip()
            if l and l not in vistas and not re.fullmatch(r"[A-Z ()0-9.\-]+", l):
                vistas.add(l)
                lineas.append(l)
    palabras = " ".join(lineas).split()
    if not palabras:
        sys.exit("No hay páginas de man disponibles (man zshall / man bash). Pasa un texto con --texto FICHERO.")
    return palabras


def es(v, dec=1):
    """Número en formato español: coma decimal y punto de miles."""
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
