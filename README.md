# kv-cache-128k

Scripts del vídeo **«Tu modelo local dice 128K de contexto. ¿Los aguanta?»** del canal **@jokioki** (IA para desarrolladores).

Sirven para medir en tu propia máquina cuánto contexto aguanta de verdad un modelo local:

1. **La caché KV se calcula a mano.** Por cada token ocupa `2 × capas × cabezas KV × dimensión de cabeza × bytes por valor`. Para Llama 3.1 8B en FP16 son 128 KiB.
2. **llama.cpp y Ollama reservan exactamente eso** al subir el contexto.
3. **La generación se frena al llenar el contexto, y se puede predecir:** `tok/s ≈ ancho de banda efectivo / (bytes de pesos + bytes de caché)`.
4. **El contexto usable queda muy por debajo del anunciado.** Lo limitan la memoria, el swap y, sobre todo, la espera hasta la primera palabra. Cuantizar la caché (q8_0, q4_0) amplía lo que cabe, pero no lo que es usable.

> ⚠️ **Antes de empezar: algunas pruebas pueden mandar el equipo al swap.** Cargar contextos grandes reserva gigas de memoria de golpe. Los scripts tienen dos protecciones:
> - **La regla del 85 %.** Antes de cada carga calculan pesos + caché + búfer de cómputo. Si el total pasa del 85 % de la memoria que el sistema deja usar a la GPU, se paran. Puedes saltarte la regla con `FORZAR=1` bajo tu responsabilidad.
> - **Un vigilante de swap.** Durante las cargas largas miran el swap y cortan si crece.
>
> Aun así, cierra el navegador y las apps pesadas antes de las pruebas grandes, y no las lances en mitad de algo importante.

## Requisitos

- **Probado en macOS con Apple Silicon.** Es la única plataforma probada; en Linux o Windows puede haber que retocar cosas.
- [Ollama](https://ollama.com). Probado con la 0.35.0.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): `llama-server`, `llama-bench` y `llama-perplexity`. En macOS, `brew install llama.cpp`. Probado con la 0.5.0. Los flags de llama.cpp cambian entre versiones (por ejemplo, `-fa on`), así que compruébalos con `--help` si usas otra.
- Python 3.8 o superior (solo biblioteca estándar), bash, `curl` y `unzip`.
- Las páginas de manual de `zsh` y `bash` (`man zshall`, `man bash`). Los prompts largos se generan con ellas; si no las tienes, pasa un texto propio con `--texto`.
- El modelo: `ollama pull llama3.1:8b`. Los scripts usan **el mismo GGUF** en Ollama y en llama.cpp, y lo encuentran con `ollama show --modelfile`.

### Qué es específico de Mac

- **La memoria para la GPU.** En un Mac con memoria unificada, macOS solo deja usar a la GPU una parte de la RAM: la `recommendedMaxWorkingSetSize` de Metal. Los scripts la leen de `llama-server --list-devices`, que también la da con NVIDIA (CUDA0). Si no la detectan, defínela con `GPU_MIB`.
- **El swap.** En macOS se mide con `sysctl vm.swapusage` y `vm_stat`. La versión para Linux (`/proc/meminfo` y `/proc/vmstat`) está escrita pero **sin probar**.
- **La app de Ollama.** Su deslizador de contexto le pasa `OLLAMA_CONTEXT_LENGTH` al servidor. Para medir en limpio hay que cerrarla y usar `scripts/servidor_ollama.sh`. En Linux con systemd, para antes el servicio con `sudo systemctl stop ollama`.

## Configuración (variables de entorno)

| Variable | Por defecto | Para qué |
|---|---|---|
| `MODELO` | `llama3.1:8b` | Modelo de Ollama |
| `GGUF` | detectado con `ollama show --modelfile` | Ruta al GGUF para llama.cpp |
| `RESULTADOS` | `./resultados` | Dónde se guardan logs, CSV y descargas (fuera de git) |
| `GPU_MIB` | detectado con `llama-server --list-devices` | Memoria para la GPU en MiB |
| `UMBRAL` | `0.85` | Límite de la regla de seguridad |
| `FORZAR` | — | `1` para saltarse la regla del 85 % |
| `OLLAMA_URL` | `http://localhost:11434` | Servidor de Ollama |
| `OLLAMA_LOG` | detectado | Log del servidor de Ollama, para leer caché, capas y avisos de truncado |
| `PUERTO` | `18080` | Puerto para llama-server |

## Paso a paso

Hay dos reglas que se aplican a todo:
- **Nunca tengas Ollama y llama.cpp con un modelo cargado a la vez.** Los scripts de llama.cpp se niegan a arrancar si Ollama tiene uno cargado.
- **Descarga el modelo de Ollama entre medidas.** Los scripts ya lo hacen con `ollama stop`. El llama-server que lanza Ollama guarda en RAM el estado del último prompt (hasta 8 GiB), y eso solo ya puede mandar el equipo al swap.

**0. Servidor de Ollama controlado.** Un solo slot (`OLLAMA_NUM_PARALLEL=1`, porque Ollama multiplica el contexto por los slots), sin `OLLAMA_CONTEXT_LENGTH` y con el log en `resultados/`:
```sh
scripts/servidor_ollama.sh            # por defecto
scripts/servidor_ollama.sh parar
```

**1. La fórmula en papel.**
```sh
python3 scripts/kv.py modelo                          # el modelo de Ollama: KiB por token y total de 4K a 128K
python3 scripts/kv.py hf microsoft/Phi-3.5-mini-instruct unsloth/gemma-3-12b-it   # desde config.json (se descarga)
python3 scripts/kv.py seguridad 65536                 # ¿cabe 64K? (regla del 85 %)
```
`kv.py hf` avisa de dos trampas: usar `num_attention_heads` en vez de `num_key_value_heads`, y un `head_dim` distinto de `hidden_size / cabezas`. También cuenta solo las capas de atención completa cuando el modelo usa ventana deslizante. Si un repo de Hugging Face está restringido, usa un espejo público.

**2. Memoria exacta en llama.cpp.**
```sh
scripts/llamacpp_cache.sh 8192            # caché K y V, búfer de cómputo y pesos (con flash attention)
scripts/llamacpp_cache.sh 32768 off       # sin flash attention: el búfer de cómputo crece con el contexto
scripts/llamacpp_cache.sh 65536 on q8_0   # caché cuantizada
python3 scripts/fit_decision.py           # qué decide --fit sin -c ni -ngl (corta antes de cargar)
```

**3. Ollama: contexto por defecto y truncado.**
```sh
python3 scripts/ollama_carga.py 4096 16384 32768 65536   # size, size_vram, PROCESSOR, caché y capas
python3 scripts/ollama_carga.py 131072 --solo-decision    # cuántas capas dejaría en GPU, sin cargar
python3 scripts/truncado.py --repeticiones 3              # la prueba de la contraseña, sin y con num_ctx
```
Para Ollama, sin `num_ctx`, el contexto es el que él elige según la memoria de la GPU (con poca memoria, 4.096). Si el prompt no cabe, **lo trunca sin avisar en la respuesta**: se queda con `num_keep` tokens del principio y el final, hasta la mitad del contexto. El aviso solo aparece en el log del servidor (`truncating input prompt`).

**4. Velocidad.**
```sh
scripts/bench.sh fp16 -d 0,8192,32768                          # llama-bench por profundidad, 3 repeticiones
python3 scripts/prediccion.py resultados/bench_fp16.jsonl      # predicción frente a lo medido
python3 scripts/espera.py --num-ctx 20480 --tokens 15000       # segundos hasta la primera palabra (Ollama)
```

**5. Caché cuantizada.**
```sh
scripts/servidor_ollama.sh q8_0          # OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
python3 scripts/ollama_carga.py 65536
scripts/bench.sh q8 -d 32768 -ctk q8_0 -ctv q8_0
scripts/perplejidad.sh                   # descarga WikiText-2 y compara f16, q8_0 y q4_0
scripts/servidor_ollama.sh               # vuelve al servidor por defecto
```

Todo lo que se mide queda en `resultados/` (CSV, JSON y logs), que no entra en git.

## Scripts

| Script | Qué mide |
|---|---|
| `scripts/kv.py` | La fórmula de la caché: por modelo de Ollama o desde `config.json`, y la regla del 85 % |
| `scripts/llamacpp_cache.sh` | Caché KV, búfer de cómputo y pesos que reserva llama-server para N tokens |
| `scripts/fit_decision.py` | La decisión de `--fit` de llama.cpp (contexto y capas) cuando no le das `-c` ni `-ngl` |
| `scripts/servidor_ollama.sh` | Arranca `ollama serve` en condiciones controladas, con la caché en f16, q8_0 o q4_0 |
| `scripts/ollama_carga.py` | Carga con `num_ctx = N`: `/api/ps`, `ollama ps`, caché, capas y swap, con vigilante |
| `scripts/truncado.py` | La prueba de la contraseña: ¿lee Ollama todo el prompt? |
| `scripts/bench.sh` | llama-bench (prefill y generación) por profundidad, con vigilancia de swap |
| `scripts/prediccion.py` | Ancho de banda efectivo y generación predicha frente a la medida |
| `scripts/espera.py` | Segundos hasta la primera palabra y tok/s en Ollama: prefijo único, calentamiento, 3 repeticiones y mediana |
| `scripts/perplejidad.sh` | Perplejidad con la caché en f16, q8_0 y q4_0 (una señal, no un benchmark) |
| `scripts/util.py`, `scripts/comun.sh` | Funciones comunes: modelo y GGUF, memoria, swap, API de Ollama |

## Lo que aprendimos midiendo

- **Ollama ya activa flash attention en Apple Silicon por su cuenta.** El `OLLAMA_FLASH_ATTENTION:false` que aparece en su log significa «no forzada». Lo que no toca es el tipo de caché, que sigue en FP16.
- **Ollama no decide las capas en GPU por sí mismo.** En la 0.35 lanza su llama-server sin `-ngl`, y es el `--fit` de llama.cpp el que decide, dejando 1 GiB libre en la GPU. Ese margen no basta para que el equipo siga siendo usable.
- **El límite real lo puso el swap del sistema antes que la GPU.** Depende de lo que tengas abierto.
- **Cuantizar la caché no acelera la generación**, y el prefill puede ir algo más lento.

## Material de terceros

Nada de esto va en el repo. Se descarga o se genera al ejecutar:

- **WikiText-2** (Salesforce Research, licencia CC BY-SA), para `scripts/perplejidad.sh`. Se descarga del repo `ggml-org/ci` de Hugging Face.
- **`config.json` de Hugging Face**, para `scripts/kv.py hf`. Cada modelo tiene su propia licencia (por ejemplo, la Llama 3.1 Community License); consúltala en la página de cada repo.
- **Los textos de los prompts largos**, generados a partir de las páginas de manual instaladas en tu sistema (`man zshall`, `man bash`).

## About the author

**Joaquín Ruiz** — [jokiruiz.com](https://jokiruiz.com) ·
[youtube.com/@jokioki](https://youtube.com/@jokioki)

📗 [Del vibe coding al Spec-Driven Development](https://amzn.eu/d/02csLpKC)
📙 [El motor de la Inteligencia Artificial](https://amzn.eu/d/083CTN3U)
📘 [Programar con Inteligencia Artificial](https://amzn.eu/d/eK4f73N)
📙 [Explora la Inteligencia Artificial](https://amzn.eu/d/dSwYhue)

## Licencia

[MIT](LICENSE) © 2026 Joaquín Ruiz
