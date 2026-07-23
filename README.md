# Reconstrucción de identidad vocal con TTS *zero-shot* y *fine-tuning* de XTTS v2

Proyecto final de Aprendizaje Profundo (MMIA). Estudia **cuánta identidad (timbre) y cuánto
contenido (inteligibilidad) sobreviven al ciclo** `voz → texto → voz`, y cuánto aporta un
*fine-tuning* liviano frente al uso *zero-shot* del modelo.

A partir de **una sola grabación real de reunión** (Zoom, español) se construye de forma
automática un corpus `(audio, texto)` y se re-sintetiza la voz con tres motores:

| Motor | Qué es |
|---|---|
| **XTTS v2** *zero-shot* | clona la voz desde una referencia, sin entrenar |
| **XTTS-FT** | igual, pero con el GPT afinado sobre el propio locutor |
| **Bark** | *baseline* que **no** clona (control de no-clonación) |

## Pipeline

1. **Preprocesamiento** — conversión a WAV mono 22 kHz, segmentación con Silero VAD,
   filtrado del locutor principal (embeddings ECAPA + *clustering* + filtro de solape) y
   transcripción con Whisper `large-v3`.
2. **Splits** — referencia de voz dedicada + `train` / `val` / `test` (por región, sin fuga).
3. **Fine-tuning liviano** — se afina solo el GPT de XTTS; DVAE, encoder de locutor y
   HiFi-GAN quedan congelados. El *checkpoint* final se elige por métricas, no por *loss*.
4. **Reconstrucción y evaluación** — cada texto de `test` se re-sintetiza y se compara con el
   original mediante cuatro métricas: **spk_sim** (similitud de locutor), **WER/CER**
   *round-trip*, **MCD** y **UTMOS** (naturalidad).

## Resultado principal

El *fine-tuning* mejora la identidad de forma consistente (spk_sim 0.80 → 0.86, MCD 10.4 →
8.6, ambos significativos por *bootstrap* pareado) a un coste moderado de inteligibilidad y
naturalidad, al adaptarse también al canal ruidoso de Zoom. Bark queda muy por debajo en
similitud, confirmando que la ganancia es clonación real y no un artefacto de la métrica.

## Contenido

**Código**

| Archivo | Descripción |
|---|---|
| `proy_final.ipynb` | Notebook con todo el *pipeline* (§1–§15): preprocesamiento, *fine-tuning*, reconstrucción, métricas, gráficos y audios. |

**Manifiestos del corpus** — se generan en §3–§6 y permiten reanudar el pipeline sin repetir
VAD/STT (celda §6b):

| Archivo | Descripción |
|---|---|
| `manifest_frags.csv` | Todos los fragmentos detectados por el VAD (`dur`, `region`, `trim_ratio`, `peak`). |
| `cluster_results_ecapa.csv` | Resultado del filtrado de locutor: `speaker` (clúster), `sim` al centroide, `win_min` (solape) e `is_main`. |
| `dryrun_ecapa.npz` | Cache de los *embeddings* ECAPA por fragmento (evita recalcularlos). |
| `reference.csv` | Clips que componen la referencia de voz (excluidos de los splits). |
| `train.csv` · `val.csv` · `test.csv` | Splits por región, ya con la transcripción (`text`, `n_words`, `wps`). |

**Resultados de la evaluación**

| Archivo | Descripción |
|---|---|
| `results.csv` | Una fila por fragmento y backend sobre `test`: métricas + texto original y re-transcrito (`hyp`). |
| `summary_by_backend.csv` | Agregado por backend: media, desviación y error estándar de cada métrica. |
| `results_ckpt_sweep.csv` | Barrido de *checkpoints* del fine-tuning, evaluado sobre `val`. |
| `summary_ckpt_sweep.csv` | Resumen del barrido, usado para elegir el *checkpoint* final. |

> **No versionado:** los `.wav` (grabación original, fragmentos y audios reconstruidos) y los
> pesos del *fine-tuning* quedan fuera del repositorio por tamaño; los CSV los referencian por
> ruta relativa bajo `tts_dataset/`. El artículo en LaTeX con el análisis completo se mantiene
> aparte.

## Ejecución

El notebook instala sus dependencias en la primera celda (`coqui-tts`, `faster-whisper`,
`silero-vad`, `resemblyzer`, `speechbrain`, …) y elige automáticamente la GPU más libre.
Requiere GPU para el *fine-tuning*. Se recomienda ejecutarlo de arriba abajo; con los CSV de
este repositorio y la celda §6b se puede saltar directamente a la parte de modelo.

## Ética

Los experimentos se realizaron bajo aprobación del locutor en cuestión. Que ~40 s de referencia basten para una
clonación convincente subraya el riesgo de suplantación: cualquier uso de estas técnicas
debe contar con el consentimiento explícito del locutor.
