"""
Métricas de desempeño (se recomienda ejecutarlo en Colab con GPU).

Conjunto: 150 frases aleatorias de FLEURS español (es_419, test) con su
transcripción y la misma frase en inglés (scripts/preparar_datos.py).

Tareas y métricas
  ASR  (voz español -> texto español): WER y CER (menor = mejor)
  S2TT (voz español -> texto inglés) : BLEU y chrF (mayor = mejor), sacrebleu
  Velocidad: RTF = tiempo de cómputo / duración del audio

Modelos
  SeamlessM4T v2 large (el del artículo) vs. Whisper large-v3 (OpenAI, referencia
  de la que parte el artículo). Ambos con decodificación greedy y las mismas frases.

Uso:  python scripts/evaluar.py [--n 150] [--solo-graficas]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "resultados"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def cargar_datos(n: int):
    df = pd.read_parquet(SALIDA / "fleurs_es_en.parquet").head(n)
    audios = [sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0] for a in df["audio"]]
    return df, audios


class Seamless:
    nombre = "SeamlessM4T v2 (2,3B)"

    def __init__(self):
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
        self.p = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.m = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large", dtype=DTYPE).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, audio, tarea):
        x = self.p(audio=audio, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        t0 = time.perf_counter()
        ids = self.m.generate(input_features=x.input_features.to(DTYPE), attention_mask=x.attention_mask,
                              tgt_lang="spa" if tarea == "asr" else "eng", num_beams=1, do_sample=False,
                              max_new_tokens=200)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return self.p.decode(ids[0], skip_special_tokens=True).strip(), time.perf_counter() - t0


class Whisper:
    nombre = "Whisper large-v3 (1,55B)"

    def __init__(self):
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        self.p = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        self.m = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", dtype=DTYPE).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, audio, tarea):
        x = self.p(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, DTYPE)
        t0 = time.perf_counter()
        ids = self.m.generate(x, language="es", task="transcribe" if tarea == "asr" else "translate",
                              num_beams=1, do_sample=False, max_new_tokens=200)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return self.p.batch_decode(ids, skip_special_tokens=True)[0].strip(), time.perf_counter() - t0


def evaluar(n: int):
    import jiwer
    import sacrebleu
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    norm = BasicTextNormalizer()   # minúsculas, sin puntuación ni tildes de diferencia: el mismo para ambos modelos
    df, audios = cargar_datos(n)
    filas, resumen = [], []
    for Clase in [Seamless, Whisper]:
        modelo = Clase()
        modelo(audios[0], "asr")   # calentamiento
        for tarea in ["asr", "s2tt"]:
            hip, tiempos = [], []
            for a in audios:
                texto, dt = modelo(a, tarea)
                hip.append(texto); tiempos.append(dt)
            dur = df["duracion_s"].to_numpy()
            if tarea == "asr":
                ref = [norm(r) for r in df["raw_transcription"]]
                hn = [norm(h) for h in hip]
                m = {"WER_%": 100 * jiwer.wer(ref, hn), "CER_%": 100 * jiwer.cer(ref, hn)}
            else:
                ref = list(df["referencia_en"])
                m = {"BLEU": sacrebleu.corpus_bleu(hip, [ref]).score, "chrF": sacrebleu.corpus_chrf(hip, [ref]).score}
            resumen.append({"modelo": modelo.nombre, "tarea": tarea, "n": len(hip), **m,
                            "RTF": float(np.sum(tiempos) / np.sum(dur)), "latencia_media_s": float(np.mean(tiempos))})
            for i, (h, t) in enumerate(zip(hip, tiempos)):
                filas.append({"modelo": modelo.nombre, "tarea": tarea, "id": df.iloc[i]["id"],
                              "duracion_s": dur[i], "tiempo_s": t, "hipotesis": h,
                              "referencia": df.iloc[i]["raw_transcription" if tarea == "asr" else "referencia_en"]})
            print(resumen[-1], flush=True)
        del modelo
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(filas).to_csv(SALIDA / "predicciones.csv", index=False)
    r = pd.DataFrame(resumen)
    r.to_csv(SALIDA / "metricas.csv", index=False)
    print(r.round(3).to_string(index=False))


def graficar():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sacrebleu

    r = pd.read_csv(SALIDA / "metricas.csv")
    colores = {"SeamlessM4T v2 (2,3B)": "#2c56cf", "Whisper large-v3 (1,55B)": "#d9822b"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (tarea, col, titulo) in zip(ax, [("asr", "WER_%", "Transcripción español: WER % (menor = mejor)"),
                                             ("s2tt", "BLEU", "Traducción es→en: BLEU (mayor = mejor)"),
                                             ("s2tt", "chrF", "Traducción es→en: chrF (mayor = mejor)")]):
        g = r[r.tarea == tarea]
        a.bar(g.modelo, g[col], color=[colores[m] for m in g.modelo])
        for j, v in enumerate(g[col]):
            a.text(j, v, f"{v:.1f}", ha="center", va="bottom")
        a.set_title(titulo, fontsize=10); a.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(SALIDA / "metricas.png", dpi=130)

    # BLEU por frase según duración del audio (la tabla se guarda en CSV para el repositorio)
    if (SALIDA / "predicciones.csv").exists():
        p = pd.read_csv(SALIDA / "predicciones.csv")
        p = p[p.tarea == "s2tt"].copy()
        p["bleu"] = [sacrebleu.sentence_bleu(str(h), [str(rf)]).score for h, rf in zip(p.hipotesis, p.referencia)]
        p["rango"] = pd.cut(p.duracion_s, [0, 8, 12, 16, 40], labels=["<8 s", "8-12 s", "12-16 s", ">16 s"])
        p.groupby(["rango", "modelo"], observed=True).bleu.mean().unstack().round(2).to_csv(SALIDA / "bleu_por_duracion.csv")
    tabla = pd.read_csv(SALIDA / "bleu_por_duracion.csv", index_col=0)
    ax = tabla.plot(kind="bar", color=[colores[c] for c in tabla.columns], figsize=(8, 4))
    ax.set_title("BLEU promedio por frase según duración del audio"); ax.set_ylabel("BLEU"); ax.set_xlabel("")
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(SALIDA / "bleu_por_duracion.png", dpi=130)
    print("Gráficas guardadas en", SALIDA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--solo-graficas", action="store_true")
    a = ap.parse_args()
    if not a.solo_graficas:
        evaluar(a.n)
    graficar()
