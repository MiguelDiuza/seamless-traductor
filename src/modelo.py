"""
Carga de los pesos preentrenados de SeamlessM4T v2 y utilidades de audio.

Los pesos (≈ 9 GB) se descargan una sola vez del repositorio oficial de Meta en
Hugging Face (facebook/seamless-m4t-v2-large) y quedan en la caché local.
Solo se instancia la parte voz -> texto (encoder de voz + decoder de texto),
no el generador de voz, para ahorrar memoria.
"""

from __future__ import annotations

import io
import time

import numpy as np
import torch

REPO = "facebook/seamless-m4t-v2-large"
FRECUENCIA = 16_000

# Idiomas de salida que ofrece la app (código de 3 letras que usa Seamless)
IDIOMAS = {
    "Inglés": "eng", "Español": "spa", "Francés": "fra", "Portugués": "por",
    "Alemán": "deu", "Italiano": "ita", "Japonés": "jpn", "Chino (mandarín)": "cmn",
}


def cargar_modelo(repo: str = REPO, dtype=torch.float32):
    """Procesador (banco de filtros + tokenizador) y modelo voz -> texto con pesos preentrenados.
    attn_implementation="eager": atención explícita (matmul + softmax) para poder leer sus pesos."""
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    procesador = AutoProcessor.from_pretrained(repo)
    modelo = SeamlessM4Tv2ForSpeechToText.from_pretrained(repo, attn_implementation="eager", dtype=dtype)
    return procesador, modelo.eval()


def cargar_audio(fuente) -> np.ndarray:
    """Lee un archivo (ruta o bytes) y lo deja en mono, float32, 16 kHz."""
    import librosa
    import soundfile as sf

    if isinstance(fuente, (bytes, bytearray)):
        fuente = io.BytesIO(fuente)
    try:
        audio, sr = sf.read(fuente, dtype="float32", always_2d=False)
    except Exception:
        if hasattr(fuente, "seek"):
            fuente.seek(0)
        audio, sr = librosa.load(fuente, sr=None, mono=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1) if audio.shape[1] < audio.shape[0] else audio.mean(axis=0)
    if sr != FRECUENCIA:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=FRECUENCIA)
    return audio.astype(np.float32)


def preprocesar(procesador, audio: np.ndarray):
    """Banco de filtros Mel de 80 canales cada 10 ms, normalizado y apilado de a 2
    frames -> (1, T, 160) con T ≈ 50 por segundo."""
    entrada = procesador(audio=audio, sampling_rate=FRECUENCIA, return_tensors="pt")
    return entrada.input_features, entrada.attention_mask


@torch.no_grad()
def traducir_oficial(procesador, modelo, audio: np.ndarray, idioma: str = "eng", max_new_tokens: int = 200):
    """Inferencia con la implementación oficial (generate con caché K/V), greedy.
    Devuelve (texto, segundos)."""
    x, mask = preprocesar(procesador, audio)
    t0 = time.perf_counter()
    ids = modelo.generate(input_features=x, attention_mask=mask, tgt_lang=idioma,
                          num_beams=1, do_sample=False, max_new_tokens=max_new_tokens)
    dt = time.perf_counter() - t0
    texto = procesador.decode(ids[0], skip_special_tokens=True).strip()
    return texto, dt
