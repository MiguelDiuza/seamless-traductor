"""
Prepara los datos de prueba a partir de FLEURS (Google), el benchmark que usa el
artículo de Seamless:
  - Audio en español (es_419, partición test) con su transcripción.
  - Para cada frase, la MISMA frase en inglés (FLEURS es paralelo: mismo `id`)
    como referencia de traducción.

Salida:
  ejemplos/*.wav + ejemplos/referencias.csv     -> 6 audios para la app
  resultados/fleurs_es_en.parquet               -> conjunto de evaluación (scripts/evaluar.py)
"""

import io
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download

RAIZ = Path(__file__).resolve().parents[1]


def leer_tsv(idioma: str) -> pd.DataFrame:
    ruta = hf_hub_download("google/fleurs", f"data/{idioma}/test.tsv", repo_type="dataset")
    cols = ["id", "archivo", "raw", "normalizado", "fonemas", "n_muestras", "genero"]
    return pd.read_csv(ruta, sep="\t", header=None, names=cols, quoting=3)


def main(n_eval: int = 150):
    # Audio en español (parquet convertido por Hugging Face)
    ruta = hf_hub_download("google/fleurs", "es_419/test/0000.parquet", repo_type="dataset",
                           revision="refs/convert/parquet")
    es = pd.read_parquet(ruta)
    en = leer_tsv("en_us").drop_duplicates("id").set_index("id")["raw"]
    es["referencia_en"] = es["id"].map(en)
    es = es.dropna(subset=["referencia_en"]).drop_duplicates("id")
    es["duracion_s"] = es["num_samples"] / 16000
    print(f"Frases es->en con referencia: {len(es)}")

    # Conjunto de evaluación: n frases al azar (semilla fija)
    ev = es.sample(n=min(n_eval, len(es)), random_state=0)
    (RAIZ / "resultados").mkdir(exist_ok=True)
    ev[["id", "audio", "raw_transcription", "transcription", "referencia_en", "duracion_s", "gender"]] \
        .to_parquet(RAIZ / "resultados" / "fleurs_es_en.parquet")

    # 6 ejemplos de distinta duración para la app
    (RAIZ / "ejemplos").mkdir(exist_ok=True)
    filas = []
    for objetivo in [4, 7, 10, 13, 17, 22]:
        f = es.iloc[(es["duracion_s"] - objetivo).abs().argsort().iloc[0]]
        audio, sr = sf.read(io.BytesIO(f["audio"]["bytes"]), dtype="float32")
        nombre = f"es_{f['duracion_s']:04.1f}s_{f['id']}.wav"
        sf.write(RAIZ / "ejemplos" / nombre, audio, sr)
        filas.append({"archivo": nombre, "duracion_s": round(f["duracion_s"], 2),
                      "transcripcion_es": f["raw_transcription"], "traduccion_en": f["referencia_en"]})
        print(nombre, "|", f["raw_transcription"][:60], "|", f["referencia_en"][:60])
    pd.DataFrame(filas).to_csv(RAIZ / "ejemplos" / "referencias.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
