"""Genera notebooks/colab_seamless.ipynb: verificación, métricas en GPU y demo de respaldo."""
import json
from pathlib import Path

celdas = []
def md(t): celdas.append({"cell_type": "markdown", "metadata": {}, "source": t.strip()})
def code(t): celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip()})

md("""
# SeamlessM4T v2 en Colab: verificación, métricas y demo

**Artículo:** Seamless Communication *et al.*, "Seamless: Multilingual Expressive and Streaming Speech Translation", arXiv:2312.05187 (2023).
**Pesos:** `facebook/seamless-m4t-v2-large` · **Repositorio del proyecto:** https://github.com/MiguelDiuza/seamless-traductor

Usar con GPU: *Entorno de ejecución → Cambiar tipo de entorno → GPU T4*.
""")
code("""
!git clone -q https://github.com/MiguelDiuza/seamless-traductor.git
%cd seamless-traductor
!pip -q install transformers==4.57.6 sentencepiece sacrebleu jiwer soundfile librosa streamlit plotly
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no')
""")
md("## 1. Datos: FLEURS español con su traducción al inglés")
code("!python scripts/preparar_datos.py")
md("""
## 2. Verificación: nuestra implementación paso a paso = implementación oficial
Compara memoria del encoder, pesos de atención, logits y texto (transcripción y traducción).
""")
code("!python scripts/verificar_equivalencia.py")
md("## 3. Métricas: SeamlessM4T v2 vs. Whisper large-v3 (150 frases)")
code("!python scripts/evaluar.py --n 150")
code("""
import pandas as pd
from IPython.display import Image, display
display(pd.read_csv('resultados/metricas.csv').round(3))
display(Image('resultados/metricas.png')); display(Image('resultados/bleu_por_duracion.png'))
""")
md("Descargar los resultados para subirlos al repositorio:")
code("""
!zip -q -r resultados.zip resultados/metricas.csv resultados/predicciones.csv resultados/*.png
from google.colab import files; files.download('resultados.zip')
""")
md("""
## 4. (Respaldo) Abrir la app Streamlit desde Colab
Si el computador de la sustentación no tiene memoria suficiente, la app puede correr aquí y abrirse con un enlace público temporal.
""")
code("""
!wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared
!nohup streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
!sleep 5 && nohup ./cloudflared tunnel --url http://localhost:8501 > tunel.log 2>&1 &
!sleep 10 && grep -o 'https://[a-z0-9-]*\\.trycloudflare\\.com' tunel.log | head -1
""")

nb = {"cells": celdas, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
salida = Path(__file__).resolve().parents[1] / "notebooks" / "colab_seamless.ipynb"
salida.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok", salida)
