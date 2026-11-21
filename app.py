"""
App interactiva: SeamlessM4T v2 (Transformer encoder-decoder voz -> texto)
==========================================================================
Ejecutar:   streamlit run app.py

Carga un audio (ejemplo, archivo o micrófono), lo transcribe y lo traduce con los
pesos preentrenados de Meta, y muestra la inferencia por dentro: Q, K, V,
Q·Kᵀ/√d_k, la máscara causal, el softmax y la cross-attention audio↔texto.
"""

from __future__ import annotations

import hashlib
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from src.modelo import FRECUENCIA, IDIOMAS, cargar_audio, cargar_modelo, preprocesar, traducir_oficial  # noqa: E402
from src.seamless_paso_a_paso import SeamlessPasoAPaso  # noqa: E402

st.set_page_config(page_title="SeamlessM4T v2 · Transformer por dentro", page_icon="🌐", layout="wide")
torch.set_num_threads(max(1, torch.get_num_threads()))

CAPAS_ENCODER = [0, 11, 23]          # capas del Conformer cuyas atenciones se guardan
SEG_FRAME_ENC = 0.02                 # cada frame del encoder = 20 ms
SEG_FRAME_MEM = 0.16                 # cada vector de memoria (tras el adaptador ×8) = 160 ms


@st.cache_resource(show_spinner="Cargando pesos preentrenados (≈ 6 GB en RAM)…")
def obtener_modelo():
    proc, modelo = cargar_modelo()
    return proc, modelo, SeamlessPasoAPaso(modelo, proc.tokenizer)


@st.cache_resource(max_entries=4, show_spinner="Ejecutando la inferencia paso a paso…")
def inferir(clave: str, idioma_audio: str, idioma_destino: str, _audio: np.ndarray):
    proc, modelo, manual = obtener_modelo()
    x, _ = preprocesar(proc, _audio)
    t0 = time.perf_counter()
    codificado = manual.codificar(x[0], set(CAPAS_ENCODER))       # el encoder corre UNA vez
    t_enc = time.perf_counter() - t0
    transcripcion = manual.traducir(x[0], idioma_audio, codificado=codificado)
    traduccion = manual.traducir(x[0], idioma_destino, codificado=codificado) if idioma_destino != idioma_audio else None
    t_manual = time.perf_counter() - t0
    return x[0].numpy(), transcripcion, traduccion, t_enc, t_manual


@st.cache_data(max_entries=8, show_spinner="Ejecutando la implementación oficial para comparar…")
def inferir_oficial(clave: str, idioma: str, _audio: np.ndarray):
    proc, modelo, _ = obtener_modelo()
    return traducir_oficial(proc, modelo, _audio, idioma)


def etiqueta(tok: str) -> str:
    return tok.replace("▁", "␣").replace("<", "‹").replace(">", "›")


def normalizar(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]", " ", s.lower())).strip()


# ------------------------------------------------------------------ barra lateral --- #
st.sidebar.title("🌐 SeamlessM4T v2")
st.sidebar.caption("Transformer encoder–decoder · voz → texto en ~100 idiomas")
fuente = st.sidebar.radio("Fuente de audio", ["Ejemplos", "Subir archivo", "Micrófono"])
audio_bytes, nombre_audio = None, None
if fuente == "Ejemplos":
    ejemplos = sorted((RAIZ / "ejemplos").glob("*.wav"))
    if ejemplos:
        elegido = st.sidebar.selectbox("Audio de ejemplo (FLEURS, español)", ejemplos, format_func=lambda p: p.stem)
        audio_bytes, nombre_audio = elegido.read_bytes(), elegido.name
elif fuente == "Subir archivo":
    subido = st.sidebar.file_uploader("WAV / FLAC / MP3 / OGG", type=["wav", "flac", "mp3", "ogg", "m4a"])
    if subido:
        audio_bytes, nombre_audio = subido.getvalue(), subido.name
else:
    grabado = st.sidebar.audio_input("Graba una frase")
    if grabado:
        audio_bytes, nombre_audio = grabado.getvalue(), "micrófono"

idioma_audio = IDIOMAS[st.sidebar.selectbox("Idioma del audio (para transcribir)", list(IDIOMAS), index=1)]
idioma_destino = IDIOMAS[st.sidebar.selectbox("Traducir a", list(IDIOMAS), index=0)]
comparar = st.sidebar.checkbox("Comparar con la implementación oficial", value=False,
                               help="Ejecuta también model.generate() de transformers y compara textos y tiempos.")
st.sidebar.divider()
st.sidebar.markdown(
    "**Artículo base:** Seamless Communication *et al.*, *Seamless: Multilingual Expressive and "
    "Streaming Speech Translation*, arXiv:2312.05187, 2023.  \n"
    "[Repositorio oficial](https://github.com/facebookresearch/seamless_communication) · "
    "[Pesos](https://huggingface.co/facebook/seamless-m4t-v2-large)"
)
st.sidebar.markdown(
    "**Integrantes (Grupo 5):** Natalia Hernandez · Juan Ospina · Miguel Diuza · Yolima Alejandra Guadir"
)


@st.cache_resource
def inicio_despliegue() -> str:
    """Fecha y hora en que arrancó el servidor (se calcula una sola vez por proceso)."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


AVISO_DESPLIEGUE = (
    f"🟢 **Desplegado en local** (http://localhost:8502 · CPU · Python {platform.python_version()}) · "
    f"servidor iniciado: **{inicio_despliegue()}** · hora de esta ejecución: **{time.strftime('%Y-%m-%d %H:%M:%S')}**"
)

st.title("SeamlessM4T v2: el Transformer encoder–decoder por dentro")
st.success(AVISO_DESPLIEGUE)
if audio_bytes is None:
    st.info("Elige un audio en la barra lateral para comenzar.")
    st.stop()

audio = cargar_audio(audio_bytes)
if len(audio) / FRECUENCIA > 30:
    st.warning("Se usan los primeros 30 s del audio.")
    audio = audio[: 30 * FRECUENCIA]
duracion = len(audio) / FRECUENCIA
clave = hashlib.md5(audio.tobytes()).hexdigest()
feats, res_t, res_tr, t_enc, t_manual = inferir(clave, idioma_audio, idioma_destino, audio)
proc, modelo, _ = obtener_modelo()
cfg = modelo.config
resultados = {"Transcripción": res_t} | ({"Traducción": res_tr} if res_tr is not None else {})

tabs = st.tabs(["🎙️ Resultado", "🔗 Cross-attention", "🔢 Q · K · V paso a paso", "🧠 Encoder (Conformer)",
                "🎭 Decoder (máscara)", "🏗️ Arquitectura y dimensiones", "📊 Métricas"])

# ======================================================================= RESULTADO === #
with tabs[0]:
    st.audio(audio, sample_rate=FRECUENCIA)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duración", f"{duracion:.2f} s")
    c2.metric("Frames del encoder", f"{res_t.frames_encoder}", help="50 por segundo (20 ms cada uno)")
    c3.metric("Vectores de memoria", f"{res_t.frames_memoria}", help="Tras el adaptador de longitud (÷8): ≈ 6 por segundo")
    c4.metric("Tiempo (implementación propia)", f"{t_manual:.1f} s", help=f"Encoder: {t_enc:.1f} s")

    refs = RAIZ / "ejemplos" / "referencias.csv"
    fila = pd.read_csv(refs).query("archivo == @nombre_audio") if (fuente == "Ejemplos" and refs.exists()) else pd.DataFrame()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(f"Transcripción ({idioma_audio})")
        st.success(f"**{res_t.texto}**")
        if len(fila):
            import jiwer
            ref = fila.iloc[0]["transcripcion_es"]
            wer = jiwer.wer(normalizar(ref), normalizar(res_t.texto))
            st.caption(f"Referencia humana: {ref}  \n**WER de este audio: {100 * wer:.1f} %**")
    if res_tr is not None:
        with col_b:
            st.subheader(f"Traducción ({idioma_destino})")
            st.success(f"**{res_tr.texto}**")
            if len(fila) and idioma_destino == "eng":
                import sacrebleu
                ref = fila.iloc[0]["traduccion_en"]
                bleu = sacrebleu.sentence_bleu(res_tr.texto, [ref]).score
                st.caption(f"Traducción de referencia: {ref}  \n**BLEU de esta frase: {bleu:.1f}**")
    if comparar:
        for nombre, r, idioma in [("Transcripción", res_t, idioma_audio)] + \
                ([("Traducción", res_tr, idioma_destino)] if res_tr is not None else []):
            texto_hf, t_hf = inferir_oficial(clave, idioma, audio)
            st.caption(f"{nombre} oficial (`model.generate`, {t_hf:.1f} s): «{texto_hf}» → "
                       f"{'✔ idéntica' if texto_hf == r.texto else '✘ distinta'} a la implementación propia.")

    t = np.arange(len(audio)) / FRECUENCIA
    paso = max(1, len(audio) // 6000)
    fig = go.Figure(go.Scatter(x=t[::paso], y=audio[::paso], line=dict(width=1, color="#2f6fdf")))
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=30, b=10), title="Forma de onda", xaxis_title="segundos")
    st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure(go.Heatmap(z=feats[:, :80].T, x=np.arange(len(feats)) * SEG_FRAME_ENC, colorscale="Magma", showscale=False))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10), title="Entrada real del modelo: banco de filtros Mel (80 de las 160 dims por frame)",
                      xaxis_title="segundos", yaxis_title="filtro Mel")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Generación autoregresiva token a token")
    st.caption("El decoder empieza con ‹/s› + el token del idioma destino. En cada paso produce un softmax sobre "
               "256 102 tokens y se elige el más probable (greedy).")
    elegido = st.radio("Ver", list(resultados), horizontal=True, key="tabla_tokens")
    r = resultados[elegido]
    toks = [etiqueta(t) for t in proc.tokenizer.convert_ids_to_tokens(r.tokens)]
    tabla = pd.DataFrame({
        "paso": range(1, len(r.probabilidades) + 1),
        "entrada (lo ya escrito)": [" ".join(toks[: i + 2])[-60:] for i in range(len(r.probabilidades))],
        "token elegido": toks[2: 2 + len(r.probabilidades)],
        "probabilidad": r.probabilidades,
        "alternativas (top-5)": [", ".join(f"{etiqueta(a)} {p:.2f}" for a, p in alt) for alt in r.top_alternativas],
    })
    st.dataframe(tabla, use_container_width=True, hide_index=True,
                 column_config={"probabilidad": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})


def filas_decoder(r):
    """Etiquetas de las filas del decoder: la fila i predice el token i+1."""
    salida = [etiqueta(t) for t in proc.tokenizer.convert_ids_to_tokens(r.tokens[1:])]
    n = r.registros_decoder[0].pesos.shape[1]
    return [f"{i}:{t}" for i, t in enumerate(salida)][:n]


# ================================================================= CROSS-ATTENTION === #
with tabs[1]:
    st.markdown("""
Filas = tokens que escribe el decoder; columnas = vectores de memoria del audio (cada uno ≈ 160 ms).
En la cross-attention **Q sale del decoder** (lo ya escrito) y **K, V salen del encoder** (el audio).
- En la **transcripción** aparece una diagonal: el texto avanza al ritmo de la voz.
- En la **traducción** la diagonal se rompe donde el otro idioma **cambia el orden** de las palabras
  (p. ej. el adjetivo antes del sustantivo en inglés): el decoder salta a la parte del audio que necesita.
""")
    nl = cfg.decoder_layers
    c1, c2 = st.columns(2)
    capa = c1.slider("Capa del decoder", 0, nl - 1, nl // 2, key="cx_capa")
    cab = c2.selectbox("Cabeza", ["Promedio de cabezas"] + [f"Cabeza {h}" for h in range(cfg.decoder_attention_heads)], key="cx_cab")
    cols = st.columns(len(resultados))
    for col, (nombre, r) in zip(cols, resultados.items()):
        P = [x for x in r.registros_decoder if x.tipo == "cross"][capa].pesos
        M = (P.mean(0) if cab.startswith("Prom") else P[int(cab.split()[-1])]).numpy()
        filas = filas_decoder(r)
        fig = make_subplots(rows=2, cols=1, row_heights=[0.16, 0.84], shared_xaxes=True, vertical_spacing=0.02)
        paso = max(1, len(audio) // 3000)
        fig.add_trace(go.Scatter(x=t[::paso], y=audio[::paso], line=dict(width=1, color="#888"), showlegend=False), 1, 1)
        fig.add_trace(go.Heatmap(z=M, x=(np.arange(M.shape[1]) + 0.5) * SEG_FRAME_MEM, y=filas, colorscale="Viridis",
                                 showscale=False), 2, 1)
        fig.update_yaxes(autorange="reversed", row=2, col=1)
        fig.update_layout(height=max(420, 20 * len(filas) + 160), margin=dict(l=10, r=10, t=30, b=10),
                          title=nombre, xaxis2_title="tiempo del audio (s)")
        col.plotly_chart(fig, use_container_width=True)

# ======================================================================== Q K V ===== #
with tabs[2]:
    st.markdown("Recorre **una** atención con los números reales del modelo.")
    opciones = {"Cross-attention (decoder → audio)": "cross", "Self-attention enmascarada (decoder)": "decoder_self",
                "Self-attention del encoder (Conformer)": "encoder_self"}
    c1, c2, c3, c4 = st.columns(4)
    tipo = opciones[c1.selectbox("Tipo de atención", list(opciones))]
    nombre_res = c2.selectbox("Salida", list(resultados), key="qkv_res")
    r = resultados[nombre_res]
    regs = [x for x in r.registros_encoder if x.tipo == "encoder_self"] if tipo == "encoder_self" \
        else [x for x in r.registros_decoder if x.tipo == tipo]
    nombres_capa = [str(c) for c in CAPAS_ENCODER] if tipo == "encoder_self" else [str(i) for i in range(len(regs))]
    ic = nombres_capa.index(c3.select_slider("Capa", nombres_capa, value=nombres_capa[len(nombres_capa) // 2]))
    reg = regs[ic]
    H, Tq, dk = reg.Q.shape
    Tk = reg.K.shape[1]
    h = c4.slider("Cabeza", 0, H - 1, 0, key="qkv_cab")
    filas = filas_decoder(r)
    eq = [f"{i * SEG_FRAME_ENC:.2f}s" for i in range(Tq)] if tipo == "encoder_self" else filas
    ek = (filas if tipo == "decoder_self" else
          [f"{i * SEG_FRAME_ENC:.2f}s" for i in range(Tk)] if tipo == "encoder_self" else
          [f"{(i + .5) * SEG_FRAME_MEM:.2f}s" for i in range(Tk)])
    i = st.select_slider("Posición que pregunta (fila de Q)", options=list(range(Tq)), value=min(Tq - 1, 3),
                         format_func=lambda k: eq[k])
    origen_q = "frames del audio" if tipo == "encoder_self" else "decoder (texto ya escrito)"
    origen_kv = "decoder (texto ya escrito)" if tipo == "decoder_self" else (
        "frames del audio" if tipo == "encoder_self" else "memoria del encoder (audio)")
    st.markdown(f"""
| Tensor | Forma (esta cabeza) | De dónde sale | Rol |
|---|---|---|---|
| **Q** = X·W_Q | {Tq} × {dk} | {origen_q} | ¿Qué busco? |
| **K** = X·W_K | {Tk} × {dk} | {origen_kv} | ¿Qué información tengo? |
| **V** = X·W_V | {Tk} × {dk} | {origen_kv} | ¿Qué voy a comunicar? |
| **Q·Kᵀ/√d_k** | {Tq} × {Tk} | — | compatibilidad (√d_k = √{dk} = {np.sqrt(dk):.0f}){" + término de posición relativa" if tipo == "encoder_self" else ""} |
| **softmax** | {Tq} × {Tk} | — | cada fila suma 1 |
| **pesos·V** | {Tq} × {dk} | — | promedio ponderado = vector contextual |
""")
    crudos = (reg.Q[h, i] @ reg.K[h].T).numpy()
    escalados = reg.puntajes[h, i].numpy()
    pesos = reg.pesos[h, i].numpy()
    mask = escalados < -1e30
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(y=reg.Q[h, i].numpy(), marker_color="#2f6fdf"))
        fig.update_layout(height=230, title=f"① Vector q (fila {i}): {dk} números", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        fig = go.Figure(go.Heatmap(z=reg.K[h].numpy().T, colorscale="RdBu", zmid=0, x=ek))
        fig.update_layout(height=230, title=f"② Matriz K de la cabeza {h} ({Tk} × {dk})", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ek, y=np.where(mask, 0, crudos), name="q·kⱼ (crudo)", marker_color="#bbb"))
        fig.add_trace(go.Bar(x=ek, y=np.where(mask, 0, escalados), name="escalado (÷√d_k)", marker_color="#f08a24"))
        fig.update_layout(height=230, title="③ Puntajes", barmode="group", margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        fig = go.Figure(go.Bar(x=ek, y=pesos, marker_color="#2a9d55"))
        fig.update_layout(height=230, title=f"④ softmax → pesos (suman {pesos.sum():.3f})", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    if mask.any():
        st.info(f"🎭 {int(mask.sum())} posiciones futuras enmascaradas (−∞ antes del softmax → peso 0).")
    top = np.argsort(pesos)[::-1][:5]
    st.markdown("**⑤ Resultado:** salida = Σⱼ pesoⱼ · vⱼ. Más aportan: "
                + ", ".join(f"`{ek[j]}` ({pesos[j]:.2f})" for j in top))

# ======================================================================= ENCODER ==== #
with tabs[3]:
    st.markdown("""
Cada bloque **Conformer** combina: ½ FFN → **self-attention** (cada frame mira todo el audio, con **posición relativa**:
se suma un término q·r que depende de la distancia entre frames) → **convolución** (patrones locales de unos 0,6 s) → ½ FFN.
Aquí se ven las matrices frame × frame de algunas capas.
""")
    regs = [x for x in res_t.registros_encoder if x.tipo == "encoder_self"]
    c1, c2 = st.columns(2)
    nc = c1.select_slider("Capa del Conformer", [str(c) for c in CAPAS_ENCODER], key="enc_capa")
    P = regs[[str(c) for c in CAPAS_ENCODER].index(nc)].pesos
    cab = c2.selectbox("Cabeza", ["Promedio de cabezas"] + [f"Cabeza {h}" for h in range(P.shape[0])], key="enc_cab")
    M = (P.mean(0) if cab.startswith("Prom") else P[int(cab.split()[-1])]).numpy()
    ejes = np.arange(M.shape[0]) * SEG_FRAME_ENC
    fig = go.Figure(go.Heatmap(z=M, x=ejes, y=ejes, colorscale="Viridis"))
    fig.update_yaxes(autorange="reversed", title="frame que pregunta (Q) — s")
    fig.update_xaxes(title="frame consultado (K) — s")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**8 de las 16 cabezas de esta capa** (cada una aprende un patrón distinto):")
    cols = st.columns(4)
    for hh in range(8):
        f = go.Figure(go.Heatmap(z=P[hh].numpy(), colorscale="Viridis", showscale=False))
        f.update_yaxes(autorange="reversed", showticklabels=False)
        f.update_xaxes(showticklabels=False)
        f.update_layout(height=190, margin=dict(l=0, r=0, t=25, b=0), title=f"cabeza {hh}")
        cols[hh % 4].plotly_chart(f, use_container_width=True)

# ======================================================================= DECODER ==== #
with tabs[4]:
    st.markdown("""
Cada token solo mira **los anteriores** (triángulo inferior): la **máscara causal** pone −∞ sobre el futuro antes del softmax.
En entrenamiento el decoder recibe la frase completa; sin máscara copiaría la respuesta. En inferencia el futuro aún no existe.
""")
    c1, c2, c3 = st.columns(3)
    nombre_res = c1.selectbox("Salida", list(resultados), key="ds_res")
    r = resultados[nombre_res]
    dself = [x for x in r.registros_decoder if x.tipo == "decoder_self"]
    capa = c2.slider("Capa del decoder", 0, len(dself) - 1, 0, key="ds_capa")
    cab = c3.selectbox("Cabeza", ["Promedio de cabezas"] + [f"Cabeza {h}" for h in range(cfg.decoder_attention_heads)], key="ds_cab")
    P = dself[capa].pesos
    M = (P.mean(0) if cab.startswith("Prom") else P[int(cab.split()[-1])]).numpy()
    ent = [etiqueta(x) for x in proc.tokenizer.convert_ids_to_tokens(r.tokens)][: M.shape[0]]
    lab = [f"{i}:{x}" for i, x in enumerate(ent)]
    fig = go.Figure(go.Heatmap(z=M, x=lab, y=lab, colorscale="Viridis", texttemplate="%{z:.2f}" if len(ent) <= 22 else None))
    fig.update_yaxes(autorange="reversed", title="token que pregunta (Q)")
    fig.update_xaxes(title="token consultado (K)")
    fig.update_layout(height=max(400, 26 * len(ent) + 120), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ================================================================== ARQUITECTURA ==== #
with tabs[5]:
    Te, Tm = res_t.frames_encoder, res_t.frames_memoria
    D, H = cfg.hidden_size, cfg.speech_encoder_attention_heads
    st.markdown(f"""
### Recorrido de los tensores para ESTE audio ({duracion:.2f} s)

| Paso | Operación | Forma |
|---|---|---|
| Entrada | onda a 16 kHz | {len(audio):,} muestras |
| Banco de filtros | 80 filtros Mel cada 10 ms, normalizados, 2 frames apilados | {Te} × 160 |
| Proyección | LayerNorm + Linear 160 → {D} | {Te} × {D} |
| Encoder × {cfg.speech_encoder_layers} (Conformer) | ½FFN(swish, {cfg.speech_encoder_intermediate_size}) → Self-Attn ({H} cabezas, pos. relativa) → Conv (k={cfg.conv_depthwise_kernel_size}) → ½FFN → LN | {Te} × {D} |
| FFN intermedia | ½ FFN (ReLU) | {Te} × {D} |
| Adaptador de longitud | Conv1d k={cfg.adaptor_kernel_size}, stride {cfg.adaptor_stride} + GLU + Self-Attn + FFN | **{Tm} × {D}** |
| Memoria | LayerNorm → **K, V de la cross-attention** | {Tm} × {D} |
| Decoder entrada | ‹/s› + ‹idioma› + tokens → Embedding ({cfg.vocab_size:,} × {D}) × √{D} + posición sinusoidal | T × {D} |
| Decoder × {cfg.decoder_layers} | LN → Self-Attn **enmascarada** → LN → **Cross-Attn** (Q texto, K/V audio) → LN → FFN ReLU ({cfg.decoder_ffn_dim}) | T × {D} |
| Salida | LayerNorm → lm_head {D} → {cfg.vocab_size:,} → softmax | T × {cfg.vocab_size:,} |

d_k = {D} / {H} = **{D // H}** · posición relativa del encoder: distancias de −{cfg.left_max_position_embeddings} a +{cfg.right_max_position_embeddings} frames.
""")
    partes = {
        "Encoder de voz (Conformer + adaptador)": sum(p.numel() for p in modelo.speech_encoder.parameters()),
        "Decoder: self-attention": sum(p.numel() for n, p in modelo.text_decoder.named_parameters() if "self_attn." in n),
        "Decoder: cross-attention": sum(p.numel() for n, p in modelo.text_decoder.named_parameters() if "cross_attention." in n),
        "Decoder: FFN + normas": sum(p.numel() for n, p in modelo.text_decoder.named_parameters()
                                     if "layers" in n and "attn" not in n and "attention" not in n),
        "Embedding de tokens (compartido con lm_head)": modelo.text_decoder.embed_tokens.weight.numel(),
    }
    total = sum(p.numel() for p in modelo.parameters())
    df = pd.DataFrame({"bloque": list(partes), "parámetros": list(partes.values())})
    df["%"] = 100 * df["parámetros"] / total
    st.markdown(f"**Parámetros del modelo voz → texto: {total / 1e9:.2f} mil millones** (el modelo completo con síntesis de voz tiene 2,3)")
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"%": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})

# ======================================================================= MÉTRICAS === #
with tabs[6]:
    res_dir = RAIZ / "resultados"
    if (res_dir / "metricas.csv").exists():
        st.markdown("Resultados de `scripts/evaluar.py` sobre frases de FLEURS (español → inglés).")
        st.dataframe(pd.read_csv(res_dir / "metricas.csv").round(3), hide_index=True, use_container_width=True)
        for img in sorted(res_dir.glob("*.png")):
            st.image(str(img))
    else:
        st.info("Aún no hay métricas. Ejecuta `python scripts/evaluar.py` (recomendado en Colab con GPU).")
