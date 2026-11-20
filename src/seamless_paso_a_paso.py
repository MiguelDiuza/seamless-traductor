"""
SeamlessM4T v2 "paso a paso": reimplementación manual de la inferencia voz -> texto
==================================================================================

No se entrena nada: se toman los pesos preentrenados oficiales (Meta,
facebook/seamless-m4t-v2-large) y se vuelve a escribir, operación por operación,
el modelo X2T (voz -> texto) descrito en la sección 3.2.2 del artículo:

  audio 16 kHz ─► banco de filtros Mel (80) apilado ×2 = 160 por frame (50 frames/s)
     ─► proyección 160 -> 1024
     ─► ENCODER DE VOZ w2v-BERT 2.0: 24 bloques Conformer
            ½FFN → self-attention (posición relativa) → convolución → ½FFN → LN
     ─► ½FFN intermedia ─► ADAPTADOR DE LONGITUD (conv stride 8 + self-attention)
     ─► memoria: ≈ 6 vectores por segundo de audio (1024 dims)  ──────── K, V
                                                                          │
  </s> + <idioma destino> + tokens ─► embedding×32 + posición sinusoidal  │
     ─► DECODER DE TEXTO (NLLB): 24 bloques                               ▼
            self-attention ENMASCARADA → CROSS-attention (Q texto; K,V audio) → FFN
     ─► LayerNorm ─► lm_head (1024 -> 256 102 tokens) ─► softmax ─► siguiente token

¿Por qué hacerlo a mano? Para poder VER cada tensor (Q, K, V, Q·Kᵀ/√d_k, la
máscara, el softmax, pesos·V) y para demostrar que entendemos cada fórmula:
scripts/verificar_equivalencia.py compara contra la implementación oficial.

Notación: T = longitud, D = 1024, H = 16 cabezas, d_k = D/H = 64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

EPS = 1e-5


@dataclass
class RegistroAtencion:
    """Lo que ocurre dentro de UNA capa de atención (para visualizarlo)."""

    tipo: str               # "encoder_self" | "adaptador" | "decoder_self" | "cross"
    capa: int
    Q: torch.Tensor         # (H, T_q, d_k)  "¿qué busco?"
    K: torch.Tensor         # (H, T_k, d_k)  "¿qué información tengo?"
    V: torch.Tensor         # (H, T_k, d_k)  "¿qué voy a comunicar?"
    puntajes: torch.Tensor  # (H, T_q, T_k)  Q·Kᵀ/√d_k (+ posición relativa) (+ máscara)
    pesos: torch.Tensor     # (H, T_q, T_k)  softmax por filas


@dataclass
class ResultadoInferencia:
    tokens: list[int]
    texto: str
    probabilidades: list[float]
    top_alternativas: list[list[tuple[str, float]]]
    frames_encoder: int          # frames que entran al Conformer (≈ 50/s)
    frames_memoria: int          # vectores que salen del adaptador (≈ 6/s)
    registros_encoder: list[RegistroAtencion] = field(default_factory=list)
    registros_decoder: list[RegistroAtencion] = field(default_factory=list)
    memoria: torch.Tensor | None = None
    logits_finales: torch.Tensor | None = None


# --------------------------------------------------------------------------- #
def ln(x, modulo):
    """LayerNorm: cada vector a media 0 y varianza 1, luego escala y sesgo aprendidos."""
    return F.layer_norm(x, (x.shape[-1],), modulo.weight, modulo.bias, EPS)


def ffn(x, modulo, activacion):
    """Red feed-forward por posición: Linear -> activación -> Linear."""
    return F.linear(activacion(F.linear(x, modulo.intermediate_dense.weight, modulo.intermediate_dense.bias)),
                    modulo.output_dense.weight, modulo.output_dense.bias)


def atencion(x_q, x_kv, w_q, w_k, w_v, w_o, H, tipo, capa,
             causal=False, dist_emb=None, izq=64, der=8):
    """Multi-Head Scaled Dot-Product Attention escrita a mano.

    x_q: (T_q, D) de donde salen las consultas; x_kv: (T_k, D) de donde salen K y V.
    w_*: capas lineales (weight, bias). dist_emb: tabla de posición relativa (encoder).
    """
    T_q, D = x_q.shape
    T_k = x_kv.shape[0]
    d_k = D // H

    # 1) Proyecciones: el mismo vector en tres roles (matrices aprendidas)
    Q = F.linear(x_q, w_q.weight, w_q.bias).view(T_q, H, d_k).transpose(0, 1)
    K = F.linear(x_kv, w_k.weight, w_k.bias).view(T_k, H, d_k).transpose(0, 1)
    V = F.linear(x_kv, w_v.weight, w_v.bias).view(T_k, H, d_k).transpose(0, 1)

    # 2) Q·Kᵀ / √d_k : compatibilidad de cada consulta con cada clave
    puntajes = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

    # 3) Posición RELATIVA (solo encoder Conformer): se suma q·r_(j-i), donde r es un
    #    vector aprendido para cada distancia j-i (recortada a [-64, +8]). Así la
    #    atención sabe "qué tan lejos" está cada frame, sin una tabla absoluta.
    if dist_emb is not None:
        i = torch.arange(T_q)[:, None]
        j = torch.arange(T_k)[None, :]
        dist = torch.clamp(j - i, -izq, der) + izq                 # (T_q, T_k) índices
        R = dist_emb.weight[dist]                                  # (T_q, T_k, d_k)
        puntajes = puntajes + torch.einsum("hld,lrd->hlr", Q, R) / math.sqrt(d_k)

    # 4) Máscara causal (decoder): el token i no ve los tokens j > i
    if causal:
        futuro = torch.triu(torch.ones(T_q, T_k, dtype=torch.bool), diagonal=1)
        puntajes = puntajes.masked_fill(futuro, torch.finfo(puntajes.dtype).min)

    # 5) Softmax por filas -> pesos que suman 1;  6) pesos·V -> promedio ponderado
    pesos = torch.softmax(puntajes.float(), dim=-1).to(Q.dtype)
    z = (pesos @ V).transpose(0, 1).reshape(T_q, D)               # concatenar cabezas
    salida = F.linear(z, w_o.weight, w_o.bias)                    # mezclar con W_O
    return salida, RegistroAtencion(tipo, capa, Q, K, V, puntajes, pesos)


# --------------------------------------------------------------------------- #
class SeamlessPasoAPaso:
    def __init__(self, modelo_hf, tokenizer):
        self.m = modelo_hf.eval()
        self.tok = tokenizer
        self.c = modelo_hf.config
        self.H_enc = self.c.speech_encoder_attention_heads
        self.H_dec = self.c.decoder_attention_heads
        self.lang_ids = modelo_hf.generation_config.text_decoder_lang_to_code_id

    # ------------------------------ ENCODER DE VOZ ------------------------------ #
    @torch.no_grad()
    def codificar(self, features: torch.Tensor, guardar_capas=None):
        """features: (T, 160) del procesador -> memoria (T_mem, 1024).
        guardar_capas: conjunto de capas Conformer cuyas atenciones se guardan (None = todas)."""
        se = self.m.speech_encoder
        fp = se.feature_projection
        h = F.linear(ln(features, fp.layer_norm), fp.projection.weight, fp.projection.bias)   # (T, 1024)
        registros = []

        for n, capa in enumerate(se.encoder.layers):
            # (a) Medio FFN ("macaron"): la mitad antes y la mitad después de la atención
            h = h + 0.5 * ffn(ln(h, capa.ffn1_layer_norm), capa.ffn1, F.silu)
            # (b) Self-attention con posición relativa: cada frame mira a todos los demás
            sa = capa.self_attn
            a, reg = atencion(ln(h, capa.self_attn_layer_norm), ln(h, capa.self_attn_layer_norm),
                              sa.linear_q, sa.linear_k, sa.linear_v, sa.linear_out, self.H_enc,
                              "encoder_self", n, dist_emb=sa.distance_embedding,
                              izq=sa.left_max_position_embeddings, der=sa.right_max_position_embeddings)
            h = h + a
            # (c) Módulo convolucional: patrones LOCALES (formantes, transiciones de fonemas)
            cm = capa.conv_module
            c = ln(h, cm.layer_norm).T[None]                                   # (1, D, T)
            c = F.glu(F.conv1d(c, cm.pointwise_conv1.weight), dim=1)            # compuerta GLU
            c = F.pad(c, (cm.depthwise_conv.kernel_size[0] - 1, 0))             # causal: solo pasado
            c = F.conv1d(c, cm.depthwise_conv.weight, groups=c.shape[1])        # conv por canal, k=31
            c = F.silu(ln(c[0].T, cm.depthwise_layer_norm)).T[None]
            c = F.conv1d(c, cm.pointwise_conv2.weight)[0].T
            h = h + c
            # (d) Segundo medio FFN y LayerNorm final del bloque
            h = h + 0.5 * ffn(ln(h, capa.ffn2_layer_norm), capa.ffn2, F.silu)
            h = ln(h, capa.final_layer_norm)
            if guardar_capas is None or n in guardar_capas:
                registros.append(reg)
        h = ln(h, se.encoder.layer_norm)
        frames_encoder = h.shape[0]

        # FFN intermedia (ReLU) con peso 0.5
        h = h + 0.5 * ffn(h, se.intermediate_ffn, F.relu)

        # ADAPTADOR DE LONGITUD: conv de stride 8 -> ~8 veces menos vectores.
        # El texto es mucho más corto que el audio; acortar la memoria abarata la
        # cross-attention y acerca las dos longitudes.
        for capa in se.adapter.layers:
            k, s = capa.kernel_size, capa.stride
            res = F.glu(F.conv1d(ln(h, capa.residual_layer_norm).T[None], capa.residual_conv.weight,
                                 capa.residual_conv.bias, stride=s, padding=s // 2), dim=1)[0].T
            x = F.glu(F.conv1d(ln(h, capa.self_attn_layer_norm).T[None], capa.self_attn_conv.weight,
                               capa.self_attn_conv.bias, stride=s, padding=s // 2), dim=1)[0].T
            sa = capa.self_attn
            a, reg = atencion(x, x, sa.linear_q, sa.linear_k, sa.linear_v, sa.linear_out, self.H_enc,
                              "adaptador", 0)
            h = a + res
            h = h + ffn(ln(h, capa.ffn_layer_norm), capa.ffn, F.relu)
            registros.append(reg)
        memoria = ln(h, se.inner_layer_norm)
        return memoria, registros, frames_encoder

    # ------------------------------ DECODER DE TEXTO ----------------------------- #
    def posiciones(self, T: int) -> torch.Tensor:
        """Codificación posicional sinusoidal (posiciones 1..T, el 0 es el padding)."""
        D = self.c.hidden_size
        mitad = D // 2
        frec = torch.exp(torch.arange(mitad).float() * -(math.log(10000) / (mitad - 1)))
        ang = torch.arange(1, T + 1).float()[:, None] * frec[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=1)

    @torch.no_grad()
    def decodificar(self, tokens: list[int], memoria: torch.Tensor, guardar=True, solo_ultimo=False):
        dec = self.m.text_decoder
        escala = math.sqrt(self.c.hidden_size) if self.c.scale_embedding else 1.0
        h = F.embedding(torch.tensor(tokens), dec.embed_tokens.weight) * escala + self.posiciones(len(tokens))
        registros = []
        for n, capa in enumerate(dec.layers):
            sa, ca = capa.self_attn, capa.cross_attention
            # (a) Self-attention ENMASCARADA: solo mira los tokens anteriores
            x = ln(h, capa.self_attn_layer_norm)
            a, r1 = atencion(x, x, sa.q_proj, sa.k_proj, sa.v_proj, sa.out_proj, self.H_dec,
                             "decoder_self", n, causal=True)
            h = h + a
            # (b) CROSS-attention: Q del texto escrito, K y V de la memoria del audio
            x = ln(h, capa.cross_attention_layer_norm)
            a, r2 = atencion(x, memoria, ca.q_proj, ca.k_proj, ca.v_proj, ca.out_proj, self.H_dec, "cross", n)
            h = h + a
            # (c) FFN ReLU 1024 -> 8192 -> 1024
            x = ln(h, capa.ffn_layer_norm)
            h = h + F.linear(F.relu(F.linear(x, capa.ffn.fc1.weight, capa.ffn.fc1.bias)),
                             capa.ffn.fc2.weight, capa.ffn.fc2.bias)
            if guardar:
                registros += [r1, r2]
        h = ln(h, dec.layer_norm)
        if solo_ultimo:
            h = h[-1:]
        logits = F.linear(h, self.m.lm_head.weight)               # (T, 256 102)
        return logits, registros

    # ------------------------------ GENERACIÓN ----------------------------------- #
    @torch.no_grad()
    def traducir(self, features: torch.Tensor, idioma_destino: str = "eng", max_tokens: int = 120,
                 capas_encoder=None, codificado=None) -> ResultadoInferencia:
        """Greedy: </s> <idioma> -> predecir -> añadir -> repetir hasta </s>.
        El token de idioma le dice al decoder EN QUÉ idioma escribir: con el idioma
        del audio hace transcripción (ASR); con otro, traducción (S2TT).
        codificado: salida de codificar() ya calculada (el encoder se ejecuta UNA vez
        y la misma memoria sirve para transcribir y para traducir)."""
        memoria, reg_enc, T_enc = codificado if codificado is not None else self.codificar(features, capas_encoder)
        tokens = [self.c.decoder_start_token_id, self.lang_ids[idioma_destino]]
        probs, alternativas = [], []
        for _ in range(max_tokens):
            logits, _ = self.decodificar(tokens, memoria, guardar=False, solo_ultimo=True)
            p = torch.softmax(logits[-1].float(), dim=-1)
            top_p, top_i = p.topk(5)
            sig = int(top_i[0])
            probs.append(float(top_p[0]))
            alternativas.append([(self.tok.convert_ids_to_tokens(int(i)), float(v)) for v, i in zip(top_p, top_i)])
            tokens.append(sig)
            if sig == self.c.eos_token_id:
                break
        entrada = tokens[:-1] if tokens[-1] == self.c.eos_token_id else tokens
        logits, reg_dec = self.decodificar(entrada, memoria)
        texto = self.tok.decode(tokens, skip_special_tokens=True).strip()
        return ResultadoInferencia(tokens, texto, probs, alternativas, T_enc, memoria.shape[0],
                                   reg_enc, reg_dec, memoria, logits)
