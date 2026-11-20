"""
Prueba de que la reimplementación manual (src/seamless_paso_a_paso.py) es correcta:
con los mismos pesos y el mismo audio debe producir la misma memoria del encoder,
las mismas atenciones, los mismos logits y el mismo texto que la implementación
oficial de Hugging Face, tanto en transcripción (spa) como en traducción (eng).

Uso:  python scripts/verificar_equivalencia.py [ruta_audio]
"""

import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.modelo import cargar_audio, cargar_modelo, preprocesar, traducir_oficial  # noqa: E402
from src.seamless_paso_a_paso import SeamlessPasoAPaso  # noqa: E402


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else sorted((RAIZ / "ejemplos").glob("*.wav"))[0]
    proc, modelo = cargar_modelo()
    manual = SeamlessPasoAPaso(modelo, proc.tokenizer)
    audio = cargar_audio(str(ruta))
    x, mask = preprocesar(proc, audio)
    print(f"Audio: {Path(ruta).name} ({len(audio) / 16000:.1f} s) | input_features {tuple(x.shape)}")

    ok_total = True
    for idioma in ["spa", "eng"]:
        res = manual.traducir(x[0], idioma)
        texto_hf, _ = traducir_oficial(proc, modelo, audio, idioma)
        entrada = res.tokens[:-1] if res.tokens[-1] == modelo.config.eos_token_id else res.tokens
        with torch.no_grad():
            out = modelo(input_features=x, attention_mask=mask, decoder_input_ids=torch.tensor([entrada]),
                         output_attentions=True)
            enc = modelo.speech_encoder(x, attention_mask=mask, output_attentions=True)
        d_mem = (enc.last_hidden_state[0] - res.memoria).abs().max().item()
        d_att = max((a[0] - r.pesos).abs().max().item()
                    for a, r in zip(enc.attentions, [r for r in res.registros_encoder if r.tipo == "encoder_self"]))
        cross = [r.pesos for r in res.registros_decoder if r.tipo == "cross"]
        d_cross = max((a[0] - b).abs().max().item() for a, b in zip(out.cross_attentions, cross))
        d_logits = (out.logits[0] - res.logits_finales).abs().max().item()
        ok = res.texto == texto_hf and d_logits < 1e-2
        ok_total &= ok
        print(f"\n[{idioma}] tokens: {len(res.tokens)}")
        print(f"  máx |Δ| memoria del encoder : {d_mem:.2e}")
        print(f"  máx |Δ| atención encoder    : {d_att:.2e}")
        print(f"  máx |Δ| cross-attention     : {d_cross:.2e}")
        print(f"  máx |Δ| logits              : {d_logits:.2e}")
        print(f"  manual  : {res.texto}")
        print(f"  oficial : {texto_hf}")
        print("  RESULTADO:", "IDÉNTICO ✔" if ok else "DIFERENTE ✘")
    sys.exit(0 if ok_total else 1)


if __name__ == "__main__":
    main()
