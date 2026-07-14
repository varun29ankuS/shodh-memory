#!/usr/bin/env python3
"""Export the GLiNER bi-edge NER model to ONNX for the Rust runtime (typer Task 4/5).

`knowledgator/gliner-bi-edge-v2.0` is a *bi-encoder* span GLiNER:

  - TEXT tower : `jhu-clsp/ettin-encoder-32m` (ModernBERT, 384-d) + an RNN span
                 representation head + span/label bilinear scorer.
  - LABEL tower: `sentence-transformers/all-MiniLM-L6-v2` (BERT, 384-d), pooled to
                 one vector per entity type.

The bi-encoder's whole point: entity-type labels are encoded **once, offline**. So the
ONNX we ship is the TEXT tower + scorer, taking pre-computed label embeddings as an
INPUT (`labels_embeds`, shape (num_labels, 384)). At runtime Rust feeds the 141
pre-computed fine-label embeddings (this script also emits them) and never runs the
label tower — cheap, deterministic, on-device.

## Which escalation rung produced this
Rung 1 (upgrade `gliner`/`optimum`/`onnxruntime` to latest) is applied first. gliner
0.2.27 *has* a bi-encoder ONNX exporter (`export_to_onnx(from_labels_embeddings=True)`)
and the upgraded `torch.onnx` traces the ettin+RNN text tower cleanly at opset 19.
BUT gliner 0.2.27's built-in `export_to_onnx(from_labels_embeddings=True)` is broken
by three defects (see WORKAROUNDS below), so we drive gliner's own export *primitives*
(`encode_labels`, `_build_dummy_batch`, the core module, `torch.onnx.export`) directly
with a correct embeddings wrapper. Net: gliner's export machinery + a thin correct shim
= Rung 1/2. No pre-made community ONNX (Rung 5) was needed.

## gliner 0.2.27 defects worked around (all in the bi-encoder path)
1. `BiEncoder.encode_labels` (modeling/encoder.py) forwards the text-encoder-only
   packing kwargs `token_lengths`/`word_lengths` into the MiniLM label encoder, which
   is a plain `BertModel` and raises `TypeError: unexpected keyword argument
   'token_lengths'`. It only pops `packing_config`/`pair_attention_mask`. → we monkeypatch
   `encode_labels` to also drop `token_lengths`/`word_lengths`. Needed for the torch
   reference to run at all.
2. `export_to_onnx` forwards `from_labels_embeddings=` into `_build_dummy_batch`, which
   only accepts `labels`/`text` → `TypeError`. → we call `_build_dummy_batch(labels=...)`
   ourselves.
3. Even past (2), the built-in `BiEncoderWrapper` reads its parameter names from the
   *full* input spec (8 names incl. `labels_input_ids`/`labels_attention_mask`) while the
   embeddings batch supplies 7 tensors, so `zip()` mis-binds the label embeddings to
   `labels_input_ids`. → we build our own 7-input embeddings wrapper.

## Outputs (into --out-dir, git-ignored via `models/`)
  model.onnx              fp32 text-tower + scorer, `labels_embeds` input
  model_int8.onnx         int8 dynamic-quantized (QInt8) — optional, best-effort
  label_embeddings.bin    raw little-endian float32, (num_fine, 384) row-major, fine order
  label_embeddings.json   sidecar: label order + shape + dtype for the Rust loader
  gliner_config.json      model config (emitted by gliner)
  tokenizer.json + tokenizer_config.json + special_tokens_map.json   text-tower tokenizer
  parity.json             parity-gate results (fp32 & int8 vs torch on the passages)

Run:
  python scripts/export_gliner_bi_edge.py \
      --model-dir models/gliner-bi-edge-v2_0 \
      --schema src/entity_type/entity-type-schema.json \
      --passages demos/gdelt-bridge/passages_100.jsonl \
      --out-dir models/gliner-bi-edge
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np


BASE_INPUTS = ["input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask"]

# Key entities the GDELT Baltimore-bridge demo depends on: surface -> acceptable label(s).
# The fine schema carries a dedicated `cargo ship` label (more specific than the generic
# `ship`), so the ship family is all acceptable for the vessel mention.
KEY_ENTITIES = {
    "baltimore": {"city"},
    "cargo ship": {"cargo ship", "ship", "warship", "vessel", "watercraft"},
    "francis scott key bridge": {"bridge"},
    "patapsco": {"river"},
}

# Opset 19: minimum that covers the ModernBERT/ettin ops emitted by torch 2.10's exporter.
OPSET = 19


def log(msg: str) -> None:
    print(msg, flush=True)


def patch_encode_labels_leak():
    """Workaround defect (1): stop text-encoder packing kwargs leaking into the label tower."""
    import gliner.modeling.encoder as enc

    orig = enc.BiEncoder.encode_labels

    def patched(self, input_ids, attention_mask, *args, **kwargs):
        for k in ("packing_config", "pair_attention_mask", "token_lengths", "word_lengths"):
            kwargs.pop(k, None)
        return orig(self, input_ids, attention_mask, *args, **kwargs)

    enc.BiEncoder.encode_labels = patched


def load_fine_labels(schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    labels = [entry["label"] for entry in schema["fine"]]
    if not labels:
        raise ValueError(f"no fine labels found in {schema_path}")
    # De-dup while preserving order (row order of label_embeddings.bin must be stable).
    seen: dict[str, None] = {}
    for lab in labels:
        seen.setdefault(lab, None)
    return list(seen.keys())


def build_embeddings_wrapper(core):
    import torch.nn as nn

    class EmbeddingsWrapper(nn.Module):
        """Text tower + scorer taking pre-computed label embeddings as input (7 inputs)."""

        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask, labels_embeds):
            out = self.core(
                input_ids=input_ids,
                attention_mask=attention_mask,
                words_mask=words_mask,
                text_lengths=text_lengths,
                span_idx=span_idx,
                span_mask=span_mask,
                labels_embeds=labels_embeds,
            )
            return out.logits if hasattr(out, "logits") else out[0]

    return EmbeddingsWrapper(core).eval()


def export_fp32(model, fine_labels, out_dir: Path):
    """Export the embeddings-mode text tower to fp32 ONNX. Returns (onnx_path, label_embeds np)."""
    import torch

    core = model.model.to("cpu").eval()

    # Pre-compute the fine-label embeddings once (clean label-tower path).
    label_embeds = model.encode_labels(fine_labels).to("cpu").float().contiguous()
    log(f"  label embeddings: {tuple(label_embeds.shape)} {label_embeds.dtype}")

    # Realistic text-tower inputs (shapes are dynamic; values only seed the trace).
    batch = model._build_dummy_batch(labels=fine_labels)
    base_inputs = tuple(batch[n] for n in BASE_INPUTS)

    wrapper = build_embeddings_wrapper(core)
    onnx_path = out_dir / "model.onnx"
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "words_mask": {0: "batch", 1: "seq"},
        "text_lengths": {0: "batch", 1: "v"},
        "span_idx": {0: "batch", 1: "num_spans", 2: "idx"},
        "span_mask": {0: "batch", 1: "num_spans"},
        "labels_embeds": {0: "num_labels", 1: "hidden"},
        "logits": {0: "batch", 1: "seq", 2: "num_spans", 3: "num_classes"},
    }
    all_inputs = base_inputs + (label_embeds,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper,
            all_inputs,
            f=str(onnx_path),
            input_names=BASE_INPUTS + ["labels_embeds"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
        )
    log(f"  fp32 ONNX: {onnx_path.name}  {onnx_path.stat().st_size/1e6:.1f} MB")
    return onnx_path, label_embeds.numpy().astype(np.float32)


def quantize_int8(fp32_path: Path, int8_path: Path) -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as e:  # noqa: BLE001
        log(f"  int8: onnxruntime.quantization unavailable ({e}); skipped")
        return False
    try:
        quantize_dynamic(model_input=str(fp32_path), model_output=str(int8_path), weight_type=QuantType.QInt8)
    except Exception as e:  # noqa: BLE001
        log(f"  int8: quantization failed ({e}); skipped")
        return False
    log(f"  int8 ONNX: {int8_path.name}  {int8_path.stat().st_size/1e6:.1f} MB")
    return True


def serialize_label_embeddings(label_embeds: np.ndarray, fine_labels: list[str], out_dir: Path):
    assert label_embeds.shape[0] == len(fine_labels), "embedding rows must match label count"
    bin_path = out_dir / "label_embeddings.bin"
    label_embeds.astype("<f4", copy=False).tofile(bin_path)
    meta = {
        "labels": fine_labels,
        "num_labels": int(label_embeds.shape[0]),
        "hidden_size": int(label_embeds.shape[1]),
        "dtype": "float32",
        "byte_order": "little",
        "layout": "row-major (num_labels, hidden_size); row i is the embedding of labels[i]",
    }
    (out_dir / "label_embeddings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"  label_embeddings.bin: {bin_path.stat().st_size/1e6:.2f} MB  ({meta['num_labels']}x{meta['hidden_size']})")


def load_passages(path: Path) -> list[str]:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        texts.append(json.loads(line)["text"])
    return texts


def entity_key(ent, with_label: bool):
    return (ent["start"], ent["end"], ent["label"]) if with_label else (ent["start"], ent["end"])


def span_f1(ref_sets, cand_sets) -> float:
    """Micro-F1 of predicted spans across all passages (candidate vs reference)."""
    tp = fp = fn = 0
    for ref, cand in zip(ref_sets, cand_sets):
        tp += len(ref & cand)
        fp += len(cand - ref)
        fn += len(ref - cand)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 1.0


def run_onnx_parity(model, session, label_embeds_np, fine_labels, texts, threshold):
    """Reference = torch; candidate = same pipeline but span logits from the ONNX session.

    We swap only the neural forward's logits, so pre-processing and the span decoder are
    byte-identical between reference and candidate — the comparison isolates the exported
    graph. Label order fed to ONNX == fine_labels == the collator's class order.
    """
    import types
    import torch

    core = model.model

    def onnx_forward(self, *args, **kw):
        out = orig_forward(*args, **kw)
        feeds = {n: kw[n].cpu().numpy() for n in BASE_INPUTS}
        feeds["labels_embeds"] = label_embeds_np
        logits = session.run(None, feeds)[0]
        out.logits = torch.from_numpy(np.ascontiguousarray(logits))
        return out

    ref_with, ref_wo, cand_with, cand_wo = [], [], [], []
    ref_key_hits, cand_key_hits = {}, {}

    def scan_keys(ents, bucket):
        for e in ents:
            surf = e["text"].strip().lower()
            for key, ok_labels in KEY_ENTITIES.items():
                if key in surf and e["label"] in ok_labels:
                    bucket[key] = True

    # Reference pass (pure torch).
    ref_entities = []
    for text in texts:
        ents = model.predict_entities(text, fine_labels, threshold=threshold)
        ref_entities.append(ents)
        ref_with.append({entity_key(e, True) for e in ents})
        ref_wo.append({entity_key(e, False) for e in ents})
        scan_keys(ents, ref_key_hits)

    # Candidate pass (ONNX logits swapped in).
    orig_forward = core.forward
    core.forward = types.MethodType(onnx_forward, core)
    try:
        for text in texts:
            ents = model.predict_entities(text, fine_labels, threshold=threshold)
            cand_with.append({entity_key(e, True) for e in ents})
            cand_wo.append({entity_key(e, False) for e in ents})
            scan_keys(ents, cand_key_hits)
    finally:
        core.forward = orig_forward

    return {
        "span_f1_span_only": round(span_f1(ref_wo, cand_wo), 4),
        "span_f1_span_label": round(span_f1(ref_with, cand_with), 4),
        "ref_total_spans": sum(len(s) for s in ref_wo),
        "cand_total_spans": sum(len(s) for s in cand_wo),
        "key_entities_ref": {k: ref_key_hits.get(k, False) for k in KEY_ENTITIES},
        "key_entities_cand": {k: cand_key_hits.get(k, False) for k in KEY_ENTITIES},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--model-dir", type=Path, default=repo / "models" / "gliner-bi-edge-v2_0")
    ap.add_argument("--schema", type=Path, default=repo / "src" / "entity_type" / "entity-type-schema.json")
    ap.add_argument("--passages", type=Path, default=repo / "demos" / "gdelt-bridge" / "passages_100.jsonl")
    ap.add_argument("--out-dir", type=Path, default=repo / "models" / "gliner-bi-edge")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--span-f1-gate", type=float, default=0.95)
    ap.add_argument("--no-int8", action="store_true", help="skip int8 quantization")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    t_start = time.time()

    import gliner
    import onnxruntime as ort
    import torch
    import transformers

    log("versions: "
        f"gliner={gliner.__version__} onnxruntime={ort.__version__} "
        f"torch={torch.__version__} transformers={transformers.__version__}")

    patch_encode_labels_leak()
    from gliner import GLiNER

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading torch model from {args.model_dir}")
    model = GLiNER.from_pretrained(str(args.model_dir), load_onnx_model=False)
    model.eval()
    log(f"  model class: {type(model).__name__}  labels_encoder: {model.config.labels_encoder}")

    fine_labels = load_fine_labels(args.schema)
    log(f"fine labels: {len(fine_labels)}")

    log("exporting fp32 ONNX (text tower + scorer, labels_embeds input)")
    fp32_path, label_embeds_np = export_fp32(model, fine_labels, args.out_dir)

    serialize_label_embeddings(label_embeds_np, fine_labels, args.out_dir)

    # gliner_config.json + tokenizer files for the text tower.
    model.config.to_json_file(str(args.out_dir / "gliner_config.json"))
    model.data_processor.transformer_tokenizer.save_pretrained(str(args.out_dir))
    log("  wrote gliner_config.json + tokenizer files")

    int8_path = args.out_dir / "model_int8.onnx"
    have_int8 = False
    if not args.no_int8:
        log("quantizing int8 (dynamic, QInt8)")
        have_int8 = quantize_int8(fp32_path, int8_path)

    texts = load_passages(args.passages)
    log(f"parity: {len(texts)} passages, threshold={args.threshold}")

    log("  parity: fp32 ONNX vs torch")
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    parity_fp32 = run_onnx_parity(model, sess_fp32, label_embeds_np, fine_labels, texts, args.threshold)

    parity_int8 = None
    if have_int8:
        log("  parity: int8 ONNX vs torch")
        sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        parity_int8 = run_onnx_parity(model, sess_int8, label_embeds_np, fine_labels, texts, args.threshold)

    sizes = {
        "model.onnx_MB": round(fp32_path.stat().st_size / 1e6, 1),
        "model_int8.onnx_MB": round(int8_path.stat().st_size / 1e6, 1) if have_int8 else None,
        "label_embeddings.bin_MB": round((args.out_dir / "label_embeddings.bin").stat().st_size / 1e6, 3),
    }
    report = {
        "versions": {
            "gliner": gliner.__version__,
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "opset": OPSET,
        "num_fine_labels": len(fine_labels),
        "threshold": args.threshold,
        "sizes": sizes,
        "parity_fp32": parity_fp32,
        "parity_int8": parity_int8,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (args.out_dir / "parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def key_ok(p):
        return all(p["key_entities_cand"].values())

    log("\n=== PARITY ===")
    log(f"  fp32  span-F1(span+label)={parity_fp32['span_f1_span_label']}  "
        f"span-F1(span-only)={parity_fp32['span_f1_span_only']}  "
        f"key-entities={parity_fp32['key_entities_cand']}")
    if parity_int8:
        log(f"  int8  span-F1(span+label)={parity_int8['span_f1_span_label']}  "
            f"span-F1(span-only)={parity_int8['span_f1_span_only']}  "
            f"key-entities={parity_int8['key_entities_cand']}")
    log(f"  sizes: {sizes}")
    log(f"  report: {args.out_dir / 'parity.json'}")

    gate = parity_fp32["span_f1_span_label"] >= args.span_f1_gate and key_ok(parity_fp32)
    log(f"\nGATE (fp32 span+label F1 >= {args.span_f1_gate} AND all key entities): {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
