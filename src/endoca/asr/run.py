#!/usr/bin/env python3
"""Run ASR reconciliation from atomic-question answers."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any


from endoca.inference import runner as shared_runner


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def row_key(item: dict[str, Any]) -> str:
    return str(item.get("probe_id") or item.get("sample_id") or "")


def row_is_success(item: dict[str, Any]) -> bool:
    return not item.get("error") and bool(str(item.get("revised_answer") or item.get("prediction") or "").strip())


def completed_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row_is_success(row):
                keys.add(row_key(row))
    return keys


def build_prompt(item: dict[str, Any]) -> str:
    return (
        "You are an Atomic-Support Reconciliation module for endoscopic VQA.\n"
        "Use the complex question, the model's original direct answer, and the atomic-question answers below as contextual premises. "
        "Treat these atomic answers as tentative premises rather than verified facts.\n\n"
        "Tasks:\n"
        "1. Produce revised_answer: a concise final answer to the complex question that is best supported by the atomic-question premises.\n"
        "2. If the direct answer conflicts with the atomic-question answers, revise it. If the atomic answers are insufficient, give the best cautious answer supported by them.\n"
        "3. Produce support_status as exactly one of: fully_supported, partially_supported, insufficient_or_conflicting.\n"
        "4. Produce selective_answer: repeat revised_answer only when support_status is fully_supported; otherwise use exactly insufficient support.\n\n"
        "Return only one compact JSON object with keys: revised_answer, support_status, selective_answer.\n\n"
        f"Complex question: {item.get('question')}\n"
        f"Original direct answer: {item.get('direct_prediction')}\n\n"
        f"Atomic-question answers used as contextual premises:\n{item.get('atom_context')}\n\n"
        "JSON:"
    )


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return {}


def normalize_status(value: Any) -> str:
    text = re.sub(r"[^a-z_]+", "_", str(value or "").strip().lower()).strip("_")
    if text in {"full", "fully", "supported", "fully_supported", "sufficient", "sufficiently"}:
        return "fully_supported"
    if text in {"partial", "partially_supported", "partially"}:
        return "partially_supported"
    if text in {"insufficient", "conflicting", "insufficient_or_conflicting", "unsupported"}:
        return "insufficient_or_conflicting"
    return "insufficient_or_conflicting"


def move_inputs(inputs: Any, device: Any) -> Any:
    try:
        return inputs.to(device)
    except Exception:
        if isinstance(inputs, dict):
            return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        return inputs


def first_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return getattr(model, "device", "cuda:0")


def ensure_generation_mixin(module: Any) -> None:
    if hasattr(module, "generate"):
        return
    from transformers.generation.utils import GenerationMixin

    module.__class__ = type(f"{module.__class__.__name__}WithGenerationMixin", (module.__class__, GenerationMixin), {})


def generic_text_predict(bundle: dict[str, Any], prompt: str, max_new_tokens: int) -> str:
    processor = bundle["processor"]
    model = bundle["model"]
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    texts = []
    try:
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    except Exception:
        pass
    texts.append(prompt)
    last_error = None
    for text in texts:
        try:
            inputs = processor(text=[text], return_tensors="pt", padding=True)
        except Exception as exc:
            last_error = exc
            try:
                inputs = processor(text, return_tensors="pt")
            except Exception as exc2:
                last_error = exc2
                continue
        inputs = move_inputs(inputs, first_device(model))
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return shared_runner.decode(processor, generated, inputs)
    raise RuntimeError(f"Processor text-only failed: {type(last_error).__name__}: {last_error}")


def predict_text(bundle: dict[str, Any], prompt: str, max_new_tokens: int) -> str:
    adapter = bundle["adapter"]
    if adapter == "internvl":
        language_model = getattr(bundle["model"], "language_model", None)
        if language_model is not None:
            ensure_generation_mixin(language_model)
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        return bundle["model"].chat(bundle["tokenizer"], None, prompt, generation_config)
    if adapter == "minicpm":
        msgs = [{"role": "user", "content": prompt}]
        return bundle["model"].chat(
            msgs=msgs,
            image=None,
            tokenizer=bundle["tokenizer"],
            processor=bundle["processor"],
            max_new_tokens=max_new_tokens,
            sampling=False,
        )
    return generic_text_predict(bundle, prompt, max_new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR reconciliation")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/models"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--prompt-version", default="bibm_asr_reconcile_v1")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--adapter",
        choices=["auto", "generic", "internvl", "minicpm", "gemma", "qwen25", "qwen3_vl", "simula_medgemma", "simula_qwen25", "llava_med"],
        default="auto",
    )
    parser.add_argument("--device-map", choices=["auto", "balanced", "balanced_low_0", "sequential"], default="auto")
    parser.add_argument("--internvl-device-map", choices=["single", "split"], default="single")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--adapter-path", default="", help="Optional local PEFT adapter path.")
    parser.add_argument("--llava-model-path", default="", help="Optional assembled LLaVA-Med model path.")
    parser.add_argument("--offline", action="store_true", help="Use only locally cached model files.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    shared_runner.LOCAL_FILES_ONLY = bool(args.offline)
    shared_runner.QWEN25_ADAPTER_PATH = args.adapter_path
    shared_runner.LLAVA_MODEL_PATH = args.llava_model_path
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    started = time.time()
    bundle = shared_runner.load_any_model(
        args.model_id.strip(),
        args.cache_dir,
        args.trust_remote_code,
        args.adapter,
        args.device_map,
        args.internvl_device_map,
    )
    rows = read_jsonl(args.input, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(args.output) if args.resume else set()
    if done:
        rows = [row for row in rows if row_key(row) not in done]
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as f:
        for item in rows:
            result = dict(item)
            result["model_id"] = args.model_id.strip()
            result["model_class"] = bundle.get("model_class")
            result["adapter"] = bundle.get("adapter")
            result["visible_gpus"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            result["prompt_style"] = "asr_reconcile_text_only"
            result["prompt_version"] = args.prompt_version
            result["max_new_tokens"] = args.max_new_tokens
            result["asr_input_boundary"] = "atomic_context_premises"
            try:
                prompt = build_prompt(item)
                t0 = time.time()
                with torch.inference_mode():
                    raw = predict_text(bundle, prompt, args.max_new_tokens)
                parsed = parse_json_object(raw)
                revised = str(parsed.get("revised_answer") or "").strip()
                status = normalize_status(parsed.get("support_status"))
                selective = str(parsed.get("selective_answer") or "").strip()
                if not revised:
                    revised = raw.strip()
                if status != "fully_supported":
                    selective = "insufficient support"
                elif not selective or selective.lower() in {"n/a", "none", "null"}:
                    selective = revised
                result["raw_output"] = raw
                result["json_valid"] = bool(parsed)
                result["prediction"] = revised
                result["revised_answer"] = revised
                result["support_status"] = status
                result["selective_answer"] = selective
                result["latency_seconds"] = round(time.time() - t0, 3)
                result["error"] = ""
            except Exception as exc:
                result["raw_output"] = ""
                result["prediction"] = ""
                result["revised_answer"] = ""
                result["support_status"] = "generation_error"
                result["selective_answer"] = "insufficient support"
                result["json_valid"] = False
                result["latency_seconds"] = None
                result["error"] = f"{type(exc).__name__}: {exc}"
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "input": str(args.input),
                "output": str(args.output),
                "records_attempted_this_run": len(rows),
                "skipped_existing_success": len(done),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
