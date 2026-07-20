#!/usr/bin/env python3
"""Run text-only ASR reconciliation with an OpenAI-compatible chat API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ALLOWED_STATUS = {"fully_supported", "partially_supported", "insufficient_or_conflicting"}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for row in read_jsonl(path):
        if row.get("sample_id") is not None and not row.get("error"):
            done.add(str(row["sample_id"]))
    return done


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_status(value: str) -> str:
    status = (value or "").strip().lower()
    status = status.replace("-", "_").replace(" ", "_")
    if status in ALLOWED_STATUS:
        return status
    if "fully" in status:
        return "fully_supported"
    if "partial" in status:
        return "partially_supported"
    return "insufficient_or_conflicting"


def post_chat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str,
) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    obj = json.loads(raw)
    content = obj["choices"][0]["message"].get("content") or ""
    return content, obj


def reconcile_row(
    row: dict[str, Any],
    template: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = template.format(
        question=row.get("question", ""),
        direct_prediction=row.get("direct_prediction", ""),
        atom_context=row.get("atom_context", ""),
    )
    started = time.time()
    last_error = ""
    for attempt in range(1, args.retries + 2):
        try:
            raw_output, response_obj = post_chat(
                args.base_url,
                args.api_key,
                args.model,
                prompt,
                args.max_tokens,
                args.timeout,
                args.reasoning_effort,
            )
            parsed = extract_json_object(raw_output)
            revised_answer = str(parsed.get("revised_answer") or "").strip()
            status = normalize_status(str(parsed.get("support_status") or ""))
            selective_answer = str(parsed.get("selective_answer") or "").strip()
            if status != "fully_supported":
                selective_answer = "insufficient support"
            elif not selective_answer or selective_answer.lower() == "insufficient support":
                selective_answer = revised_answer
            if not revised_answer:
                raise ValueError("empty revised_answer")
            return {
                "probe_id": row.get("probe_id"),
                "probe_kind": "asr_reconcile",
                "sample_id": row.get("sample_id"),
                "split": row.get("split"),
                "dataset": row.get("dataset"),
                "img_id": row.get("img_id"),
                "complexity": row.get("complexity"),
                "question_class": row.get("question_class"),
                "question": row.get("question"),
                "target_model_id": row.get("target_model_id"),
                "model_slug": row.get("model_slug"),
                "provider": "openai_compatible",
                "model": args.model,
                "api_type": "openai_compatible",
                "asr_input_version": row.get("asr_input_version"),
                "asr_runner_version": "run_asr_reconcile_api_v1",
                "reasoning_effort_requested": args.reasoning_effort,
                "direct_prediction": row.get("direct_prediction", ""),
                "atomic_total_observed": row.get("atomic_total_observed"),
                "revised_answer": revised_answer,
                "support_status": status,
                "selective_answer": selective_answer,
                "raw_output": raw_output,
                "json_valid": True,
                "error": "",
                "latency_seconds": round(time.time() - started, 3),
                "attempts": attempt,
                "usage": response_obj.get("usage", {}),
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {body[:500]}"
            else:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= args.retries:
                time.sleep(min(args.retry_sleep * attempt, 20))
    return {
        "probe_id": row.get("probe_id"),
        "probe_kind": "asr_reconcile",
        "sample_id": row.get("sample_id"),
        "target_model_id": row.get("target_model_id"),
        "model_slug": row.get("model_slug"),
        "provider": "openai_compatible",
        "model": args.model,
        "asr_runner_version": "run_asr_reconcile_api_v1",
        "revised_answer": "",
        "support_status": "",
        "selective_answer": "",
        "raw_output": "",
        "json_valid": False,
        "error": last_error,
        "latency_seconds": round(time.time() - started, 3),
        "attempts": args.retries + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR reconciliation through an OpenAI-compatible API")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("ASR_API_BASE_URL", "http://localhost:53580/v1"))
    parser.add_argument("--api-key-env", default="ASR_API_KEY")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.api_key = os.environ.get(args.api_key_env, "")
    if not args.api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")

    template = args.prompt_template.read_text(encoding="utf-8")
    rows = list(read_jsonl(args.manifest))
    if args.limit:
        rows = rows[: args.limit]
    done = load_done(args.out_jsonl)
    pending = [row for row in rows if str(row.get("sample_id")) not in done]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    completed = len(done)
    failures = 0

    print(json.dumps({"manifest_rows": len(rows), "already_done": len(done), "pending": len(pending)}, ensure_ascii=False))
    with args.out_jsonl.open("a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(reconcile_row, row, template, args): row for row in pending}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with lock:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    completed += 1
                    if result.get("error"):
                        failures += 1
                    if completed % 100 == 0 or completed == len(rows):
                        print(json.dumps({"completed": completed, "total": len(rows), "failures": failures}, ensure_ascii=False))

    print(json.dumps({"completed": completed, "total": len(rows), "failures": failures, "out_jsonl": str(args.out_jsonl)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
