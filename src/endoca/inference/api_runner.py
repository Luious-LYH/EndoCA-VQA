#!/usr/bin/env python3
"""Run EndoCA inference through closed/API VLM endpoints.

The output row shape matches the local transformers trace runner so the normal
EndoCA scoring scripts can consume it directly.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

try:
    from PIL import Image
except Exception:  # Pillow is optional for closed-API runs.
    Image = None


DEFAULT_DATA_ROOT = Path("data/upstream")
DEFAULT_PROJECT_ROOT = Path(".")
RUNNER_VERSION = "run_closed_api_trace_inference_v3_pool_failover_global_rate_limit_20260602"


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def row_key(row: Dict[str, Any]) -> str:
    return str(row.get("probe_id") or row.get("id") or row.get("sample_id") or "")


def read_resume_rows(path: Path) -> Tuple[set[str], List[Dict[str, Any]], int]:
    done: set[str] = set()
    kept: List[Dict[str, Any]] = []
    dropped = 0
    if not path.exists():
        return done, kept, dropped
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                dropped += 1
                continue
            key = row_key(item)
            raw_output = str(item.get("raw_output") or "").strip()
            prediction = str(item.get("prediction") or "").strip()
            if key and not item.get("error") and raw_output and prediction and key not in done:
                done.add(key)
                kept.append(item)
            else:
                dropped += 1
    return done, kept, dropped


def endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}{path}"
    return f"{base}/v1{path}"


def resolve_image(image_path: str, project_root: Path, data_root: Path) -> Path:
    normalized = str(image_path or "").replace("\\", "/")
    raw = Path(normalized)
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend([project_root / raw, data_root / raw])
    parts = list(raw.parts)
    if parts and parts[0].lower() == "data":
        stripped = Path(*parts[1:])
        candidates.extend([project_root / stripped, data_root / stripped])
    if parts and parts[0] in {"Kvasir-VQA-x1", "Kvasir-VQA", "EndoBench", "EndoBench-Extended"}:
        candidates.append(data_root / raw)
    if raw.name:
        candidates.extend(
            [
                data_root / "Kvasir-VQA-x1" / "images" / raw.name,
                data_root / "Kvasir-VQA" / "images" / raw.name,
                data_root / "EndoBench" / "EndoBench-Images" / raw.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve image_path={image_path}")


def image_data_url(path: Path) -> Tuple[str, str, int, int]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    width = 0
    height = 0
    if Image is not None:
        with Image.open(path) as image:
            width, height = image.size
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}", mime, width, height


def anthropic_image_source(path: Path) -> Tuple[Dict[str, str], str, int, int]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    width = 0
    height = 0
    if Image is not None:
        with Image.open(path) as image:
            width, height = image.size
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "base64", "media_type": mime, "data": encoded}, mime, width, height


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict):
                value = part.get("text") or part.get("content")
                if value:
                    pieces.append(str(value))
        return "\n".join(pieces).strip()
    return str(content)


def extract_openai_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return content_to_text(message.get("content"))


def parse_openai_response_json(response: requests.Response) -> Dict[str, Any]:
    try:
        return response.json()
    except Exception:
        pass
    text = response.text or ""
    model = ""
    usage: Dict[str, Any] = {}
    content_parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            item = json.loads(payload)
        except Exception:
            continue
        model = model or str(item.get("model") or "")
        if item.get("usage"):
            usage = item.get("usage") or {}
        for choice in item.get("choices") or []:
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content = delta.get("content")
            if content is None:
                content = message.get("content")
            if content:
                content_parts.append(content_to_text(content))
    if content_parts or model or usage:
        return {
            "model": model,
            "usage": usage,
            "choices": [{"message": {"content": "".join(content_parts)}}],
            "parsed_from_sse": True,
        }
    raise


def extract_anthropic_text(data: Dict[str, Any]) -> str:
    parts = data.get("content") or []
    text_parts: List[str] = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(str(part.get("text") or ""))
        elif isinstance(part, str):
            text_parts.append(part)
    return "\n".join(text_parts).strip()


def build_trace_prompt(item: Dict[str, Any], prompt_style: str) -> str:
    question = str(item.get("question") or item.get("source_question") or "").strip()
    if prompt_style == "json":
        return (
            "Answer the question based on the endoscopic image. "
            "Return only one compact JSON object with schema "
            '{"answer":"<short answer>"}. Do not explain.\n'
            f"Question: {question}"
        )
    return (
        "Answer the question based on the endoscopic image. "
        "Respond with a concise short answer only. Do not explain.\n"
        f"Question: {question}"
    )


def strip_json_answer(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    match = re.search(r"\{.*\}", value, flags=re.S)
    candidates = [value]
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and parsed.get("answer") is not None:
            return str(parsed.get("answer")).strip()
    return value


def parse_json(text: str) -> bool:
    if not text:
        return False
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            return isinstance(json.loads(candidate), dict)
        except Exception:
            continue
    return False


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(frozen=True)
class EndpointCandidate:
    index: int
    base_url: str
    api_key_env: str
    api_key: str
    label: str


@dataclass
class EndpointState:
    effort_disabled: bool = False
    temperature_disabled: bool = False
    cooldown_until: float = 0.0
    disabled: bool = False
    last_request_at: float = 0.0


class ApiClient:
    def __init__(self, args: argparse.Namespace, candidates: list[EndpointCandidate]) -> None:
        self.args = args
        self.candidates = candidates
        self.session = requests.Session()
        self.lock = threading.Lock()
        self.states = [EndpointState() for _ in candidates]
        self.next_candidate_index = 0

    def _candidate_order(self) -> list[int]:
        with self.lock:
            now = time.time()
            start = self.next_candidate_index
            self.next_candidate_index = (self.next_candidate_index + 1) % len(self.candidates)
            order = list(range(start, len(self.candidates))) + list(range(0, start))
            active = [
                idx
                for idx in order
                if not self.states[idx].disabled and self.states[idx].cooldown_until <= now
            ]
            return active or order

    def _wait_for_rate_limit(self, candidate_idx: int) -> None:
        if self.args.min_request_interval <= 0:
            return
        with self.lock:
            state = self.states[candidate_idx]
            now = time.time()
            wait = self.args.min_request_interval - (now - state.last_request_at)
            if wait > 0:
                time.sleep(wait)
            state.last_request_at = time.time()

    def _wait_for_global_rate_limit(self, candidate: EndpointCandidate) -> None:
        interval = float(getattr(self.args, "global_min_request_interval", 0.0) or 0.0)
        rate_dir = Path(str(getattr(self.args, "global_rate_limit_dir", "") or ""))
        if interval <= 0 or not str(rate_dir):
            return
        rate_dir.mkdir(parents=True, exist_ok=True)
        lock_key = "|".join(
            [
                str(getattr(self.args, "provider", "")),
                str(getattr(self.args, "api_type", "")),
                str(getattr(self.args, "model", "")),
                candidate.base_url.rstrip("/"),
            ]
        )
        digest = hashlib.sha256(lock_key.encode("utf-8")).hexdigest()[:16]
        lock_path = rate_dir / f"{digest}.lock"
        stamp_path = rate_dir / f"{digest}.ts"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            locked_with = ""
            try:
                import fcntl  # type: ignore

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                locked_with = "fcntl"
            except Exception:
                try:
                    import msvcrt  # type: ignore

                    lock_handle.seek(0)
                    if not lock_handle.read(1):
                        lock_handle.write("\0")
                        lock_handle.flush()
                    lock_handle.seek(0)
                    msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, 1)
                    locked_with = "msvcrt"
                except Exception:
                    pass
            try:
                last = 0.0
                if stamp_path.exists():
                    try:
                        last = float(stamp_path.read_text(encoding="utf-8").strip() or "0")
                    except Exception:
                        last = 0.0
                now = time.time()
                wait = interval - (now - last)
                if wait > 0:
                    time.sleep(wait)
                    now = time.time()
                stamp_path.write_text(f"{now:.6f}\n", encoding="utf-8")
            finally:
                if locked_with == "fcntl":
                    try:
                        import fcntl  # type: ignore

                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
                elif locked_with == "msvcrt":
                    try:
                        import msvcrt  # type: ignore

                        lock_handle.seek(0)
                        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass

    @staticmethod
    def _looks_like_effort_error(body: str) -> bool:
        lowered = body.lower()
        return any(key in lowered for key in ["reasoning_effort", "thinking"])

    @staticmethod
    def _looks_like_temperature_error(body: str) -> bool:
        lowered = body.lower()
        return "temperature" in lowered and any(key in lowered for key in ["unsupported", "unknown", "not support", "invalid"])

    @staticmethod
    def _looks_like_auth_error(status: int | None, body: str) -> bool:
        lowered = body.lower()
        return status in {401, 403} or any(
            key in lowered
            for key in [
                "api_key_disabled",
                "api key disabled",
                "invalid api key",
                "invalid_api_key",
                "unauthorized",
                "forbidden",
                "not authorized",
                "authentication",
            ]
        )

    def _should_failover(self, status: int | None, body: str) -> bool:
        if self._looks_like_auth_error(status, body):
            return True
        if status in {408, 409, 429, 500, 502, 503, 504, 524}:
            return True
        lowered = body.lower()
        return status in {404, 422} and any(key in lowered for key in ["model", "not found", "endpoint", "unsupported"])

    def _retry_delay(self, status: int | None, attempt: int, response: requests.Response | None) -> float:
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                return min(self.args.retry_max_sleep, max(0.0, float(retry_after)))
            except ValueError:
                pass
        delay = self.args.retry_sleep * (2 ** (attempt - 1))
        if status == 429:
            delay = max(delay, self.args.rate_limit_sleep)
        return min(self.args.retry_max_sleep, delay)

    def _disable_candidate(self, candidate_idx: int) -> None:
        with self.lock:
            self.states[candidate_idx].disabled = True
            self.states[candidate_idx].cooldown_until = float("inf")

    def _cooldown_candidate(self, candidate_idx: int, seconds: float) -> None:
        with self.lock:
            state = self.states[candidate_idx]
            state.cooldown_until = max(state.cooldown_until, time.time() + max(0.0, seconds))

    def call(self, prompt: str, image_path: Path) -> Tuple[str, Dict[str, Any]]:
        last_error = ""
        attempts = max(1, self.args.retries)
        for candidate_idx in self._candidate_order():
            candidate = self.candidates[candidate_idx]
            abort_all_candidates = False
            for attempt in range(1, attempts + 1):
                state = self.states[candidate_idx]
                send_effort = bool(self.args.reasoning_effort) and not state.effort_disabled
                try:
                    if self.args.api_type == "anthropic_compatible":
                        text, meta = self._call_anthropic(prompt, image_path, send_effort, candidate_idx, candidate)
                    else:
                        text, meta = self._call_openai(prompt, image_path, send_effort, candidate_idx, candidate)
                    if not str(text or "").strip():
                        last_error = f"[{candidate.label}] empty response text"
                        if attempt < attempts:
                            time.sleep(min(self.args.retry_max_sleep, self.args.retry_sleep * (2 ** (attempt - 1))))
                            continue
                        self._cooldown_candidate(candidate_idx, self.args.retry_max_sleep)
                        break
                    meta.update(
                        {
                            "attempts": attempt,
                            "api_pool_index": candidate.index,
                            "api_pool_size": len(self.candidates),
                            "api_key_env": candidate.api_key_env,
                            "api_pool_label": candidate.label,
                            "base_url": candidate.base_url,
                        }
                    )
                    return text, meta
                except requests.HTTPError as exc:
                    response = exc.response
                    status = response.status_code if response is not None else None
                    body = response.text[:1000] if response is not None else str(exc)
                    last_error = f"[{candidate.label}] HTTPError status={status}: {body}"
                    if status == 400 and send_effort and self._looks_like_effort_error(body):
                        with self.lock:
                            self.states[candidate_idx].effort_disabled = True
                        continue
                    if status == 400 and (not state.temperature_disabled) and self._looks_like_temperature_error(body):
                        with self.lock:
                            self.states[candidate_idx].temperature_disabled = True
                        continue
                    if self._looks_like_auth_error(status, body):
                        self._disable_candidate(candidate_idx)
                        break
                    if self._should_failover(status, body):
                        if attempt < attempts:
                            time.sleep(self._retry_delay(status, attempt, response))
                            continue
                        self._cooldown_candidate(candidate_idx, self._retry_delay(status, attempt, response))
                        break
                    abort_all_candidates = True
                    break
                except requests.RequestException as exc:
                    last_error = f"[{candidate.label}] {type(exc).__name__}: {exc}"
                    if len(self.candidates) == 1 and attempt < attempts:
                        time.sleep(min(self.args.retry_max_sleep, self.args.retry_sleep * (2 ** (attempt - 1))))
                        continue
                    self._cooldown_candidate(candidate_idx, self.args.retry_max_sleep)
                    break
                except Exception as exc:
                    last_error = f"[{candidate.label}] {type(exc).__name__}: {exc}"
                    if attempt < attempts:
                        time.sleep(min(self.args.retry_max_sleep, self.args.retry_sleep * (2 ** (attempt - 1))))
                        continue
                    self._cooldown_candidate(candidate_idx, self.args.retry_max_sleep)
                    break
            if abort_all_candidates:
                break
        raise RuntimeError(last_error)

    def _call_openai(
        self,
        prompt: str,
        image_path: Path,
        send_effort: bool,
        candidate_idx: int,
        candidate: EndpointCandidate,
    ) -> Tuple[str, Dict[str, Any]]:
        data_url, mime, width, height = image_data_url(image_path)
        state = self.states[candidate_idx]
        payload: Dict[str, Any] = {
            "model": self.args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": self.args.max_tokens,
        }
        if not state.temperature_disabled and not self.args.omit_temperature:
            payload["temperature"] = 0
        if send_effort:
            payload["reasoning_effort"] = self.args.reasoning_effort
        headers = {"Authorization": f"Bearer {candidate.api_key}", "Content-Type": "application/json"}
        started = time.time()
        self._wait_for_rate_limit(candidate_idx)
        self._wait_for_global_rate_limit(candidate)
        response = self.session.post(
            endpoint(candidate.base_url, "/chat/completions"),
            headers=headers,
            json=payload,
            timeout=self.args.timeout,
        )
        response.raise_for_status()
        data = parse_openai_response_json(response)
        return extract_openai_text(data), {
            "http_status": response.status_code,
            "api_response_model": data.get("model") or "",
            "image_mime_type": mime,
            "image_original_width": width,
            "image_original_height": height,
            "image_used_width": width,
            "image_used_height": height,
            "usage": data.get("usage") or {},
            "reasoning_effort_sent": self.args.reasoning_effort if send_effort else "",
            "latency_seconds": round(time.time() - started, 3),
        }

    def _call_anthropic(
        self,
        prompt: str,
        image_path: Path,
        send_effort: bool,
        candidate_idx: int,
        candidate: EndpointCandidate,
    ) -> Tuple[str, Dict[str, Any]]:
        source, mime, width, height = anthropic_image_source(image_path)
        state = self.states[candidate_idx]
        payload: Dict[str, Any] = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": source},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if not state.temperature_disabled and not self.args.omit_temperature:
            payload["temperature"] = 0
        if send_effort and self.args.reasoning_effort in {"high", "xhigh", "max"}:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.args.thinking_budget_tokens}
            payload.pop("temperature", None)
        headers = {
            "x-api-key": candidate.api_key,
            "Authorization": f"Bearer {candidate.api_key}",
            "anthropic-version": self.args.anthropic_version,
            "Content-Type": "application/json",
        }
        started = time.time()
        self._wait_for_rate_limit(candidate_idx)
        self._wait_for_global_rate_limit(candidate)
        response = self.session.post(
            endpoint(candidate.base_url, "/messages"),
            headers=headers,
            json=payload,
            timeout=self.args.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return extract_anthropic_text(data), {
            "http_status": response.status_code,
            "api_response_model": data.get("model") or "",
            "image_mime_type": mime,
            "image_original_width": width,
            "image_original_height": height,
            "image_used_width": width,
            "image_used_height": height,
            "usage": data.get("usage") or {},
            "reasoning_effort_sent": self.args.reasoning_effort if send_effort else "",
            "latency_seconds": round(time.time() - started, 3),
        }


def run_one(
    client: ApiClient,
    row: Dict[str, Any],
    project_root: Path,
    data_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    started = time.time()
    result: Dict[str, Any] = {
        **row,
        "provider": args.provider,
        "model": args.model,
        "model_id": args.model,
        "model_class": args.api_type,
        "adapter": "closed_api",
        "api_type": args.api_type,
        "base_url": args.base_url,
        "reasoning_effort_requested": args.reasoning_effort,
        "payload_compat": "omit_temperature" if args.omit_temperature else "default",
        "result_source": args.result_source,
        "prompt_style": args.prompt_style,
        "prompt_version": args.prompt_version,
        "runner_version": RUNNER_VERSION,
        "image_policy": "original_base64",
        "max_image_pixels": 0,
        "qwen_max_pixels": 0,
        "image_resized": False,
        "max_new_tokens": args.max_tokens,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    try:
        image_ref = str(row.get("image") or row.get("source_image") or row.get("image_path") or "")
        image_path = resolve_image(image_ref, project_root, data_root)
        prompt = build_trace_prompt(row, args.prompt_style)
        raw_output, meta = client.call(prompt, image_path)
        result["raw_output"] = raw_output
        result["prediction"] = strip_json_answer(raw_output) if args.prompt_style == "json" else raw_output.strip()
        result["json_valid"] = parse_json(raw_output)
        result["resolved_image_path"] = str(image_path)
        if not str(result["raw_output"] or "").strip() or not str(result["prediction"] or "").strip():
            result["error"] = "empty_raw_or_prediction"
        else:
            result["error"] = ""
        result.update(meta)
    except Exception as exc:
        result["raw_output"] = ""
        result["prediction"] = ""
        result["json_valid"] = False
        result["latency_seconds"] = round(time.time() - started, 3)
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_resume_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_endpoint_candidates(args: argparse.Namespace) -> list[EndpointCandidate]:
    base_urls = split_csv(args.base_url_list) or [args.base_url]
    api_key_envs = split_csv(args.api_key_env_list) or [args.api_key_env]
    if len(base_urls) != len(api_key_envs):
        if len(base_urls) == 1 and len(api_key_envs) > 1:
            base_urls = base_urls * len(api_key_envs)
        elif len(api_key_envs) == 1 and len(base_urls) > 1:
            api_key_envs = api_key_envs * len(base_urls)
        else:
            raise SystemExit("--base-url-list and --api-key-env-list must have matching lengths")

    candidates: list[EndpointCandidate] = []
    for idx, (base_url, api_key_env) in enumerate(zip(base_urls, api_key_envs), start=1):
        api_key = os.environ.get(api_key_env)
        if not base_url or not api_key_env or not api_key:
            continue
        candidates.append(
            EndpointCandidate(
                index=idx,
                base_url=base_url,
                api_key_env=api_key_env,
                api_key=api_key,
                label=f"pool{idx}:{base_url.rstrip('/')}",
            )
        )
    if not candidates:
        raise SystemExit(
            f"Missing API key environment variable(s): {', '.join(api_key_envs) or args.api_key_env}"
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--api-type", choices=["openai_compatible", "anthropic_compatible"], required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-url-list", default="")
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--api-key-env-list", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--omit-temperature", action="store_true")
    parser.add_argument("--result-source", default="")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prompt-style", choices=["json", "answer_only"], default="answer_only")
    parser.add_argument("--prompt-version", default="closed_api_trace_answer_only_v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--retry-max-sleep", type=float, default=30.0)
    parser.add_argument("--rate-limit-sleep", type=float, default=20.0)
    parser.add_argument("--min-request-interval", type=float, default=0.0)
    parser.add_argument("--global-rate-limit-dir", type=Path, default=Path(""))
    parser.add_argument("--global-min-request-interval", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--anthropic-version", default="2023-06-01")
    parser.add_argument("--thinking-budget-tokens", type=int, default=1024)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-error-rows", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must satisfy 0 <= shard_index < num_shards")
    candidates = build_endpoint_candidates(args)

    rows = read_jsonl(args.input, args.limit)
    rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    done, resume_rows, dropped_resume_rows = read_resume_rows(args.output) if args.resume else (set(), [], 0)
    rows = [row for row in rows if row_key(row) not in done]
    if args.resume and args.output.exists():
        write_resume_rows(args.output, resume_rows)

    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    client = ApiClient(args, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    completed = 0
    errors = 0
    skipped_errors = 0
    output_mode = "a" if args.resume else "w"
    with args.output.open(output_mode, encoding="utf-8") as f:
        with futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            future_map = {executor.submit(run_one, client, row, project_root, data_root, args): row_key(row) for row in rows}
            for future in futures.as_completed(future_map):
                item = future.result()
                if item.get("error"):
                    errors += 1
                    if args.skip_error_rows:
                        skipped_errors += 1
                        completed += 1
                        if completed % max(1, args.progress_every) == 0:
                            print(
                                json.dumps(
                                    {
                                        "provider": args.provider,
                                        "model": args.model,
                                        "shard_index": args.shard_index,
                                        "num_shards": args.num_shards,
                                        "completed_this_run": completed,
                                        "remaining_this_run": len(rows) - completed,
                                        "errors_this_run": errors,
                                        "error_rows_skipped": skipped_errors,
                                        "output": str(args.output),
                                        "api_pool_size": len(candidates),
                                        "elapsed_seconds": round(time.time() - started, 3),
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                        continue
                completed += 1
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                if completed % max(1, args.progress_every) == 0:
                    print(
                        json.dumps(
                            {
                                "provider": args.provider,
                                "model": args.model,
                                "shard_index": args.shard_index,
                                "num_shards": args.num_shards,
                                "completed_this_run": completed,
                                "remaining_this_run": len(rows) - completed,
                                "errors_this_run": errors,
                                "error_rows_skipped": skipped_errors,
                                "output": str(args.output),
                                "api_pool_size": len(candidates),
                                "elapsed_seconds": round(time.time() - started, 3),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    print(
        json.dumps(
            {
                "provider": args.provider,
                "model": args.model,
                "input": str(args.input),
                "output": str(args.output),
                "limit": args.limit,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "completed_this_run": completed,
                "errors_this_run": errors,
                "error_rows_skipped": skipped_errors,
                "resume_kept_rows": len(resume_rows),
                "resume_dropped_rows": dropped_resume_rows,
                "api_pool_size": len(candidates),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
