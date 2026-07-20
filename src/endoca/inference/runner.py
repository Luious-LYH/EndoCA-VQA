#!/usr/bin/env python3
"""Run EndoCA inference with a transformers VLM loaded once."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, ImageFilter


DEFAULT_DATA_ROOT = Path("data/upstream")
DEFAULT_MAX_IMAGE_PIXELS = 0
DEFAULT_QWEN_MAX_PIXELS = 0
LOCAL_FILES_ONLY = False
QWEN25_ADAPTER_PATH = ""
LLAVA_MODEL_PATH = ""
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


def read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def row_key(item: Dict[str, Any]) -> str:
    return str(item.get("probe_id") or item.get("id") or item.get("sample_id") or "")


def row_is_success(item: Dict[str, Any]) -> bool:
    return (
        not item.get("error")
        and bool(str(item.get("prediction") or "").strip())
        and bool(str(item.get("raw_output") or "").strip())
    )


def completed_keys(path: Path) -> set[str]:
    keys = set()
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
            if not row_is_success(row):
                continue
            key = row_key(row)
            if key:
                keys.add(key)
    return keys


def resolve_image(image: str, data_root: Path) -> Optional[Path]:
    if not image:
        return None
    p = Path(image)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        data_root / image,
        data_root / "EndoBench" / "EndoBench-Images" / image,
        data_root / "EndoBench-Extended" / "Extended-Images" / image,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def attach_image_policy_metadata(
    image: Image.Image,
    *,
    original_size: tuple[int, int],
    resized: bool,
    max_image_pixels: int,
    force_pil_input: bool = False,
) -> Image.Image:
    image._aris_image_original_size = original_size  # type: ignore[attr-defined]
    image._aris_image_used_size = image.size  # type: ignore[attr-defined]
    image._aris_image_resized = resized  # type: ignore[attr-defined]
    image._aris_max_image_pixels = max_image_pixels  # type: ignore[attr-defined]
    image._aris_force_pil_input = force_pil_input  # type: ignore[attr-defined]
    return image


def apply_image_pixel_limit(image: Image.Image, max_image_pixels: int = 0) -> Image.Image:
    original_size = image.size
    resized = False
    if max_image_pixels > 0 and image.width * image.height > max_image_pixels:
        scale = math.sqrt(max_image_pixels / float(image.width * image.height))
        new_width = max(1, int(image.width * scale))
        new_height = max(1, int(image.height * scale))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized = True
    return attach_image_policy_metadata(
        image,
        original_size=original_size,
        resized=resized,
        max_image_pixels=max_image_pixels,
    )


def image_policy_metadata(
    image: Optional[Image.Image],
    *,
    max_image_pixels: int,
    qwen_max_pixels: int,
) -> Dict[str, Any]:
    if image is None:
        original_size = (None, None)
        used_size = (None, None)
        resized = None
    else:
        original_size = getattr(image, "_aris_image_original_size", image.size)
        used_size = getattr(image, "_aris_image_used_size", image.size)
        resized = getattr(image, "_aris_image_resized", False)
    return {
        "max_image_pixels": max_image_pixels,
        "qwen_max_pixels": qwen_max_pixels,
        "image_original_width": original_size[0],
        "image_original_height": original_size[1],
        "image_used_width": used_size[0],
        "image_used_height": used_size[1],
        "image_resized": resized,
    }


def load_image_for_probe(item: Dict[str, Any], data_root: Path, max_image_pixels: int = 0) -> Image.Image:
    image_name = item.get("image") or item.get("source_image")
    path = resolve_image(str(image_name or ""), data_root)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve image: {image_name}")
    image = Image.open(path).convert("RGB")
    image.filename = str(path)
    image = apply_image_pixel_limit(image, max_image_pixels)
    image.filename = str(path)
    transform = item.get("image_transform")
    if transform == "blank_or_heavy_blur_placeholder":
        original_size = getattr(image, "_aris_image_original_size", image.size)
        resized = bool(getattr(image, "_aris_image_resized", False))
        image = image.filter(ImageFilter.GaussianBlur(radius=24))
        image.filename = str(path)
        return attach_image_policy_metadata(
            image,
            original_size=original_size,
            resized=resized,
            max_image_pixels=max_image_pixels,
            force_pil_input=True,
        )
    return image


def build_prompt(item: Dict[str, Any], prompt_style: str) -> str:
    question = str(item.get("question") or item.get("source_question") or "").strip()
    options = item.get("options")
    option_letters = "A, B, C, D, or E"
    if options:
        letters = [chr(65 + i) for i in range(len(options))]
        if len(letters) == 1:
            option_letters = letters[0]
        elif len(letters) == 2:
            option_letters = f"{letters[0]} or {letters[1]}"
        else:
            option_letters = ", ".join(letters[:-1]) + f", or {letters[-1]}"
        option_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        question = f"{question}\nOptions:\n{option_text}"
    if prompt_style == "answer_only":
        if options:
            return (
                "Answer the visual question using the image. Respond with exactly one option letter "
                f"({option_letters}) and no explanation. If the image is insufficient or corrupted, "
                "respond exactly: uncertain.\n"
                f"Question: {question}"
            )
        return (
            "Answer the visual question using the image. Respond with a short answer only. "
            "If the image is insufficient or corrupted, respond exactly: uncertain.\n"
            f"Question: {question}"
        )
    return (
        "Return only one compact JSON object. Do not repeat the question. Do not use markdown. "
        'Schema: {"answer":"<option letter or short answer>","confidence":0.0}. '
        f"For multiple-choice questions, answer must be only {option_letters}. "
        'If the image is insufficient, use {"answer":"uncertain","confidence":0.0}.\n'
        f"Question: {question}"
    )


def first_device(model: Any):
    try:
        return next(model.parameters()).device
    except Exception:
        return getattr(model, "device", "cuda:0")


def move_inputs(inputs: Any, device: Any) -> Any:
    try:
        return inputs.to(device)
    except Exception:
        if isinstance(inputs, dict):
            return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        return inputs


def build_inputs(processor: Any, image: Image.Image, prompt: str):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    texts = []
    try:
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    except Exception:
        pass
    texts.extend([f"<image>\n{prompt}", prompt])
    last_error = None
    for text in texts:
        for kwargs in (
            {"text": [text], "images": [image], "return_tensors": "pt", "padding": True},
            {"text": text, "images": image, "return_tensors": "pt"},
        ):
            try:
                return processor(**kwargs)
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"Processor failed: {type(last_error).__name__}: {last_error}")


def decode(processor: Any, generated: Any, inputs: Any) -> str:
    try:
        input_len = inputs["input_ids"].shape[-1]
        # Decoder-only VLMs usually return prompt+completion, while some
        # image-to-text classes return only newly generated tokens.
        if getattr(generated, "shape", None) is not None and generated.shape[-1] > input_len:
            generated = generated[:, input_len:]
    except Exception:
        pass
    try:
        return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    except Exception:
        return processor.decode(generated[0], skip_special_tokens=True).strip()


def parse_json(text: str) -> bool:
    if not text:
        return False
    candidates = [text.strip()]
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for candidate in candidates:
        try:
            return isinstance(json.loads(candidate), dict)
        except Exception:
            continue
    return False


def load_model(model_id: str, cache_dir: Path, trust_remote_code: bool, device_map: str):
    import torch
    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=LOCAL_FILES_ONLY,
    )
    kwargs = {
        "cache_dir": str(cache_dir),
        "trust_remote_code": trust_remote_code,
        "local_files_only": LOCAL_FILES_ONLY,
        "torch_dtype": torch.float16,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    errors = []
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            model.eval()
            return processor, model, name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors[-3:]))


def infer_adapter(model_id: str, adapter: str) -> str:
    if adapter != "auto":
        return adapter
    lowered = model_id.lower()
    if "llava-med" in lowered:
        return "llava_med"
    if "simulamet/medgemma-kvasirvqa" in lowered:
        return "simula_medgemma"
    if "simulamet/qwen2.5-vl-kvasirvqa" in lowered:
        return "simula_qwen25"
    if "lingshu" in lowered:
        return "qwen25"
    if "qwen2.5-vl" in lowered:
        return "qwen25"
    if "qwen3-vl" in lowered:
        return "qwen3_vl"
    if "internvl" in lowered:
        return "internvl"
    if "minicpm" in lowered:
        return "minicpm"
    if "gemma" in lowered:
        return "gemma"
    return "generic"


def internvl_model_key(model_id: str) -> str:
    return Path(model_id.rstrip("/")).name


def split_internvl_device_map(model_id: str):
    import math
    import torch

    layer_counts = {
        "InternVL2_5-1B": 24,
        "InternVL2_5-2B": 24,
        "InternVL2_5-4B": 36,
        "InternVL2_5-8B": 32,
        "InternVL2_5-26B": 48,
        "InternVL2_5-38B": 64,
        "InternVL2_5-78B": 80,
    }
    model_name = internvl_model_key(model_id)
    if model_name not in layer_counts:
        raise ValueError(f"Unsupported InternVL split device map for model_id={model_id}")
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise ValueError("InternVL split device map needs at least 2 visible GPUs")

    num_layers = layer_counts[model_name]
    layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    layers_per_gpu = [layers_per_gpu] * world_size
    layers_per_gpu[0] = math.ceil(layers_per_gpu[0] * 0.5)
    device_map = {}
    layer_cnt = 0
    for gpu_idx, layer_count in enumerate(layers_per_gpu):
        for _ in range(layer_count):
            if layer_cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_idx
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def load_internvl_model(model_id: str, cache_dir: Path, internvl_device_map: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "use_flash_attn": False,
        "trust_remote_code": True,
        "cache_dir": str(cache_dir),
        "local_files_only": LOCAL_FILES_ONLY,
    }
    actual_device_map = "single"
    if internvl_device_map == "split":
        kwargs["device_map"] = split_internvl_device_map(model_id)
        actual_device_map = "internvl_split"
    model = AutoModel.from_pretrained(model_id, **kwargs).eval()
    if internvl_device_map != "split":
        model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=str(cache_dir),
        local_files_only=LOCAL_FILES_ONLY,
    )
    return {
        "adapter": "internvl",
        "model": model,
        "tokenizer": tokenizer,
        "model_class": "AutoModel.chat",
        "actual_device_map": actual_device_map,
    }


def build_internvl_pixels(image: Image.Image, max_tiles: int = 4):
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    def find_best(aspect_ratio: float, ratios: list[tuple[int, int]], width: int, height: int):
        best_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in ratios:
            diff = abs(aspect_ratio - ratio[0] / ratio[1])
            if diff < best_diff:
                best_diff = diff
                best_ratio = ratio
            elif diff == best_diff and area > 0.5 * 448 * 448 * ratio[0] * ratio[1]:
                best_ratio = ratio
        return best_ratio

    width, height = image.size
    ratios = sorted(
        {
            (i, j)
            for n in range(1, max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_tiles
        },
        key=lambda x: x[0] * x[1],
    )
    target_ratio = find_best(width / height, ratios, width, height)
    target_width = 448 * target_ratio[0]
    target_height = 448 * target_ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(target_ratio[0] * target_ratio[1]):
        box = (
            (i % (target_width // 448)) * 448,
            (i // (target_width // 448)) * 448,
            ((i % (target_width // 448)) + 1) * 448,
            ((i // (target_width // 448)) + 1) * 448,
        )
        tiles.append(resized.crop(box))
    if len(tiles) != 1:
        tiles.append(image.resize((448, 448)))
    return torch.stack([transform(tile) for tile in tiles]).to(torch.bfloat16).cuda()


def load_minicpm_model(model_id: str, cache_dir: Path):
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        cache_dir=str(cache_dir),
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(cache_dir), local_files_only=LOCAL_FILES_ONLY)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(cache_dir), local_files_only=LOCAL_FILES_ONLY)
    processor.image_processor.mean = processor.image_processor.mean.tolist()
    processor.image_processor.std = processor.image_processor.std.tolist()
    return {
        "adapter": "minicpm",
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "model_class": "AutoModel.chat",
    }


def load_gemma_model(model_id: str, cache_dir: Path, trust_remote_code: bool, device_map: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=LOCAL_FILES_ONLY,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    return {
        "adapter": "gemma",
        "processor": processor,
        "model": model,
        "model_class": "AutoModelForImageTextToText",
    }


def load_qwen25_model(model_id: str, cache_dir: Path, trust_remote_code: bool, adapter: str, device_map: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    adapter_id = ""
    base_model_id = model_id
    if adapter == "simula_qwen25":
        snapshot = cache_dir / "models--Qwen--Qwen2.5-VL-7B-Instruct" / "snapshots"
        snapshots = sorted(snapshot.glob("*")) if snapshot.exists() else []
        base_model_id = str(snapshots[-1]) if snapshots else "Qwen/Qwen2.5-VL-7B-Instruct"
        adapter_id = QWEN25_ADAPTER_PATH or model_id

    processor = AutoProcessor.from_pretrained(
        base_model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    if adapter_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            adapter_id,
            cache_dir=str(cache_dir),
            local_files_only=LOCAL_FILES_ONLY,
        ).eval()
    return {
        "adapter": adapter,
        "processor": processor,
        "model": model,
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "base_model_id": base_model_id,
        "adapter_id": adapter_id,
    }


def load_qwen3_vl_model(model_id: str, cache_dir: Path, trust_remote_code: bool, device_map: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=LOCAL_FILES_ONLY,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    ).eval()
    return {
        "adapter": "qwen3_vl",
        "processor": processor,
        "model": model,
        "model_class": "AutoModelForImageTextToText",
    }


def load_medgemma_peft_model(model_id: str, cache_dir: Path, trust_remote_code: bool, device_map: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base_model_id = "google/medgemma-4b-it"
    adapter_id = model_id
    processor = AutoProcessor.from_pretrained(
        base_model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
        local_files_only=LOCAL_FILES_ONLY,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(
        base,
        adapter_id,
        cache_dir=str(cache_dir),
        local_files_only=LOCAL_FILES_ONLY,
    ).eval()
    return {
        "adapter": "simula_medgemma",
        "processor": processor,
        "model": model,
        "model_class": "AutoModelForImageTextToText+PeftModel",
        "base_model_id": base_model_id,
        "adapter_id": adapter_id,
    }


def load_llava_med_model(model_id: str):
    import transformers
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    # LLaVA-Med's cached CLIP vision tower is stored as a trusted local .bin.
    # New transformers releases block torch<2.6 torch.load by default; the
    # project uses a pinned torch2.5 LLaVA-Med environment, so we bypass this
    # check only inside the LLaVA-Med local-cache loader.
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.import_utils as import_utils

        modeling_utils.check_torch_load_is_safe = lambda: None
        import_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass

    model_path = LLAVA_MODEL_PATH or model_id
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device="cuda",
    )
    def forward_with_cache_position(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
        cache_position=None,
        **kwargs: Any,
    ):
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )
        super_forward = super(model.__class__, model).forward
        return super_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    model.forward = forward_with_cache_position
    return {
        "adapter": "llava_med",
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "model_class": "llava_med_official",
        "base_model_id": model_path,
    }


def load_any_model(
    model_id: str,
    cache_dir: Path,
    trust_remote_code: bool,
    adapter: str,
    device_map: str = "auto",
    internvl_device_map: str = "single",
):
    adapter = infer_adapter(model_id, adapter)
    if adapter == "internvl":
        bundle = load_internvl_model(model_id, cache_dir, internvl_device_map)
        bundle["requested_device_map"] = internvl_device_map
        return bundle
    if adapter == "minicpm":
        bundle = load_minicpm_model(model_id, cache_dir)
        bundle["requested_device_map"] = "single"
        return bundle
    if adapter == "gemma":
        bundle = load_gemma_model(model_id, cache_dir, trust_remote_code, device_map)
        bundle["requested_device_map"] = device_map
        return bundle
    if adapter in {"qwen25", "simula_qwen25"}:
        bundle = load_qwen25_model(model_id, cache_dir, trust_remote_code, adapter, device_map)
        bundle["requested_device_map"] = device_map
        return bundle
    if adapter == "qwen3_vl":
        bundle = load_qwen3_vl_model(model_id, cache_dir, trust_remote_code, device_map)
        bundle["requested_device_map"] = device_map
        return bundle
    if adapter == "simula_medgemma":
        bundle = load_medgemma_peft_model(model_id, cache_dir, trust_remote_code, device_map)
        bundle["requested_device_map"] = device_map
        return bundle
    if adapter == "llava_med":
        bundle = load_llava_med_model(model_id)
        bundle["requested_device_map"] = "single"
        return bundle
    processor, model, model_class = load_model(model_id, cache_dir, trust_remote_code, device_map)
    return {
        "adapter": "generic",
        "processor": processor,
        "model": model,
        "model_class": model_class,
        "requested_device_map": device_map,
    }


def predict(
    bundle: Dict[str, Any],
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    qwen_max_pixels: int = 0,
) -> str:
    adapter = bundle["adapter"]
    if adapter in {"qwen25", "simula_qwen25"}:
        import torch
        from qwen_vl_utils import process_vision_info

        processor = bundle["processor"]
        model = bundle["model"]
        image_ref = (
            image
            if getattr(image, "_aris_force_pil_input", False) or getattr(image, "_aris_image_resized", False)
            else (getattr(image, "filename", None) or image)
        )
        image_content: Dict[str, Any] = {"type": "image", "image": image_ref}
        if qwen_max_pixels > 0:
            image_content["max_pixels"] = qwen_max_pixels
        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = move_inputs(inputs, first_device(model))
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids, strict=False)
        ]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    if adapter == "llava_med":
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        image_processor = bundle["image_processor"]
        conv = conv_templates["mistral_instruct"].copy()
        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        llava_prompt = prompt
        if model.config.mm_use_im_start_end:
            llava_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + llava_prompt
        else:
            llava_prompt = DEFAULT_IMAGE_TOKEN + "\n" + llava_prompt
        conv.append_message(conv.roles[0], llava_prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        if output_ids.shape[-1] > input_ids.shape[-1]:
            output_ids = output_ids[:, input_ids.shape[-1] :]
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if adapter == "qwen3_vl":
        import torch

        processor = bundle["processor"]
        model = bundle["model"]
        image_ref = (
            image
            if getattr(image, "_aris_force_pil_input", False) or getattr(image, "_aris_image_resized", False)
            else (getattr(image, "filename", None) or image)
        )
        image_content: Dict[str, Any] = {"type": "image", "image": image_ref}
        if qwen_max_pixels > 0:
            image_content["max_pixels"] = qwen_max_pixels
        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(first_device(model))
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if generated.shape[-1] > inputs["input_ids"].shape[-1]:
            generated = generated[:, inputs["input_ids"].shape[-1] :]
        return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    if adapter == "internvl":
        pixels = build_internvl_pixels(image)
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        return bundle["model"].chat(bundle["tokenizer"], pixels, prompt, generation_config)
    if adapter == "minicpm":
        msgs = [{"role": "user", "content": [image, prompt]}]
        return bundle["model"].chat(
            msgs=msgs,
            image=None,
            tokenizer=bundle["tokenizer"],
            processor=bundle["processor"],
            max_new_tokens=max_new_tokens,
            sampling=False,
        )
    if adapter == "gemma":
        import torch

        processor = bundle["processor"]
        model = bundle["model"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(first_device(model), dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()
    if adapter == "simula_medgemma":
        import torch

        processor = bundle["processor"]
        model = bundle["model"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(first_device(model), dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()

    processor = bundle["processor"]
    model = bundle["model"]
    inputs = move_inputs(build_inputs(processor, image, prompt), first_device(model))
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return decode(processor, generated, inputs)


def main() -> int:
    global LOCAL_FILES_ONLY, QWEN25_ADAPTER_PATH, LLAVA_MODEL_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Optional YAML model registry.")
    parser.add_argument("--model", help="Model key from --config.")
    parser.add_argument("--model-id")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/models"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum queries to run; 0 runs the full manifest.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--prompt-style", choices=["json", "answer_only"], default="json")
    parser.add_argument("--prompt-version", default="shared_vlm_prompt_v1")
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=DEFAULT_MAX_IMAGE_PIXELS,
        help="Resize images above this total pixel count before inference. 0 keeps original resolution.",
    )
    parser.add_argument(
        "--qwen-max-pixels",
        type=int,
        default=DEFAULT_QWEN_MAX_PIXELS,
        help="Qwen vision max_pixels value. 0 omits the field and uses model defaults.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--device-map",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        default="auto",
        help="Transformers/Accelerate device_map for generic, Qwen, Gemma, and PEFT routes.",
    )
    parser.add_argument(
        "--internvl-device-map",
        choices=["single", "split"],
        default="single",
        help="InternVL placement. 'single' preserves the old .cuda() route; 'split' uses a multi-GPU layer map.",
    )
    parser.add_argument(
        "--adapter",
        choices=[
            "auto",
            "generic",
            "internvl",
            "minicpm",
            "gemma",
            "qwen25",
            "qwen3_vl",
            "simula_medgemma",
            "simula_qwen25",
            "llava_med",
        ],
        default="auto",
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--adapter-path", default="", help="Optional local PEFT adapter path.")
    parser.add_argument("--llava-model-path", default="", help="Optional assembled LLaVA-Med model path.")
    parser.add_argument("--offline", action="store_true", help="Use only locally cached model files.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of data shards for parallel inference")
    parser.add_argument("--shard-index", type=int, default=0, help="Current data shard index, 0-based")
    parser.add_argument("--resume", action="store_true", help="Append to existing output and skip completed probe ids")
    args = parser.parse_args()

    if args.config:
        if not args.model:
            parser.error("--model is required when --config is used")
        import yaml

        registry = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        model_config = (registry.get("models") or {}).get(args.model)
        if not model_config:
            parser.error(f"Unknown model key in {args.config}: {args.model}")
        args.model_id = args.model_id or model_config.get("model_id")
        if args.adapter == "auto":
            args.adapter = model_config.get("adapter", "auto")
        if not args.adapter_path:
            args.adapter_path = model_config.get("adapter_path", "")
        if not args.llava_model_path:
            args.llava_model_path = model_config.get("llava_model_path", "")
        args.trust_remote_code = bool(args.trust_remote_code or model_config.get("trust_remote_code", False))
    if not args.model_id:
        parser.error("provide --model-id, or use --config together with --model")

    LOCAL_FILES_ONLY = bool(args.offline)
    QWEN25_ADAPTER_PATH = args.adapter_path
    LLAVA_MODEL_PATH = args.llava_model_path
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must satisfy 0 <= shard_index < num_shards")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    started = time.time()
    bundle = load_any_model(
        args.model_id,
        args.cache_dir,
        args.trust_remote_code,
        args.adapter,
        args.device_map,
        args.internvl_device_map,
    )
    rows = read_jsonl(args.input, args.limit)
    if args.num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(args.output) if args.resume else set()
    if done:
        rows = [row for row in rows if row_key(row) not in done]
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as f:
        for idx, item in enumerate(rows):
            result = dict(item)
            result["model_id"] = args.model_id
            result["model_class"] = bundle["model_class"]
            result["adapter"] = bundle["adapter"]
            result["requested_device_map"] = bundle.get("requested_device_map", args.device_map)
            result["actual_device_map"] = bundle.get("actual_device_map", bundle.get("requested_device_map", args.device_map))
            result["prompt_style"] = args.prompt_style
            result["prompt_version"] = args.prompt_version
            result["max_new_tokens"] = args.max_new_tokens
            result.update(
                image_policy_metadata(
                    None,
                    max_image_pixels=args.max_image_pixels,
                    qwen_max_pixels=args.qwen_max_pixels,
                )
            )
            try:
                image = load_image_for_probe(item, args.data_root, args.max_image_pixels)
                result.update(
                    image_policy_metadata(
                        image,
                        max_image_pixels=args.max_image_pixels,
                        qwen_max_pixels=args.qwen_max_pixels,
                    )
                )
                prompt = build_prompt(item, args.prompt_style)
                t0 = time.time()
                with torch.inference_mode():
                    pred = predict(bundle, image, prompt, args.max_new_tokens, args.qwen_max_pixels)
                result["raw_output"] = pred
                result["prediction"] = pred
                result["json_valid"] = parse_json(pred)
                result["latency_seconds"] = round(time.time() - t0, 3)
                result["error"] = ""
            except Exception as exc:
                result["raw_output"] = ""
                result["prediction"] = ""
                result["json_valid"] = False
                result["latency_seconds"] = None
                result["error"] = f"{type(exc).__name__}: {exc}"
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
    summary = {
        "model_id": args.model_id,
        "requested_device_map": bundle.get("requested_device_map", args.device_map),
        "actual_device_map": bundle.get("actual_device_map", bundle.get("requested_device_map", args.device_map)),
        "input": str(args.input),
        "output": str(args.output),
        "records": len(rows),
        "skipped_existing": len(done),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_image_pixels": args.max_image_pixels,
        "qwen_max_pixels": args.qwen_max_pixels,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
