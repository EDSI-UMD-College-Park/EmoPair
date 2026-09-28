"""Model-native, single-text EmoBank evaluation with vLLM (BF16)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_INPUT = ROOT / "Data" / "emobank.csv"
DEFAULT_OUTPUT = ROOT / "Results" / "emotion_finetuned_baselines"
PROTOCOL = "model-native-vllm-bf16-v1"
EMOTIONS = ("Anger", "Anxiety", "Fear", "Sadness", "Disgust", "Optimism", "Excitement", "Surprise")
MODELS = {
    "bagel": {"repo": "jondurbin/bagel-7b-v0.4", "max_tokens": 128, "dimensions": ("V", "A", "D")},
    "emoloom": {"repo": "Lixeeone/Emoloom-2B", "max_tokens": 128, "dimensions": ("V", "A", "D")},
    "openemotions": {"repo": "fabozzi/OpenEmotions-Mistral-24B-Instruct", "max_tokens": 384, "dimensions": ("V", "A")},
}
BASES = {"emoloom": "Qwen/Qwen1.5-1.8B-Chat", "openemotions": "mistralai/Mistral-Small-24B-Instruct-2501"}
SHA = re.compile(r"[0-9a-f]{40}\Z")
STOP_REQUESTED = False


class InputTooLong(ValueError):
    """Input and generation budget do not fit the configured context."""


def request_stop(_signum: int, _frame: Any) -> None:
    """Finish the current item, then let the Slurm wrapper requeue the job."""
    global STOP_REQUESTED
    STOP_REQUESTED = True


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        temp = Path(f.name)
    temp.replace(path)


def read_template(name: str) -> str:
    return (HERE / "prompts" / f"{name}.txt").read_text(encoding="utf-8").rstrip("\n")


def prompt_messages(model: str, text: str) -> list[dict[str, str]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    # Split once on a fixed marker. Never run substitutions over user text.
    if model == "emoloom":
        return [
            {"role": "system", "content": read_template("emoloom_system")},
            {"role": "user", "content": read_template("emoloom_user").replace("%input_text", text, 1)},
        ]
    if model == "openemotions":
        template = read_template("openemotions")
        head, tail = template.split("%input_text", 1)
        tail = tail.replace("%emotions_list", ", ".join(EMOTIONS), 1)
        content = head + text + tail
    else:
        template = read_template("bagel")
        head, tail = template.split("%input_text", 1)
        content = head + text + tail
    return [{"role": "user", "content": content}]


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def parse_generation(model: str, raw: str, finish_reason: str | None = None) -> dict[str, Any]:
    result = {"raw_json": None, "json_parse_ok": False, "schema_ok": False,
              "vad_valid": False, "parse_mode": "strict", "status": "invalid_json",
              "error": None, "native_scores": None, "scores_01": None,
              "supported_dimensions": list(MODELS[model]["dimensions"])}
    if finish_reason == "length":
        result.update(status="length", error="generation reached max_tokens")
        return result
    if not raw or not raw.strip():
        result.update(status="empty", error="empty generation")
        return result
    try:
        obj = json.loads(raw.strip(), object_pairs_hook=unique_pairs, parse_float=finite_float,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}")))
    except (ValueError, TypeError) as exc:
        result["error"] = str(exc)
        return result
    result["json_parse_ok"] = True
    result["raw_json"] = obj
    if not isinstance(obj, dict):
        result.update(status="invalid_schema", error="top-level JSON must be an object")
        return result
    if model == "bagel":
        schema_ok = set(obj) == {"V", "A", "D"}
        values = {key: obj.get(key) for key in ("V", "A", "D")}
    elif model == "emoloom":
        vad = obj.get("vad")
        schema_ok = (set(obj) == {"labels", "vad", "rationale"}
                     and isinstance(obj["labels"], list)
                     and all(isinstance(x, str) for x in obj["labels"])
                     and isinstance(obj["rationale"], str)
                     and isinstance(vad, dict) and set(vad) == {"v", "a", "d"})
        values = {key: vad.get(key.lower()) for key in ("V", "A", "D")} if isinstance(vad, dict) else {}
    else:
        required = set(EMOTIONS) | {"Valence", "Arousal"}
        schema_ok = set(obj) == required
        values = {"V": obj.get("Valence"), "A": obj.get("Arousal")}
        # The prompt requests all ten numeric fields, even though only V/A enter EmoBank metrics.
        if schema_ok:
            values.update({key: obj[key] for key in EMOTIONS})
    if not schema_ok:
        result.update(status="invalid_schema", error="unexpected or missing JSON fields")
        return result
    result["schema_ok"] = True
    bounds = {"bagel": {"V": (1, 5), "A": (1, 5), "D": (1, 5)},
              "emoloom": {"V": (0, 1), "A": (0, 1), "D": (0, 1)},
              "openemotions": {"V": (-100, 100), "A": (0, 100)}}[model]
    if model == "openemotions":
        bounds.update({key: (0, 100) for key in EMOTIONS})
    native = {}
    for key, (lo, hi) in bounds.items():
        value = values[key]
        if model == "bagel" and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            result.update(status="invalid_value", error=f"{key} is not a finite number")
            return result
        if not lo <= value <= hi:
            result.update(status="invalid_value", error=f"{key} outside [{lo}, {hi}]")
            return result
        if key in ("V", "A", "D"):
            native[key] = float(value)
    if model == "bagel":
        scores = {key: (val - 1) / 4 for key, val in native.items()}
    elif model == "emoloom":
        scores = dict(native)
    else:
        scores = {"V": (native["V"] + 100) / 200, "A": native["A"] / 100, "D": None}
        native["D"] = None
    result.update(vad_valid=True, status="ok", native_scores=native, scores_01=scores)
    return result


def load_items(path: Path, split: str, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        id_field = "item_id" if "item_id" in fields else "id"
        if not {id_field, "split", "text", "V", "A", "D"} <= fields:
            raise ValueError("input needs id/item_id, split, text, V, A, D columns")
        rows, seen = [], set()
        for row in reader:
            if row["split"] != split:
                continue
            item_id, value = row[id_field].strip(), row["text"]
            if not item_id or not value.strip() or item_id in seen:
                raise ValueError("empty or duplicate item_id/text")
            seen.add(item_id)
            gold = {k: float(row[k]) for k in ("V", "A", "D")}
            if any(not math.isfinite(x) or not 1 <= x <= 5 for x in gold.values()):
                raise ValueError(f"invalid gold VAD for {item_id}")
            rows.append({"item_id": item_id, "split": split, "text": value, "gold": gold})
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no items in split={split}")
    return rows


def overlap_report(items: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"id", "split", "text"} <= set(reader.fieldnames or []):
            raise ValueError("overlap source must be official-style EmoBank CSV")
        train = [row for row in reader if row["split"] == "train"]
    ids = {row["id"] for row in train}
    texts = {row["text"] for row in train}
    cleaned = {row["text"].strip().rstrip('"').rstrip() for row in train}
    return {"source_path": str(source.resolve()), "source_sha256": file_hash(source),
            "official_train_rows": len(train), "selected_rows": len(items),
            "id_overlap": sum(x["item_id"] in ids for x in items),
            "raw_text_overlap": sum(x["text"] in texts for x in items),
            "bagel_cleaned_text_overlap": sum(x["text"].strip().rstrip('"').rstrip() in cleaned for x in items)}


def resolve_repo(repo: str, revision: str | None = None) -> dict[str, Any]:
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    except Exception as exc:
        raise RuntimeError(f"cannot access Hugging Face metadata for {repo}: {type(exc).__name__}") from None
    if not info.sha or not SHA.fullmatch(info.sha):
        raise RuntimeError(f"Hub did not return a commit SHA for {repo}")
    return {"repo": repo, "revision": info.sha, "files": sorted(x.rfilename for x in info.siblings),
            "gated": info.gated}


def audit_repositories(model: str, revisions: dict[str, str | None]) -> dict[str, Any]:
    repo = resolve_repo(MODELS[model]["repo"], revisions.get("model"))
    files = set(repo["files"])
    if model == "bagel" and ("config.json" not in files or not any(x.startswith("tokenizer") for x in files)
                              or not any(x.endswith(".safetensors") for x in files)):
        raise RuntimeError("Bagel is missing config or tokenizer metadata")
    if model == "emoloom" and not any(x.endswith(".safetensors") for x in files):
        raise RuntimeError("Emoloom has no safetensors weights")
    if model == "openemotions" and not {"adapter_config.json", "adapter_model.safetensors"} <= files:
        raise RuntimeError("OpenEmotions adapter files missing or access denied")
    result = {"model": repo}
    if model in BASES:
        result["base"] = resolve_repo(BASES[model], revisions.get("base"))
    if model == "openemotions":
        # Read the small config before downloading the base or adapter weights.
        from huggingface_hub import hf_hub_download
        try:
            config_file = Path(hf_hub_download(repo["repo"], "adapter_config.json", revision=repo["revision"]))
        except Exception as exc:
            raise RuntimeError(f"cannot access OpenEmotions adapter configuration: {type(exc).__name__}") from None
        adapter_config = json.loads(config_file.read_text(encoding="utf-8"))
        if adapter_config.get("base_model_name_or_path") != BASES[model]:
            raise RuntimeError("OpenEmotions adapter base differs from model card; manual investigation required")
        result["adapter_config_sha256"] = file_hash(config_file)
    return result


def download(repo_info: dict[str, Any], patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo_info["repo"], revision=repo_info["revision"],
                                      allow_patterns=patterns, max_workers=1))
    except Exception as exc:
        raise RuntimeError(f"cannot download {repo_info['repo']} at pinned revision: {type(exc).__name__}") from None


def safe_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open
    shapes = {}
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if key in shapes:
                    raise RuntimeError(f"duplicate tensor key: {key}")
                shapes[key] = tuple(tensors.get_slice(key).get_shape())
    if not shapes:
        raise RuntimeError(f"no safetensors weights in {path}")
    return shapes


def check_full_checkpoint(weights: Path, metadata: Path, tokenizer: Any) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(metadata, local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        expected = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    expected_shapes = {key: tuple(value.shape) for key, value in expected.state_dict().items()}
    actual_shapes = safe_shapes(weights)
    missing = sorted(set(expected_shapes) - set(actual_shapes))
    unexpected = sorted(set(actual_shapes) - set(expected_shapes))
    mismatched = sorted(key for key in set(actual_shapes) & set(expected_shapes)
                        if actual_shapes[key] != expected_shapes[key])
    if missing or unexpected or mismatched:
        raise RuntimeError(f"weight/config mismatch: missing={missing[:5]}, "
                           f"unexpected={unexpected[:5]}, mismatched={mismatched[:5]}")
    if "model.embed_tokens.weight" not in actual_shapes:
        raise RuntimeError("embedding tensor missing")
    vocab = actual_shapes["model.embed_tokens.weight"][0]
    if max(tokenizer.get_vocab().values()) >= vocab:
        raise RuntimeError("tokenizer token ID exceeds embedding rows")
    del expected
    return {"tensor_count": len(actual_shapes), "embedding_rows": vocab}


def check_lora_shapes(weights: Path, config: dict[str, Any]) -> dict[str, Any]:
    shapes = safe_shapes(weights)
    rank = int(config["r"])
    targets = set(config.get("target_modules") or [])
    if rank < 1 or not targets:
        raise RuntimeError("LoRA rank or target modules missing")
    pairs = {}
    for key, shape in shapes.items():
        match = re.search(r"\.(lora_[AB])(?:\.default)?\.weight$", key)
        if not match or not any(f".{target}." in key for target in targets):
            raise RuntimeError(f"unexpected LoRA tensor key: {key}")
        part = match.group(1)
        root = key[:match.start()]
        if len(shape) != 2 or shape[0 if part == "lora_A" else 1] != rank:
            raise RuntimeError(f"LoRA tensor rank mismatch: {key}")
        pairs.setdefault(root, {})[part] = shape
    if any(set(pair) != {"lora_A", "lora_B"} or pair["lora_A"][0] != pair["lora_B"][1]
           for pair in pairs.values()):
        raise RuntimeError("LoRA A/B tensor pair mismatch")
    return {"adapter_tensor_count": len(shapes), "adapter_module_count": len(pairs), "lora_rank": rank}


class VllmAdapter:
    def __init__(self, model: str, audit: dict[str, Any], max_model_len: int,
                 gpu_memory: float, tensor_parallel_size: int):
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        from vllm import LLM, SamplingParams
        self.model = model
        self.tempdir = None
        self.lora_request = None
        self.checks = {}
        repo_snapshot = download(
            audit["model"],
            ["adapter_config.json", "adapter_model.safetensors", "tokenizer*", "vocab*", "merges.txt", "*.model", "special_tokens_map.json"]
            if model == "openemotions" else None)
        if model == "bagel":
            model_path = tokenizer_path = repo_snapshot
            tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
            self.checks = check_full_checkpoint(repo_snapshot, repo_snapshot, tok)
        elif model == "emoloom":
            base = download(audit["base"], ["*.json", "tokenizer*", "vocab*", "merges.txt", "*.model"])
            tokenizer_path = base
            tok = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
            self.checks = check_full_checkpoint(repo_snapshot, base, tok)
            self.checks["metadata_reconstructed_from"] = BASES["emoloom"]
            # vLLM requires config.json alongside weights. These links contain only
            # frozen base metadata and the separately verified fine-tuned weights.
            self.tempdir = tempfile.TemporaryDirectory(prefix="emoloom_vllm_")
            model_path = Path(self.tempdir.name)
            for source in base.iterdir():
                if source.is_file():
                    (model_path / source.name).symlink_to(source.resolve())
            for source in repo_snapshot.glob("*.safetensors"):
                (model_path / source.name).symlink_to(source.resolve())
        else:
            from vllm.lora.request import LoRARequest
            base = download(audit["base"])
            config_path = repo_snapshot / "adapter_config.json"
            adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
            if adapter_config.get("base_model_name_or_path") != BASES[model]:
                raise RuntimeError("OpenEmotions adapter base does not match the model card")
            if adapter_config.get("peft_type") != "LORA":
                raise RuntimeError("OpenEmotions adapter is not a nonempty LoRA checkpoint")
            tokenizer_path = repo_snapshot if any((repo_snapshot / x).exists() for x in ("tokenizer.json", "tokenizer.model")) else base
            self.checks = {"adapter_config_sha256": file_hash(config_path),
                           **check_lora_shapes(repo_snapshot, adapter_config),
                           "tokenizer_source": "adapter" if tokenizer_path == repo_snapshot else "base_fallback_missing_adapter_tokenizer"}
            model_path = base
            self.lora_request = LoRARequest("openemotions", 1, str(repo_snapshot))
        if model == "openemotions" and tokenizer_path == repo_snapshot:
            from tokenizers import Tokenizer
            adapter_vocab = Tokenizer.from_file(str(repo_snapshot / "tokenizer.json")).get_vocab()
            base_vocab = Tokenizer.from_file(str(base / "tokenizer.json")).get_vocab()
            if adapter_vocab != base_vocab:
                raise RuntimeError("OpenEmotions adapter and base tokenizer token IDs differ")
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                tokenizer_path, local_files_only=True, trust_remote_code=False)
            probe = '{\n  "Anger": 0\n}'
            roundtrip = self.tokenizer.decode(
                self.tokenizer.encode(probe, add_special_tokens=False),
                skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if roundtrip != probe:
                raise RuntimeError("OpenEmotions fast tokenizer fails whitespace roundtrip")
            self.checks["tokenizer_loader"] = "adapter_tokenizer_json_fast"
            self.checks["tokenizer_vocab_matches_base"] = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True, trust_remote_code=False)
            self.checks["tokenizer_loader"] = type(self.tokenizer).__name__
        if not self.tokenizer.chat_template:
            raise RuntimeError("tokenizer has no chat template")
        self.checks["tokenizer_config_sha256"] = file_hash(tokenizer_path / "tokenizer_config.json")
        self.checks["model_config_sha256"] = file_hash(model_path / "config.json")
        self.checks["weight_sha256"] = {p.name: file_hash(p) for p in repo_snapshot.glob("*.safetensors")}
        opts = {"model": str(model_path), "tokenizer": str(tokenizer_path), "dtype": "bfloat16",
                "tensor_parallel_size": tensor_parallel_size, "max_model_len": max_model_len, "max_num_seqs": 1,
                "gpu_memory_utilization": gpu_memory, "trust_remote_code": False,
                "generation_config": "vllm"}
        if model == "openemotions":
            opts.update(enable_lora=True, max_loras=1, max_lora_rank=int(adapter_config["r"]),
                        attention_config={"backend": "TRITON_ATTN"})
            self.checks["attention_backend_requested"] = "TRITON_ATTN"
        self.llm = LLM(**opts)
        self.params = SamplingParams(n=1, temperature=0, top_p=1,
                                     max_tokens=MODELS[model]["max_tokens"], seed=0)
        self.max_model_len = max_model_len

    def predict_one(self, text: str) -> dict[str, Any]:
        messages = prompt_messages(self.model, text)
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, list) or not ids or not all(isinstance(i, int) for i in ids):
            raise RuntimeError("chat template did not produce a nonempty integer token ID sequence")
        if len(ids) + MODELS[self.model]["max_tokens"] > self.max_model_len:
            raise InputTooLong("prompt plus generation budget exceeds max_model_len; no truncation applied")
        started = time.monotonic()
        outputs = self.llm.generate([{"prompt_token_ids": ids}], self.params, use_tqdm=False,
                                    lora_request=self.lora_request)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("vLLM returned an unexpected number of outputs")
        generated = outputs[0].outputs[0]
        raw = (self.tokenizer.decode(generated.token_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
               if self.model == "openemotions" else generated.text)
        parsed = parse_generation(self.model, raw, generated.finish_reason)
        if generated.finish_reason not in ("stop", "length"):
            parsed.update(status="unfinished", vad_valid=False, error=f"finish_reason={generated.finish_reason}",
                          native_scores=None, scores_01=None)
        return {"raw_generation": raw, "vllm_text": generated.text if raw != generated.text else None,
                "generated_token_ids": list(generated.token_ids), **parsed,
                "messages": messages, "finish_reason": generated.finish_reason,
                "stop_reason": generated.stop_reason, "prompt_token_count": len(ids),
                "generated_token_count": len(generated.token_ids), "elapsed_seconds": time.monotonic() - started,
                "prompt_sha256": digest(json.dumps(messages, ensure_ascii=False).encode("utf-8"))}


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denom = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return sum(x*y for x, y in zip(dx, dy)) / denom if denom else None


def summarize(items: list[dict[str, Any]], records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    by_id = {r["item_id"]: r for r in records}
    if len(by_id) != len(items):
        raise ValueError("one result per selected item is required")
    summary = {"total": len(items), "status_counts": dict(Counter(r["status"] for r in records)),
               "coverage": sum(r["status"] == "ok" for r in records) / len(items), "metrics_01": {}}
    for dim in MODELS[model]["dimensions"]:
        valid = [(by_id[item["item_id"]]["scores_01"][dim], (item["gold"][dim] - 1) / 4)
                 for item in items if by_id[item["item_id"]]["status"] == "ok"]
        xs, ys = [p[0] for p in valid], [p[1] for p in valid]
        summary["metrics_01"][dim] = {"n": len(valid),
                                       "mae": sum(abs(x-y) for x, y in valid) / len(valid) if valid else None,
                                       "pearson": pearson(xs, ys)}
    return summary


def export_csv(path: Path, items: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    by_id = {r["item_id"]: r for r in records}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        fields = ["item_id", "split", "model", "status", "V_01", "A_01", "D_01", "gold_V_01", "gold_A_01", "gold_D_01"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in items:
            r = by_id[item["item_id"]]
            scores = r["scores_01"] or {}
            writer.writerow({"item_id": item["item_id"], "split": item["split"], "model": r["model_id"],
                             "status": r["status"], **{f"{k}_01": scores.get(k) for k in ("V", "A", "D")},
                             **{f"gold_{k}_01": (item["gold"][k] - 1) / 4 for k in ("V", "A", "D")}})
        temp = Path(f.name)
    temp.replace(path)


def pending_items(items: list[dict[str, Any]], out: Path) -> list[tuple[dict[str, Any], Path]]:
    """Find unsaved items and reject stale/corrupt checkpoints before resuming."""
    pending = []
    for item in items:
        path = out / "items" / f"{digest(item['item_id'].encode())}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["item_id"] != item["item_id"] or saved["text_sha256"] != digest(item["text"].encode()):
                raise ValueError("saved item does not match input")
        else:
            pending.append((item, path))
    return pending


def cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "audit", "run", "export", "overlap"))
    p.add_argument("--model", choices=tuple(MODELS), required=True)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--split", default="test")
    p.add_argument("--max-items", type=int)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--source-emobank", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--model-revision")
    p.add_argument("--base-revision")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--full", action="store_true", help="permit all selected rows during run")
    args = p.parse_args(argv)
    if args.max_items is not None and args.max_items < 1:
        p.error("--max-items must be positive")
    if args.model_revision and not SHA.fullmatch(args.model_revision):
        p.error("--model-revision must be a 40-character commit SHA")
    if args.base_revision and not SHA.fullmatch(args.base_revision):
        p.error("--base-revision must be a 40-character commit SHA")
    if not 0 < args.gpu_memory_utilization < 1 or args.max_model_len < 512 or args.tensor_parallel_size < 1:
        p.error("invalid vLLM memory or context setting")
    items = load_items(args.input, args.split, args.max_items)
    templates = {name: digest((HERE / "prompts" / f"{name}.txt").read_bytes())
                 for name in ("bagel", "emoloom_system", "emoloom_user", "openemotions")}
    identity = {"protocol_id": PROTOCOL, "model": args.model, "split": args.split,
                "input_sha256": file_hash(args.input), "selected_ids_sha256": digest(json.dumps([x["item_id"] for x in items]).encode()),
                "max_items": args.max_items, "max_model_len": args.max_model_len,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "model_revision_request": args.model_revision, "base_revision_request": args.base_revision,
                "prompt_sha256": templates, "code_sha256": file_hash(Path(__file__)),
                "dtype": "bfloat16", "quantization": "none (BF16 weights)",
                "generation_config": {"temperature": 0, "top_p": 1, "n": 1,
                                      "max_tokens": MODELS[args.model]["max_tokens"]}}
    out = args.output_root / args.model / args.split
    if args.command == "plan":
        print(json.dumps({"identity": identity, "selected_count": len(items), "output_dir": str(out)}, indent=2))
        return 0
    if args.command == "overlap":
        print(json.dumps(overlap_report(items, args.source_emobank), indent=2))
        return 0
    if args.command == "audit":
        audit = audit_repositories(args.model, {"model": args.model_revision, "base": args.base_revision})
        print(json.dumps(audit, indent=2))
        return 0
    manifest_path = out / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if existing and existing["identity"] != identity:
        raise ValueError("existing run identity differs; select a new output directory")
    if args.command == "export":
        if not existing:
            raise ValueError("run manifest missing")
        records = []
        for item in items:
            path = out / "items" / f"{digest(item['item_id'].encode())}.json"
            if not path.exists():
                raise ValueError(f"missing saved item {item['item_id']}")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["item_id"] != item["item_id"] or record["text_sha256"] != digest(item["text"].encode()):
                raise ValueError("saved item does not match input")
            records.append(record)
        summary = summarize(items, records, args.model)
        atomic_json(out / "summary.json", summary)
        export_csv(out / "predictions.csv", items, records)
        print(json.dumps(summary, indent=2))
        return 0
    if not args.execute or (args.max_items is None and not args.full):
        raise ValueError("run needs --execute and either --max-items N or --full")
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    pending = pending_items(items, out)
    if not pending:
        print("all selected items already saved; run export")
        return 0
    audit = existing["repositories"] if existing else audit_repositories(
        args.model, {"model": args.model_revision, "base": args.base_revision})
    backend = VllmAdapter(args.model, audit, args.max_model_len,
                          args.gpu_memory_utilization, args.tensor_parallel_size)
    if not existing:
        versions = {name: importlib.metadata.version(name) for name in
                    ("vllm", "transformers", "huggingface-hub", "safetensors", "torch")}
        manifest = {"identity": identity, "repositories": audit, "checks": backend.checks,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "software": versions, "python": sys.version.split()[0],
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "hardware": __import__("torch").cuda.get_device_name(0),
                    "kv_cache_note": "vLLM uses KV cache; Emoloom author's use_cache=False is not reproduced"}
        atomic_json(manifest_path, manifest)
    if STOP_REQUESTED:
        print("stop requested before inference; saved outputs remain resumable", flush=True)
        return 75
    for index, (item, path) in enumerate(pending, 1):
        try:
            prediction = backend.predict_one(item["text"])
        except InputTooLong as exc:
            prediction = {"raw_generation": "", "raw_json": None, "json_parse_ok": False,
                          "schema_ok": False, "vad_valid": False, "parse_mode": "strict",
                          "status": "input_too_long", "error": str(exc), "native_scores": None,
                          "scores_01": None, "supported_dimensions": list(MODELS[args.model]["dimensions"]),
                          "messages": prompt_messages(args.model, item["text"]),
                          "finish_reason": None, "stop_reason": None}
        atomic_json(path, {"item_id": item["item_id"], "split": item["split"],
                           "text_sha256": digest(item["text"].encode()), "model_id": MODELS[args.model]["repo"],
                           "model_revision": audit["model"]["revision"],
                           "base_revision": audit.get("base", {}).get("revision"),
                           "adapter_revision": audit["model"]["revision"] if args.model == "openemotions" else None,
                           "protocol_id": PROTOCOL, "dtype": "bfloat16", "quantization": "none",
                           **prediction})
        print(f"saved {index}/{len(pending)}", flush=True)
        if STOP_REQUESTED:
            print("stop requested; current item saved, asking Slurm to requeue", flush=True)
            return 75
    print(f"run complete; export with the same CLI options from {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(cli())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
