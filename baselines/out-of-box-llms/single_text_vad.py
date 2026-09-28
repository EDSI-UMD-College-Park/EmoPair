"""One-text, zero-shot VAD inference. Imports model SDKs only when `run` is requested."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
from collections.abc import Mapping


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_INPUT = ROOT / "Results" / "emopair.csv"
DEFAULT_PROMPT = HERE / "prompts" / "single_text_vad.txt"
DEFAULT_OUTPUT = ROOT / "Results" / "zero_shot_baselines"
PLACEHOLDER = "{{text_json}}"
FIELDS = ("valence", "arousal", "dominance")
MODELS = {
    "gpt-6-luna": {"backend": "openai", "model_id": "gpt-6-luna", "reasoning_effort": "none"},
    "gpt-6-sol": {"backend": "openai", "model_id": "gpt-6-sol", "reasoning_effort": "none"},
    "gemini-3.8-flash": {"backend": "gemini", "model_id": "gemini-3.8-flash", "thinking_level": "low"},
    "gemma-4-12b-it": {"backend": "vllm", "model_id": "google/gemma-4-12B-it", "thinking": False},
    "qwen3.5-9b": {"backend": "vllm", "model_id": "Qwen/Qwen3.5-9B", "thinking": False},
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def read_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8")
    if prompt.count(PLACEHOLDER) != 1:
        raise ValueError(f"Prompt must contain {PLACEHOLDER} exactly once")
    return prompt


def make_prompt(template: str, text: str) -> str:
    return template.replace(PLACEHOLDER, json.dumps(text, ensure_ascii=False))


def load_items(path: Path, split: str, max_items: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"item_id", "split", "text"}.issubset(reader.fieldnames or []):
            raise ValueError("Input CSV requires item_id, split, and text columns")
        items = []
        seen = set()
        for row in reader:
            if row["split"] != split:
                continue
            item_id, text = row["item_id"].strip(), row["text"]
            if not item_id or not text or not text.strip():
                raise ValueError("Selected row has an empty item_id or text")
            if item_id in seen:
                raise ValueError(f"Duplicate selected item_id: {item_id}")
            seen.add(item_id)
            items.append({"item_id": item_id, "split": split, "text": text})
            if max_items is not None and len(items) >= max_items:
                break
    if not items:
        raise ValueError(f"No items found for split={split}")
    return items


def parse_scores(raw: str) -> dict[str, float]:
    try:
        value = json.loads(raw.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("Response is not a JSON object") from exc
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise ValueError("Response must contain exactly valence, arousal, dominance")
    scores = {}
    for field in FIELDS:
        score = value[field]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{field} is not numeric")
        if not math.isfinite(score) or not 1.0 <= score <= 5.0:
            raise ValueError(f"{field} is outside [1, 5]")
        scores[field] = float(score)
    return scores


def identity(args: argparse.Namespace, items: list[dict[str, str]], template: str) -> dict[str, Any]:
    model = MODELS[args.model]
    return {
        "protocol": "single-text-vad-zero-shot-v1",
        "model_key": args.model,
        "model": model,
        "requested_revision": args.revision if model["backend"] == "vllm" else None,
        "split": args.split,
        "max_items": args.max_items,
        "input_path": str(args.input.resolve()),
        "selected_input_sha256": digest(canonical(items)),
        "prompt_path": str(args.prompt.resolve()),
        "prompt_sha256": digest(template.encode("utf-8")),
        "inference_code_sha256": digest(Path(__file__).read_bytes()),
        "item_count": len(items),
        "decoding": {"temperature": 0, "max_output_tokens": 256} if model["backend"] != "gemini"
                    else {"thinking_level": "low", "sampling": "Gemini 3.8 Flash default"},
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def output_dir(args: argparse.Namespace) -> Path:
    return args.output_root / args.model / args.split


def item_path(directory: Path, item_id: str) -> Path:
    return directory / "items" / (digest(item_id.encode("utf-8")) + ".json")


def checked_manifest(directory: Path, planned: dict[str, Any]) -> dict[str, Any] | None:
    path = directory / "manifest.json"
    if not path.exists():
        if (directory / "items").exists() and any((directory / "items").iterdir()):
            raise ValueError("Output has item files but no manifest")
        return None
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("identity") != planned:
        raise ValueError(f"Run identity differs from {path}; choose a new output root")
    return existing


def software_versions(backend: str) -> dict[str, str]:
    packages = {"openai": ("openai",), "gemini": ("google-genai",),
                "vllm": ("vllm", "transformers", "tokenizers", "huggingface-hub")}[backend]
    try:
        versions = {name: importlib.metadata.version(name) for name in packages}
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Missing inference dependency: {exc}") from exc
    return {"python": sys.version.split()[0], **versions}


class OpenAIBackend:
    def __init__(self, model: dict[str, Any]):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required")
        if os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") not in (
            "https://api.openai.com/v1", "https://us.api.openai.com/v1"
        ):
            raise RuntimeError("OPENAI_BASE_URL must use the official OpenAI API endpoint")
        self.model = model
        self.client = OpenAI(max_retries=2, timeout=120.0)

    def generate(self, prompts: list[str]) -> list[tuple[str, dict[str, Any]]]:
        results = []
        for prompt in prompts:
            response = self.client.responses.create(
                model=self.model["model_id"],
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                reasoning={"effort": self.model["reasoning_effort"]},
                temperature=0,
                top_p=1,
                max_output_tokens=256,
                truncation="disabled",
                store=False,
            )
            if response.status != "completed":
                raise RuntimeError(f"OpenAI response status: {response.status}")
            results.append((response.output_text or "", {
                "response_id": response.id,
                "returned_model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
            }))
        return results


class GeminiBackend:
    def __init__(self, model: dict[str, Any]):
        from google import genai

        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
        self.model = model
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])

    def generate(self, prompts: list[str]) -> list[tuple[str, dict[str, Any]]]:
        results = []
        for prompt in prompts:
            interaction = self.client.interactions.create(
                model=self.model["model_id"],
                input=prompt,
                generation_config={"thinking_level": self.model["thinking_level"]},
                store=False,
            )
            if getattr(interaction, "status", "completed") != "completed":
                raise RuntimeError(f"Gemini interaction status: {interaction.status}")
            results.append((interaction.output_text or "", {
                "interaction_id": getattr(interaction, "id", None),
                "returned_model": getattr(interaction, "model", None),
                "usage": interaction.usage.model_dump() if getattr(interaction, "usage", None) else None,
            }))
        return results


class VllmBackend:
    def __init__(self, model: dict[str, Any], revision: str | None, batch_size: int):
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model = model
        self.revision = (revision if revision and re.fullmatch(r"[0-9a-f]{40}", revision)
                         else HfApi().model_info(model["model_id"], revision=revision).sha)
        snapshot = snapshot_download(model["model_id"], revision=self.revision)
        self.snapshot = Path(snapshot)
        errors = []
        for loader in (AutoTokenizer, AutoProcessor):
            try:
                obj = loader.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
                obj.apply_chat_template([{"role": "user", "content": "ping"}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False)
                self.template = obj
                break
            except Exception as exc:
                errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")
        else:
            raise RuntimeError("Could not load official chat template: " + "; ".join(errors))
        self.llm = LLM(
            model=str(self.snapshot), tokenizer=str(self.snapshot), dtype="bfloat16",
            tensor_parallel_size=1, max_model_len=16384, max_num_seqs=batch_size,
            max_num_batched_tokens=16384, gpu_memory_utilization=0.85,
            trust_remote_code=False, language_model_only=True, skip_mm_profiling=True,
            generation_config="vllm",
        )
        self.params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=256, seed=0)

    def generate(self, prompts: list[str]) -> list[tuple[str, dict[str, Any]]]:
        requests = []
        for prompt in prompts:
            ids = self.template.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
            )
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if len(ids) + 256 > 16384:
                raise ValueError("Prompt exceeds configured model context; refusing truncation")
            requests.append({"prompt_token_ids": ids})
        outputs = self.llm.generate(requests, self.params, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned the wrong number of outputs")
        results = []
        for output in outputs:
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM returned multiple completions")
            choice = output.outputs[0]
            results.append((choice.text, {
                "returned_model": self.model["model_id"],
                "resolved_revision": self.revision,
                "finish_reason": choice.finish_reason,
                "input_tokens": len(output.prompt_token_ids or []),
                "output_tokens": len(choice.token_ids),
            }))
        return results


def backend_for(model: dict[str, Any], manifest: dict[str, Any], batch_size: int):
    if model["backend"] == "openai":
        return OpenAIBackend(model)
    if model["backend"] == "gemini":
        return GeminiBackend(model)
    return VllmBackend(model, manifest.get("resolved_revision") or
                       manifest["identity"]["requested_revision"], batch_size)


def run(args: argparse.Namespace, items: list[dict[str, str]], template: str, planned: dict[str, Any]) -> None:
    if not args.execute:
        raise ValueError("Inference requires the explicit --execute flag")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    directory = output_dir(args)
    manifest = checked_manifest(directory, planned)
    pending = []
    for item in items:
        path = item_path(directory, item["item_id"])
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("item_id") != item["item_id"] or saved.get("text_sha256") != digest(item["text"].encode("utf-8")):
                raise ValueError(f"Saved prediction does not match input: {path}")
        else:
            pending.append(item)
    if not pending:
        print(f"All {len(items)} item outputs already exist in {directory}")
        return
    software = software_versions(MODELS[args.model]["backend"])
    if manifest is not None and manifest.get("software") != software:
        raise ValueError("Installed inference software differs from the saved run manifest")
    backend = backend_for(MODELS[args.model], manifest or {"identity": planned}, args.batch_size)
    if manifest is None:
        manifest = {"identity": planned, "resolved_revision": getattr(backend, "revision", None),
                    "software": software}
        atomic_json(directory / "manifest.json", manifest)
    # Cloud calls are sequential; commit each one before issuing the next so an
    # interrupted run does not resend earlier completed requests in a batch.
    write_batch_size = 1 if MODELS[args.model]["backend"] != "vllm" else args.batch_size
    for start in range(0, len(pending), write_batch_size):
        batch = pending[start:start + write_batch_size]
        outputs = backend.generate([make_prompt(template, item["text"]) for item in batch])
        if len(outputs) != len(batch):
            raise RuntimeError("Backend returned the wrong number of outputs")
        for item, (raw, metadata) in zip(batch, outputs):
            try:
                scores = parse_scores(raw)
                status, parse_error = "ok", None
            except ValueError as exc:
                scores, status, parse_error = None, "invalid", str(exc)
            if metadata.get("finish_reason") == "length":
                scores, status, parse_error = None, "invalid", "Generation reached the output token limit"
            atomic_json(item_path(directory, item["item_id"]), {
                "item_id": item["item_id"], "split": item["split"],
                "text_sha256": digest(item["text"].encode("utf-8")),
                "prompt_sha256": digest(make_prompt(template, item["text"]).encode("utf-8")),
                "requested_model": MODELS[args.model]["model_id"],
                "raw_output": raw, "scores": scores, "parse_status": status,
                "parse_error": parse_error, "metadata": metadata,
            })
        print(f"Saved {min(start + len(batch), len(pending))}/{len(pending)} pending items", flush=True)
    print(f"Inference outputs: {directory}")


def export(args: argparse.Namespace, items: list[dict[str, str]], planned: dict[str, Any]) -> None:
    directory = output_dir(args)
    if checked_manifest(directory, planned) is None:
        raise ValueError(f"No run manifest in {directory}")
    rows = []
    for item in items:
        path = item_path(directory, item["item_id"])
        if not path.exists():
            raise ValueError(f"Missing prediction for {item['item_id']}; export requires a complete run")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("item_id") != item["item_id"] or result.get("text_sha256") != digest(item["text"].encode("utf-8")):
            raise ValueError(f"Saved prediction does not match input: {path}")
        scores = result["scores"] or {}
        rows.append({"item_id": item["item_id"], "split": item["split"],
                     "model": args.model, "parse_status": result["parse_status"],
                     "valence": scores.get("valence"), "arousal": scores.get("arousal"),
                     "dominance": scores.get("dominance")})
    target = directory / "predictions.csv"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=directory, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "split", "model", "parse_status", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(target)
    print(f"Exported {len(rows)} predictions to {target}")


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "export"))
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-items", type=int, help="Optional first N items; use a separate output root for pilots")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--revision", help="Optional Hugging Face commit for an open-weight model")
    parser.add_argument("--execute", action="store_true", help="Required for run; allows model/API inference")
    args = parser.parse_args(argv)
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive")
    if args.revision and MODELS[args.model]["backend"] != "vllm":
        parser.error("--revision applies only to open-weight models")
    template = read_prompt(args.prompt)
    items = load_items(args.input, args.split, args.max_items)
    planned = identity(args, items, template)
    if args.command == "plan":
        print(json.dumps({"identity": planned, "output_dir": str(output_dir(args))}, indent=2, ensure_ascii=False))
    elif args.command == "run":
        run(args, items, template, planned)
    else:
        export(args, items, planned)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(cli())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
