#!/usr/bin/env bash
# Submit through submit_slurm.py so partition/GPU constraints are set together.
#SBATCH --signal=B:USR1@120
set -Eeuo pipefail

# Slurm copies batch scripts into its spool directory; BASH_SOURCE points there.
repo_root="${EMOPAIR_REPO_ROOT:?submit through submit_slurm.py to set EMOPAIR_REPO_ROOT}"
[[ -f "$repo_root/baselines/emotion-finetuned-llms/revisions.json" ]] || {
    echo "project revision manifest is missing" >&2
    exit 2
}
runner="$repo_root/baselines/emotion-finetuned-llms/emobank_vllm.py"
output_root="${EMOPAIR_OUTPUT_ROOT:-$repo_root/Results/emotion_finetuned_baselines}"

set -a
source "${EMOPAIR_ENV_FILE:-$HOME/.env}"
set +a
unset OPENAI_API_KEY
: "${HF_TOKEN:?HF_TOKEN must be set in the environment file}"
: "${HF_HOME:?HF_HOME must be set in the environment file}"
[[ "$HF_HOME" == /* && "$HF_HOME" != "$HOME" && "$HF_HOME" != "$HOME/"* ]] || {
    echo "HF_HOME must be an absolute path outside the home directory" >&2
    exit 2
}
py="${EMOPAIR_PYTHON:?set EMOPAIR_PYTHON to a Python executable with vLLM installed}"
[[ -x "$py" ]] || { echo "Python environment is missing: $py" >&2; exit 2; }
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# FlashInfer's sampling JIT needs nvcc, which the compute image does not ship.
# vLLM's native sampler supports the same greedy decoding protocol.
export VLLM_USE_FLASHINFER_SAMPLER=0
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$(dirname "$HF_HOME")/xdg-cache/emopair}"
mkdir -p "$XDG_CACHE_HOME"
cd "$repo_root"

case "${EMOPAIR_SLURM_MODEL:-}" in
    small_array)
        case "${SLURM_ARRAY_TASK_ID:?}" in
            0) model=bagel ;;
            1) model=emoloom ;;
            *) echo "invalid array task ID" >&2; exit 2 ;;
        esac
        ;;
    openemotions)
        model=openemotions
        ;;
    *) echo "EMOPAIR_SLURM_MODEL must be small_array or openemotions" >&2; exit 2 ;;
esac

mapfile -t revisions < <("$py" - "$repo_root/baselines/emotion-finetuned-llms/revisions.json" "$model" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]
print(data["model"])
if "base" in data:
    print(data["base"])
PY
)
revision_args=(--model-revision "${revisions[0]}")
if [[ "${#revisions[@]}" -gt 1 ]]; then
    revision_args+=(--base-revision "${revisions[1]}")
fi

"$py" - "$model" <<'PY'
import sys
import torch
model = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the allocated Slurm job")
gpu = torch.cuda.get_device_properties(0)
gib = gpu.total_memory / 1024**3
print(f"allocated GPU: {gpu.name}, {gib:.1f} GiB", flush=True)
if model == "openemotions" and gib <= 60:
    raise SystemExit("OpenEmotions needs a GPU with more than 60 GiB VRAM")
if model != "openemotions" and "A6000" not in gpu.name.upper():
    raise SystemExit("Bagel and Emoloom require the allocated RTX A6000")
PY

child=""
forward_stop() {
    if [[ -n "$child" ]]; then
        kill -USR1 "$child" 2>/dev/null || true
    fi
}
cancel_without_requeue() {
    trap - USR1 TERM
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
    fi
    exit 143
}
trap forward_stop USR1
trap cancel_without_requeue TERM

"$py" "$runner" run --model "$model" --split test --full --execute \
    --output-root "$output_root" "${revision_args[@]}" &
child=$!
status=0
while true; do
    status=0
    wait "$child" || status=$?
    if kill -0 "$child" 2>/dev/null; then
        continue
    fi
    break
done
trap - USR1 TERM

if [[ "$status" -eq 75 ]]; then
    echo "safe checkpoint reached; requeueing ${SLURM_JOB_ID:?}" >&2
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
if [[ "$status" -ne 0 ]]; then
    exit "$status"
fi
"$py" "$runner" export --model "$model" --split test \
    --output-root "$output_root" "${revision_args[@]}"
