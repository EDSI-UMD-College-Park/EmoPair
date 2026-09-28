# Emotion-finetuned LLMs on EmoBank

`emobank_vllm.py` evaluates the model-native single-text prompts of Bagel,
Emoloom, and OpenEmotions against the local `Data/emobank.csv` test split. It
loads each model once, sends one prompt and one generation per text, and saves
every raw generation before exporting scores and metrics. Only text enters the
prompt. Gold V/A/D values are read separately for metrics.

| CLI model | Hugging Face checkpoint | Supported dimensions | Native scale |
| --- | --- | --- | --- |
| `bagel` | `jondurbin/bagel-7b-v0.4` | V/A/D | 1–5 |
| `emoloom` | `Lixeeone/Emoloom-2B` | V/A/D | 0–1 |
| `openemotions` | `fabozzi/OpenEmotions-Mistral-24B-Instruct` | V/A | V −100–100; A 0–100 |

The model-specific prompts are frozen in `prompts/`. Bagel's task text is from
its [EmoBank loader](https://raw.githubusercontent.com/jondurbin/bagel/main/bagel/data_sources/emobank.py).
Emoloom's system and user text come from its [model card Usage section](https://huggingface.co/Lixeeone/Emoloom-2B);
the card's ordinary Python string has doubled JSON braces, which this version
corrects to single braces. The single-text invocation leaves `Context:` empty.
OpenEmotions' template is transcribed from [Appendix B, PDF page 10](https://arxiv.org/pdf/2605.16613v1),
using a fixed list order: Anger, Anxiety, Fear, Sadness, Disgust, Optimism,
Excitement, Surprise. This list order is an experiment choice, not a claim about
the authors' training string. The exact template file hashes are in the run
manifest.

The decoding protocol is `model-native-vllm-bf16-v1`. BF16 here is the weight
and compute dtype, **not** low-bit quantization. There is no automatic precision
fallback. Decoding uses greedy sampling (`temperature=0`, one sequence), with
128, 128, and 384 output tokens respectively. vLLM uses a KV cache, so the
Emoloom author's `use_cache=False` is not reproduced. This is a distinct
protocol from a Transformers-based inference recipe.
No JSON grammar or response repair is applied.

## Setup and small run

Use Python 3.12 on a CUDA host with enough GPU RAM. Install
`requirements-open.txt`. OpenEmotions is gated: accept its Hugging Face terms
and authenticate through the normal Hugging Face environment before `audit`
or `run`. No token is stored in the outputs.

```bash
python baselines/emotion-finetuned-llms/emobank_vllm.py plan --model bagel --max-items 5
python baselines/emotion-finetuned-llms/emobank_vllm.py audit --model bagel --max-items 5
python baselines/emotion-finetuned-llms/emobank_vllm.py run --model bagel --max-items 5 --execute
python baselines/emotion-finetuned-llms/emobank_vllm.py export --model bagel --max-items 5
```

Repeat with `--model emoloom` and `--model openemotions`, using separate output
directories. `audit` resolves model and base commit SHAs and inspects the
OpenEmotions adapter config before downloading large weights. For reproducible
reruns, pass `--model-revision` and, where applicable, `--base-revision` with
40-character commit SHAs. `run` stores the resolved SHAs, template/code/config
hashes, weight hashes, software versions, hardware name, and loading checks in
`manifest.json`. Keep `--max-items` and the same options for `export`; use a
fresh `--output-root` for a different pilot. A full run requires both `--full`
and `--execute`.

Emoloom currently publishes weights without config or tokenizer files. The
runner reconstructs those metadata files from the pinned
`Qwen/Qwen1.5-1.8B-Chat` base, compares every safetensors key and shape with a
meta-device model, and checks that tokenizer IDs fit the embeddings. It fails
on any mismatch. This validates loading compatibility, but cannot establish
the author's exact training metadata. OpenEmotions loads the declared Mistral
base through vLLM's LoRA interface only after checking the adapter config and
LoRA tensor pairs. A missing adapter tokenizer triggers a recorded fallback to
the pinned base tokenizer. Access denial or a different declared base fails.

Each item JSON contains the actual chat messages, raw completion, finish
reason, strict parse flags, native scores, normalized `[0,1]` scores, and any
error. `predictions.csv` has blank scores for failures and unsupported D.
`summary.json` reports all-item coverage/status counts and MAE/Pearson on valid
outputs only, with effective N per dimension. Scores are mapped by fixed
endpoints, without fitting or clipping: Bagel `(x−1)/4`, Emoloom `x`,
OpenEmotions V `(x+100)/200` and A `x/100`. Gold EmoBank scores use `(x−1)/4`.
This unit conversion does not calibrate psychological scales.

## Overlap audit and local checks

Bagel's loader selects the official EmoBank `train` split. To check whether
selected inputs have been seen, run:

```bash
python baselines/emotion-finetuned-llms/emobank_vllm.py overlap --model bagel --max-items 1000
```

On this checkout's first 1,000 EmoBank test rows, the local overlap audit found
0 train ID matches, 18 raw text matches, and 19 matches after the Bagel loader's
text cleanup. These counts do not establish whether Bagel saw the test IDs
through any other source. For `Results/emopair.csv`, pass `--input` and keep
`Data/emobank.csv` as `--source-emobank`.

Offline checks verified prompt isolation and strict parsing. The evaluation used
Python 3.12 and vLLM 0.26.0. Hub metadata and the gated OpenEmotions adapter
config were checked using credentials supplied at runtime; the exact commit
SHAs are frozen in `revisions.json`. The completed Slurm jobs verified GPU
weight loading and generated the full test split.

## Slurm evaluation

`submit_slurm.py` launches a two-task A6000 array (Bagel and Emoloom) and one
OpenEmotions job on a partition with larger GPUs. Both request one GPU and four
CPUs. The A6000 jobs use 16 GB host RAM; OpenEmotions uses 24 GB. Each has a
12-hour limit and `--requeue`. Set `EMOPAIR_SMALL_PARTITION`,
`EMOPAIR_SMALL_ACCOUNT`, and `EMOPAIR_SMALL_QOS` for the A6000 partition, and
their `EMOPAIR_LARGE_*` counterparts for the large-GPU partition. Set
`EMOPAIR_PYTHON` to a Python executable with the dependencies installed, or
invoke the submitter with that Python. The submitter passes the repository root
and Python executable to the batch script without embedding a user path.
OpenEmotions uses the live `sinfo` inventory to exclude every known GPU node
whose installed GPU types include a device with at most 60 GiB VRAM. The batch
script checks the allocated GPU again before loading weights. Unknown GPU
types stop submission for review, so the allowlist cannot silently widen.
The submitter refuses a second job with the same benchmark name while one is
pending, running, or requeued; inspect or resume the existing job instead.

```bash
python baselines/emotion-finetuned-llms/submit_slurm.py --test-only
python baselines/emotion-finetuned-llms/submit_slurm.py
```

The job sources `$HOME/.env` by default for `HF_TOKEN` and a scratch-backed
`HF_HOME`; set `EMOPAIR_ENV_FILE` to use another environment file. Slurm logs and
outputs go under the ignored `Results/emotion_finetuned_baselines/` directory;
set `EMOPAIR_OUTPUT_ROOT` to use another output directory. At 120 seconds
before a time limit, Slurm sends USR1 to the batch shell; the runner completes
its current item, saves it atomically, and the wrapper requeues the job. A hard
preemption can discard only the in-flight item. On restart, saved items are
verified and skipped, and `export` runs after all items are complete. To inspect
or manually restart an interrupted allocation, use `squeue`, `sacct`, and
`scontrol requeue <job-id>`; do not launch a second writer to the same output.
To permanently cancel a requeue-enabled job on a Slurm setup that requeues
cancelled jobs, first run
`scontrol update JobId=<job-id> Requeue=0`, then `scancel <job-id>` and verify
with `squeue`. A plain `scancel` was observed to leave a restarted attempt.

The batch script disables vLLM's FlashInfer sampler, whose JIT may require
`nvcc`, and uses its native sampler for greedy decoding. Hub
snapshots download one file at a time to keep the 24 GB OpenEmotions host-memory
request within budget. OpenEmotions explicitly uses `TRITON_ATTN` so nodes
where automatic FlashInfer attention would need `nvcc` can run the same protocol.

## Completed test split

All three jobs completed and saved 1,000/1,000 valid outputs.
The table reports MAE and Pearson correlation against EmoBank gold scores after
the fixed `[0,1]` endpoint mappings above. OpenEmotions does not predict D.

| Model | V MAE / Pearson | A MAE / Pearson | D MAE / Pearson |
| --- | ---: | ---: | ---: |
| Bagel | 0.0432 / 0.7932 | 0.0445 / 0.4655 | 0.0388 / 0.3606 |
| Emoloom | 0.0546 / 0.5689 | 0.0617 / 0.3328 | 0.0835 / 0.2073 |
| OpenEmotions | 0.0982 / 0.7812 | 0.3143 / 0.4581 | unsupported |

The exact metrics and per-item statuses are in each run's `summary.json` and
`items/` directory. `predictions.csv` contains all 1,000 rows per model. The
Bagel overlap audit above found repeated train text, so this result should not
be interpreted as a wholly unseen-source evaluation.
