# Zero-shot single-text VAD baselines

This directory prepares five inference-only baselines for the EmoPair paper. Each
model receives **the same single user message** from
[`prompts/single_text_vad.txt`](prompts/single_text_vad.txt), containing one text
and definitions of Valence, Arousal, and Dominance. There are no labeled
examples, training steps, CGCoT breakdowns, or pairwise inputs. The runner
reads only `item_id`, `split`, and `text` from `Results/emopair.csv`; original
EmoBank scores and EmoPair BT targets are never included in requests.

| CLI name | Backend | Requested model ID | Inference mode |
| --- | --- | --- | --- |
| `gpt-6-luna` | OpenAI Responses | `gpt-6-luna` | `reasoning.effort=none`, temperature 0 |
| `gpt-6-sol` | OpenAI Responses | `gpt-6-sol` | `reasoning.effort=none`, temperature 0 |
| `gemini-3.8-flash` | Google Interactions | `gemini-3.8-flash` | `thinking_level=low`, vendor default sampling |
| `gemma-4-12b-it` | offline vLLM | `google/gemma-4-12B-it` | official chat template, `enable_thinking=False`, greedy |
| `qwen3.5-9b` | offline vLLM | `Qwen/Qwen3.5-9B` | official chat template, `enable_thinking=False`, greedy |

The Gemma ID is the official Gemma 4 12B instruction-tuned checkpoint. Gemini
3.8 Flash does not offer a no-thinking setting, so `low` is explicit; its
sampling parameters follow Google's guidance for Gemini 3.8. This difference
must be reported with the results. The API model IDs are aliases, so record
the returned model information and inference date. For open weights, the
runner resolves and stores a Hugging Face commit SHA before loading weights;
resume uses that SHA. The design follows the qscope benchmark's model IDs,
chat-template/vLLM path, and OpenAI Responses pattern, while using one
standalone runner here.

## Set up

Use Python 3.12. For OpenAI and Gemini, install
`baselines/out-of-box-llms/requirements-api.txt` and export `OPENAI_API_KEY` or
`GEMINI_API_KEY` (the Google SDK also accepts `GOOGLE_API_KEY`). For Gemma and
Qwen, install `baselines/out-of-box-llms/requirements-open.txt` on a Linux/CUDA machine with
sufficient GPU memory. Gemma may require accepting its Hugging Face license
and setting `HF_TOKEN`. The open-weight requirements match the qscope stack;
vLLM is imported only when an open-weight run starts. No credentials are
written to the output files.

The default input is `Results/emopair.csv`, the default split is `test` (1,000
items), and the default output root is `Results/zero_shot_baselines/`. The
output directory is ignored by Git. Start with a read-only plan:

```bash
python baselines/out-of-box-llms/single_text_vad.py plan --model gpt-6-luna
```

Running inference requires both the `run` command **and** `--execute`. These
examples are for the later execution step; they are not run as part of code
preparation:

```bash
python baselines/out-of-box-llms/single_text_vad.py run --model gpt-6-luna --execute
python baselines/out-of-box-llms/single_text_vad.py run --model gemini-3.8-flash --execute
python baselines/out-of-box-llms/single_text_vad.py run --model gemma-4-12b-it --execute
```

The other names are `gpt-6-sol` and `qwen3.5-9b`. To check formatting on a
few development items later, set `--split dev --max-items 5` and use a
separate `--output-root` from the full run. Once a model's run is complete,
export its predictions without calling a model:

```bash
python baselines/out-of-box-llms/single_text_vad.py export --model gpt-6-luna
```

This produces `predictions.csv` with `item_id`, `split`, `model`,
`parse_status`, `valence`, `arousal`, and `dominance`. Each raw response and
metadata record is retained under `items/` for audit. Scores must be a JSON
object with exactly three numeric values in `[1, 5]`; malformed, out-of-range,
or length-truncated responses are recorded as `invalid` with blank scores.
There is no automatic repair prompt or label-dependent calibration. The
runner saves each item atomically and skips already saved items on resume.
Cloud responses are saved one at a time to limit duplicate requests after an
interruption. An interruption after a provider completes a request but before
its local file is saved can still cause that one request to be repeated.
`export` requires every selected item to have an output. Changing the input,
prompt, model, split, or item limit requires a new output directory.

## Evaluation handoff

Join `predictions.csv` to `Results/emopair.csv` by `item_id` and use the
existing 1,000-item test split. The paper's comparison table uses Valence MAE
and Pearson correlation, Arousal/Dominance Pearson correlation with BT and
original EmoBank scores, and pairwise accuracy derived from item-score order.
Report the invalid-response rate and effective N for each metric. The
full pairwise decision file referenced by `model compairson.ipynb`
(`Results/emopair_pairs_w_breakdowns.csv`) is absent from the current
checkout, so reproducing the A/D pairwise-accuracy columns needs that file.
Also distinguish the GPT-5.1/Claude-Sonnet-generated EmoPair labels from
independent human evaluation when interpreting correlations.

## Model/API references

- [OpenAI GPT-6 model IDs](https://developers.openai.com/api/docs/models)
- [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
- [Gemini Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)
- [Gemma 4 12B IT](https://huggingface.co/google/gemma-4-12B-it)
- [Qwen3.5 9B](https://huggingface.co/Qwen/Qwen3.5-9B)
