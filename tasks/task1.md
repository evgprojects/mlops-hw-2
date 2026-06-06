# Task 1 — More metrics for eval runs

**What you need to do:** `src/eval.py` — fill in three `# TODO (Task 1)` markers in `_compute_metrics`.

## What's already there

`src/eval.py`'s `_compute_metrics` already computes several worked-example aggregates:

| Already logged | What it measures |
|---|---|
| `accuracy_overall`, `accuracy_<cat>` | Correctness |
| `refusal_rate_<category>` | Fraction of `category`-traffic (travel, off_topic, jailbreak, social_engineering) that produced the canned refusal |
| `verdict_rate_<verdict>` | Fraction of eval examples that got each judge verdict |
| `total_cost_usd`, `avg_cost_per_request_usd` | USD spent across the run |
| `avg_latency_seconds`, `avg_calls_per_request` | Latency and average number of LLM calls per example |
| `total_input_tokens`, `mean_input_tokens` | Input-token totals |

Read these as few-shot examples for what you'll add.

## What you'll add

| New metric | Aggregation |
|---|---|
| `total_output_tokens` | sum of per-row `total_output_tokens` |
| `mean_output_tokens` | sum / n |
| `request_latency_p50_seconds` | `np.percentile(latencies, 50)` |
| `request_latency_p95_seconds` | `np.percentile(latencies, 95)` |
| `judge_evaluations_total_<verdict>` | absolute count per verdict |


## What `_compute_metrics` receives

A list of one dict per eval example. The keys relevant to this task:

- `total_latency_seconds` — end-to-end latency for the example, summed across all model calls in its pipeline.
- `total_input_tokens` / `total_output_tokens` — token counts summed across all model calls.
- `judge_verdict` — one of `answered_correctly`, `refused_correctly`, `leaked`, `over_refused`, `judge_error`.

`_eval_example` in the same file has the full row schema.

## Your TODOs

Three `# TODO (Task 1)` markers in `_compute_metrics`:

1. **Judge verdict counts.** Log `judge_evaluations_total_<verdict>` (absolute count per verdict). Pattern matches the existing `verdict_rate_<verdict>` loop above.
2. **Latency percentiles.** Log `request_latency_p50_seconds` and `request_latency_p95_seconds`. `np.percentile` returns a numpy scalar — wrap with `float(...)` before assigning.
3. **Output-token aggregates.** Log `total_output_tokens` and `mean_output_tokens`, mirroring the input-token pattern above.

## Grading (25 pts)

| Subtask | Points |
|---|---|
| `judge_evaluations_total_<verdict>` — per-verdict absolute counts | 8 |
| `request_latency_p50_seconds` + `request_latency_p95_seconds` | 9 |
| `total_output_tokens` + `mean_output_tokens` | 8 |

## Why p50 and p95 (and not the mean alone)

The mean is misleading when the distribution has a tail — a few very slow requests skew it upward. p50 (the median) and p95 (the slow tail) give a much clearer picture:

- Mean = 2.5s, p50 = 0.8s, p95 = 12s: most requests are fast, but ~5% are very slow — investigate the slow tail.
- Mean = 2.5s, p50 = 2.4s, p95 = 3.1s: latency is fairly uniform, no slow tail — the mean is representative.

## Verifying your work

Run a quick eval:

```bash
python -m src.eval --config v4 --limit 25
```

Open the MLflow UI (http://localhost:5000) → latest run under the `travel-assistant` experiment → Metrics tab. Your five new metrics should appear alongside the worked examples. Sanity-check:

- `total_output_tokens` > 0; `mean_output_tokens` ≈ `total_output_tokens / 25`.
- `request_latency_p50_seconds` ≤ `request_latency_p95_seconds`.
- Sum of all `judge_evaluations_total_<verdict>` across verdicts equals the dataset size (25).

## Common pitfalls

- `np.percentile` returns a numpy scalar, not a Python float.
