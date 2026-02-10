## Runbook: capture → export → explain (A100)

### Prereqs

- You have Nsight Systems CLI (`nsys`) available on the node.
- You can run vLLM in your environment (module/conda/etc).
- You are profiling on **A100**.

### 1) Configure your environment (once)

Open `capture_nsys_a100.sbatch` and update the `module load` / `conda activate` section to match your cluster.
Alternatively, you can set `VLLM_VENV=/path/to/venv` to activate an existing venv directly.

### 2) Submit the job

```bash
# A100
sbatch --gres=gpu:a100:1 nsys_llm_explainer/capture_nsys_a100.sbatch
```

Optional overrides:

```bash
RUN_ID=my_run MODEL=Qwen/Qwen2.5-0.5B-Instruct DTYPE=float16 sbatch --gres=gpu:a100:1 nsys_llm_explainer/capture_nsys_a100.sbatch
```

If `nsys` is not on PATH by default, you can auto-load a CUDA module that provides it:

```bash
CUDA_MODULE=cuda-12.4.1-gcc-12.1.0 sbatch --gres=gpu:a100:1 nsys_llm_explainer/capture_nsys_a100.sbatch
```

### 3) Artifacts

The job writes:

- `artifacts/<run_id>/trace.nsys-rep`
- `artifacts/<run_id>/trace.sqlite`
- `artifacts/<run_id>/report.md`
- `artifacts/<run_id>/report.json`
- `artifacts/<run_id>/tables/*.csv`
- `artifacts/<run_id>/metadata.txt` (nsys version, GPU model, vLLM version, commands used)

### Notes

- If your workload is too long, lower the iteration count in the script.
- For better phase attribution, add NVTX ranges in your inference code around prefill/decode/sampling.
- The “GPU idle” metric in the report is computed over the **observed kernel time window**. If your capture includes model load, a long warmup, or a long idle tail, idle% can look extreme. Prefer capturing a short steady-state window.
- If your Nsight Systems version supports capture-range options, consider restricting capture to an NVTX range (see `nsys profile --help` for your version).

### PR writeup template (copy/paste)

Use this as a starting point for a reviewer-friendly PR description:

```text
## Summary
<1–3 sentences: what changed and what the tool does>

## Why
<motivation / user pain: why this is needed>

## Minimal repro
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=node -o trace python your_workload.py
nsys export --type sqlite --output trace.sqlite --force-overwrite=true --lazy=false trace.nsys-rep
nsys-llm-explain trace.sqlite --out artifacts/run_YYYYMMDD_HHMMSS/

## Docs changes
- <what changed in README/RUNBOOK/examples>

## Risk / limitations
- <schema differences, timestamp assumptions, NVTX coverage, PID decoding>

## Test plan
python -m unittest discover -s tests -p "test*.py" -q
```

