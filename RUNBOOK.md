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

