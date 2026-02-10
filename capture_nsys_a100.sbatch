#!/bin/bash
#SBATCH --job-name=nsys_llm_explain_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/nsys_llm_explain_%j.out
#SBATCH --error=logs/nsys_llm_explain_%j.err

set -euo pipefail

# Slurm copies the script to a spool directory; use the submit directory as repo root.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="artifacts/${RUN_ID}"
TRACE_PREFIX="${OUT_DIR}/trace"

mkdir -p "${OUT_DIR}" logs

echo "RUN_ID=${RUN_ID}"
echo "OUT_DIR=${OUT_DIR}"

###############################################################################
# Environment setup (edit to match your cluster)
###############################################################################
# module purge
# module load cuda/12.2
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate vllm

# Optional: auto-load a CUDA module to provide `nsys` (set CUDA_MODULE=...).
if ! command -v nsys >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1 && [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "${CUDA_MODULE}"
  fi
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: \`nsys\` not found in PATH. Load Nsight Systems (often via a CUDA module) and retry." >&2
  exit 127
fi

# Python environment: prefer an explicit venv python if provided.
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${VLLM_VENV:-}" ]]; then
  PYTHON_BIN="${VLLM_VENV}/bin/python"
fi

###############################################################################
# Workload (small vLLM generation loop; keep iterations low)
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DTYPE="${DTYPE:-float16}"

cat > "${OUT_DIR}/workload_vllm.py" <<'PY'
import os

from vllm import LLM, SamplingParams

model = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
dtype = os.environ.get("DTYPE", "float16")
enforce_eager = os.environ.get("ENFORCE_EAGER", "1") != "0"
attn_backend = os.environ.get("ATTENTION_BACKEND", "").strip()

attention_config = {"backend": attn_backend} if attn_backend else None

llm = LLM(
    model=model,
    dtype=dtype,
    tensor_parallel_size=1,
    enforce_eager=enforce_eager,
    attention_config=attention_config,
)
params = SamplingParams(max_tokens=32, temperature=0.0)

batch = int(os.environ.get("BATCH", "8"))
warmup_iters = int(os.environ.get("WARMUP_ITERS", "1"))
iters = int(os.environ.get("ITERS", "3"))

prompts = ["Hello! Summarize: Nsight Systems SQLite export."] * batch

# Warmup
for _ in range(warmup_iters):
    llm.generate(prompts, params)

# Short measured loop
for _ in range(iters):
    llm.generate(prompts, params)
print("done")
PY

WORKLOAD_CMD=("${PYTHON_BIN}" "${OUT_DIR}/workload_vllm.py")

###############################################################################
# Nsight Systems capture (CUDA + NVTX if present)
###############################################################################
NSYS_TRACE_FLAGS=(
  --trace=cuda,nvtx,osrt
  --cuda-graph-trace=node
  --sample=none
  --cpuctxsw=none
  --force-overwrite=true
  -o "${TRACE_PREFIX}"
)

echo "Profiling workload with nsys..."
nsys profile "${NSYS_TRACE_FLAGS[@]}" "${WORKLOAD_CMD[@]}"

echo "Exporting SQLite..."
nsys export --type sqlite --output "${OUT_DIR}/trace.sqlite" --force-overwrite=true --lazy=false "${TRACE_PREFIX}.nsys-rep"

###############################################################################
# Record run metadata (for README/examples)
###############################################################################
{
  echo "run_id=${RUN_ID}"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model=${MODEL}"
  echo "dtype=${DTYPE}"
  echo ""
  echo "nsys_version:"
  nsys --version || true
  echo ""
  echo "nvidia_smi:"
  nvidia-smi -L || true
  echo ""
  echo "python:"
  "${PYTHON_BIN}" -V || true
  echo ""
  echo "vllm_version:"
  "${PYTHON_BIN}" -c "import importlib.metadata as md; print(md.version('vllm'))" 2>/dev/null || true
  echo ""
  echo "command_nsys_profile:"
  printf "nsys profile"
  printf " %q" "${NSYS_TRACE_FLAGS[@]}"
  printf " %q" "${WORKLOAD_CMD[@]}"
  echo ""
  echo ""
  echo "command_nsys_export:"
  echo "nsys export --type sqlite --output ${OUT_DIR}/trace.sqlite ${TRACE_PREFIX}.nsys-rep"
} > "${OUT_DIR}/metadata.txt"

###############################################################################
# Analyze SQLite offline
###############################################################################
echo "Running explainer..."
"${PYTHON_BIN}" -m pip install -e "nsys_llm_explainer" >/dev/null
"${PYTHON_BIN}" -m nsys_llm_explainer.cli "${OUT_DIR}/trace.sqlite" --out "${OUT_DIR}/"

echo "Done. Report: ${OUT_DIR}/report.md"

