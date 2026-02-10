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

# Optional: activate a venv directly (set VLLM_VENV=/path/to/venv)
if [[ -n "${VLLM_VENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VLLM_VENV}/bin/activate"
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

llm = LLM(model=model, dtype=dtype, tensor_parallel_size=1, enforce_eager=enforce_eager)
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

WORKLOAD_CMD=(python "${OUT_DIR}/workload_vllm.py")

###############################################################################
# Nsight Systems capture (CUDA + NVTX if present)
###############################################################################
NSYS_TRACE_FLAGS=(
  --trace=cuda,nvtx,osrt
  --sample=none
  --cpuctxsw=none
  --force-overwrite=true
  -o "${TRACE_PREFIX}"
)

echo "Profiling workload with nsys..."
nsys profile "${NSYS_TRACE_FLAGS[@]}" "${WORKLOAD_CMD[@]}"

echo "Exporting SQLite..."
nsys export --type sqlite --output "${OUT_DIR}/trace.sqlite" "${TRACE_PREFIX}.nsys-rep"

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
  python -V || true
  echo ""
  echo "vllm_version:"
  python -c "import importlib.metadata as md; print(md.version('vllm'))" 2>/dev/null || true
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
python -m pip install -e "nsys_llm_explainer" >/dev/null
nsys-llm-explain "${OUT_DIR}/trace.sqlite" --out "${OUT_DIR}/"

echo "Done. Report: ${OUT_DIR}/report.md"

