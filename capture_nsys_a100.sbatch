#!/bin/bash
#SBATCH --job-name=nsys_llm_explain_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/nsys_llm_explain_%j.out
#SBATCH --error=logs/nsys_llm_explain_%j.err

set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="artifacts/${RUN_ID}"
TRACE_PREFIX="${OUT_DIR}/trace"

mkdir -p "${OUT_DIR}" logs

echo "RUN_ID=${RUN_ID}"
echo "OUT_DIR=${OUT_DIR}"

###############################################################################
# Environment (placeholders - edit to match your cluster)
###############################################################################
# module purge
# module load cuda/12.2
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate vllm

###############################################################################
# Workload (small vLLM generation loop; keep iterations low)
###############################################################################
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
DTYPE="${DTYPE:-float16}"

WORKLOAD_CMD=(python3.11 - <<'PY'
import os
from vllm import LLM, SamplingParams

model = os.environ.get("MODEL", "meta-llama/Llama-2-7b-hf")
dtype = os.environ.get("DTYPE", "float16")

llm = LLM(model=model, dtype=dtype, tensor_parallel_size=1)
params = SamplingParams(max_tokens=32, temperature=0.0)

prompts = ["Hello! Summarize: Nsight Systems SQLite export."] * 8

# Warmup
llm.generate(prompts, params)

# Short measured loop
for _ in range(3):
    llm.generate(prompts, params)
print("done")
PY
)

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
# Analyze SQLite offline
###############################################################################
echo "Running explainer..."
python3.11 -m pip install -e "nsys_llm_explainer" >/dev/null
nsys-llm-explain "${OUT_DIR}/trace.sqlite" --out "${OUT_DIR}/"

echo "Done. Report: ${OUT_DIR}/report.md"

