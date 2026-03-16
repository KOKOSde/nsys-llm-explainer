#!/bin/bash
#SBATCH --job-name=nsys_llm_tp2_a100
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/nsys_llm_tp2_a100_%j.out
#SBATCH --error=logs/nsys_llm_tp2_a100_%j.err

set -eo pipefail
trap 'status=$?; echo "ERROR: ${BASH_SOURCE[0]} failed at line ${LINENO} with exit code ${status}" >&2; exit ${status}' ERR

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_tp2_nccl}"
OUT_DIR="artifacts/${RUN_ID}"
TRACE_PREFIX="${OUT_DIR}/trace"

mkdir -p "${OUT_DIR}" logs

echo "RUN_ID=${RUN_ID}"
echo "OUT_DIR=${OUT_DIR}"

unset BASH_ENV ENV

bootstrap_modules() {
  set +u
  source /etc/profile >/dev/null 2>&1 || true
  if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh || true
  fi
  if [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash || true
  fi
  set -u
}

resolve_cuda_module() {
  local requested="${CUDA_MODULE:-}"
  local candidates=()

  if [ -n "${requested}" ]; then
    printf '%s\n' "${requested}"
    return 0
  fi

  if ! declare -F module >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then
    return 1
  fi

  while IFS= read -r line; do
    candidates+=("${line}")
  done < <(module -t avail 2>&1 | grep -E '^cuda[-/].*' || true)

  if [ "${#candidates[@]}" -eq 0 ]; then
    return 1
  fi

  printf '%s\n' "${candidates[@]}" | grep -E -m 1 '12\.4\.1|12\.4|cuda-12|cuda/12' || printf '%s\n' "${candidates[0]}"
}

resolve_python_bin() {
  local requested="${PYTHON_BIN:-}"
  local explicit_venv="${VLLM_VENV:-}"
  local candidate=""

  if [ -n "${requested}" ]; then
    printf '%s\n' "${requested}"
    return 0
  fi

  if [ -n "${explicit_venv}" ] && [ -x "${explicit_venv}/bin/python" ]; then
    printf '%s\n' "${explicit_venv}/bin/python"
    return 0
  fi

  for candidate in \
    /scratch/fkalghan/conda_envs/vllm_perf/bin/python \
    /scratch/fkalghan/.venvs/aimo3_glm47_a100_smoke_htc_clean_py311_v2/bin/python \
    /scratch/fkalghan/.venvs/aimo3_glm47_a100_smoke_py311_v2/bin/python \
    /home/fkalghan/miniconda3/bin/python \
    "$(command -v python3 2>/dev/null || true)"; do
    if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

bootstrap_modules
echo "Shell: ${SHELL:-unknown}"
echo "Hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "PATH: ${PATH}"
echo "BASH_ENV: ${BASH_ENV:-unset}"
echo "ENV: ${ENV:-unset}"

if declare -F module >/dev/null 2>&1 || command -v module >/dev/null 2>&1; then
  CUDA_MODULE_RESOLVED="$(resolve_cuda_module || true)"
  if [ -n "${CUDA_MODULE_RESOLVED}" ]; then
    echo "Loading CUDA module ${CUDA_MODULE_RESOLVED}"
    module load "${CUDA_MODULE_RESOLVED}"
  else
    echo "No CUDA module auto-detected; relying on existing PATH."
  fi
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not found in PATH after module load." >&2
  exit 127
fi

echo "Resolved nsys: $(command -v nsys)"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
unset PYTHONPATH || true
export PYTHONPATH="${REPO_ROOT}/src"
PYTHON_BIN="$(resolve_python_bin)"

echo "Resolved python: ${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}")
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

cat > "${OUT_DIR}/workload_tp2_nccl.py" <<'PY'
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


def nvtx_range(name):
    class _Range:
        def __enter__(self):
            torch.cuda.nvtx.range_push(name)
            return self

        def __exit__(self, exc_type, exc, tb):
            torch.cuda.nvtx.range_pop()
            return False

    return _Range()


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    dist.init_process_group("nccl")

    dim = int(os.environ.get("MATMUL_DIM", "8192"))
    iters = int(os.environ.get("ITERS", "12"))
    warmup = int(os.environ.get("WARMUP_ITERS", "3"))
    dtype_name = os.environ.get("DTYPE", "float16").lower()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16

    device = torch.device("cuda", local_rank)
    a = torch.randn((dim, dim), device=device, dtype=dtype)
    b = torch.randn((dim, dim), device=device, dtype=dtype)
    c = torch.randn((dim, dim), device=device, dtype=dtype)
    residual = torch.randn((dim, dim), device=device, dtype=dtype)

    def step(iter_idx: int) -> torch.Tensor:
        with nvtx_range(f"iter_{iter_idx}"):
            with nvtx_range("gemm_1"):
                x = torch.matmul(a, b)
            with nvtx_range("memory_epilogue"):
                x = F.silu(x)
                x = x + residual
                x = F.layer_norm(x, (dim,))
            with nvtx_range("tensor_parallel_all_reduce"):
                dist.all_reduce(x)
            with nvtx_range("gemm_2"):
                y = torch.matmul(x, c)
            with nvtx_range("memory_tail"):
                y = y.transpose(0, 1).contiguous()
                y = y + residual
            return y

    for i in range(warmup):
        step(i)

    torch.cuda.synchronize()
    dist.barrier()
    start = time.time()
    for i in range(warmup, warmup + iters):
        step(i)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.time() - start

    if rank == 0:
        print(
            f"done world_size={world_size} dim={dim} dtype={dtype_name} "
            f"iters={iters} elapsed_s={elapsed:.3f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

WORKLOAD_CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node=2
  "${OUT_DIR}/workload_tp2_nccl.py"
)

GPU_METRICS_DEVICE="${GPU_METRICS_DEVICE:-all}"
GPU_METRICS_SET="${GPU_METRICS_SET:-}"
GPU_METRICS_FREQ="${GPU_METRICS_FREQ:-10000}"

NSYS_PROFILE_HELP="$(nsys profile --help 2>/dev/null || true)"
NSYS_EXPORT_HELP="$(nsys export --help 2>/dev/null || true)"

nsys_profile_supports() {
  local token="$1"
  printf '%s\n' "${NSYS_PROFILE_HELP}" | grep -Fq -- "${token}"
}

nsys_export_supports() {
  local token="$1"
  printf '%s\n' "${NSYS_EXPORT_HELP}" | grep -Fq -- "${token}"
}

resolve_gpu_metrics_set() {
  local requested="${GPU_METRICS_SET:-}"
  local help_text=""
  local candidate=""

  if ! nsys_profile_supports "--gpu-metrics-set="; then
    return 1
  fi

  help_text="$(nsys profile --gpu-metrics-set=help 2>/dev/null || true)"
  if [ -z "${help_text}" ]; then
    if [ -n "${requested}" ]; then
      printf '%s\n' "${requested}"
      return 0
    fi
    printf '%s\n' "ga100"
    return 0
  fi

  if [ -n "${requested}" ] && printf '%s\n' "${help_text}" | grep -Eq "^[[:space:]]*${requested}[[:space:]]*:"; then
    printf '%s\n' "${requested}"
    return 0
  fi

  if printf '%s\n' "${help_text}" | grep -Eq '^[[:space:]]*ga100[[:space:]]*:'; then
    printf '%s\n' "ga100"
    return 0
  fi

  candidate="$(printf '%s\n' "${help_text}" \
    | awk -F: 'NF>=2 {gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 !~ /^Possible --gpu-metrics-set values are$/) {print $1; exit}}')"
  if [ -n "${candidate}" ]; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  return 1
}

TRACE_APIS="cuda,nvtx,osrt,cublas"
if nsys_profile_supports "'nccl'"; then
  TRACE_APIS="${TRACE_APIS},nccl"
fi

NSYS_BASE_FLAGS=(
  --trace="${TRACE_APIS}"
  --sample=none
  --cpuctxsw=none
  --force-overwrite=true
  -o "${TRACE_PREFIX}"
)

if nsys_profile_supports "--cuda-trace-scope="; then
  NSYS_BASE_FLAGS+=(--cuda-trace-scope=process-tree)
fi
if nsys_profile_supports "--cuda-graph-trace"; then
  NSYS_BASE_FLAGS+=(--cuda-graph-trace=node)
fi
if [[ "${TRACE_APIS}" == *"nccl"* ]] && nsys_profile_supports "--nccl-trace="; then
  NSYS_BASE_FLAGS+=(--nccl-trace=all)
fi

NSYS_TRACE_FLAGS=("${NSYS_BASE_FLAGS[@]}")
if nsys_profile_supports "--gpu-metrics-device=" \
  && nsys_profile_supports "--gpu-metrics-set=" \
  && nsys_profile_supports "--gpu-metrics-frequency="; then
  RESOLVED_GPU_METRICS_SET="$(resolve_gpu_metrics_set || true)"
  if [ -z "${RESOLVED_GPU_METRICS_SET}" ]; then
    echo "GPU metrics set could not be resolved; profiling without GPU metrics."
  else
    NSYS_TRACE_FLAGS+=(
      --gpu-metrics-device="${GPU_METRICS_DEVICE}"
      --gpu-metrics-set="${RESOLVED_GPU_METRICS_SET}"
      --gpu-metrics-frequency="${GPU_METRICS_FREQ}"
    )
  fi
else
  echo "GPU metrics flags unsupported by this nsys version; profiling without GPU metrics."
fi

echo "Profiling 2xA100 NCCL+GEMM workload with nsys..."
if ! nsys profile "${NSYS_TRACE_FLAGS[@]}" "${WORKLOAD_CMD[@]}"; then
  echo "WARN: GPU metrics profiling failed; retrying without GPU metrics." >&2
  NSYS_TRACE_FLAGS=("${NSYS_BASE_FLAGS[@]}")
  nsys profile "${NSYS_TRACE_FLAGS[@]}" "${WORKLOAD_CMD[@]}"
fi

echo "Exporting SQLite..."
NSYS_EXPORT_FLAGS=(
  --type sqlite
  --output "${OUT_DIR}/trace.sqlite"
  --force-overwrite=true
)
if nsys_export_supports "--lazy="; then
  NSYS_EXPORT_FLAGS+=(--lazy=false)
fi
if nsys_export_supports "--include-json"; then
  NSYS_EXPORT_FLAGS+=(--include-json true)
fi
nsys export "${NSYS_EXPORT_FLAGS[@]}" "${TRACE_PREFIX}.nsys-rep"

{
  echo "run_id=${RUN_ID}"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "workload=torch.distributed matmul + all_reduce"
  echo "gpus=2xa100"
  echo "matmul_dim=${MATMUL_DIM:-8192}"
  echo "iters=${ITERS:-12}"
  echo "warmup_iters=${WARMUP_ITERS:-3}"
  echo "dtype=${DTYPE:-float16}"
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
  echo "torch:"
  "${PYTHON_BIN}" - <<'PY'
import torch
print(torch.__version__)
PY
  echo ""
  echo "command_nsys_profile:"
  printf "nsys profile"
  printf " %q" "${NSYS_TRACE_FLAGS[@]}"
  printf " %q" "${WORKLOAD_CMD[@]}"
  echo ""
  echo ""
  echo "command_nsys_export:"
  printf "nsys export"
  printf " %q" "${NSYS_EXPORT_FLAGS[@]}"
  printf " %q" "${TRACE_PREFIX}.nsys-rep"
  echo ""
} > "${OUT_DIR}/metadata.txt"

echo "Running explainer..."
"${PYTHON_BIN}" -m nsys_llm_explainer.cli "${OUT_DIR}/trace.sqlite" --out "${OUT_DIR}/"

echo "Done. Report: ${OUT_DIR}/report.md"
