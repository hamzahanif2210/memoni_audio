#!/bin/bash
# submit_memoni_jobs.sh
#
# Fan-out launcher: submits one SLURM job per chunk of videos.
#
# Usage:
#   bash submit_memoni_jobs.sh [OPTIONS]
#
# Options (all optional — defaults shown):
#   --total-videos  N    Total number of videos in the CSV after filtering (default: 1000)
#   --chunk-size    N    Videos assigned to each job (default: 200)
#   --account       STR  SBATCH --account value            (default: def-mdanning)
#   --hf-username   STR  Hugging Face username             (default: Aqiba)
#   --dataset-name  STR  Hugging Face dataset name         (default: memoni_clean_audio)
#   --base-dir      STR  Scratch dir for temp audio files  (default: /scratch/$USER/memoni)
#   --script        STR  Path to memoni_audio_creation_v3.py
#                        (default: same directory as this submit script)
#   --test               Submit only the first chunk (job-index 0) for a quick sanity check
#
# Example — submit 5 jobs, 200 videos each (covers 1000 videos total):
#   bash submit_memoni_jobs.sh --total-videos 1000 --chunk-size 200 --account def-mdanning
#
# Example — tighter jobs (100 videos each):
#   bash submit_memoni_jobs.sh --total-videos 600 --chunk-size 100
#
# Example — test run (first chunk only):
#   bash submit_memoni_jobs.sh --chunk-size 200 --account def-mdanning --test

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
TOTAL_VIDEOS=1918
CHUNK_SIZE=200
ACCOUNT="def-mdanning"
HF_USERNAME="hamzahanif"
DATASET_NAME="memoni_clean_audio"
BASE_DIR="/scratch/${USER}/memoni"
SCRIPT_DIR="/project/ctb-stelzer/hamza95/memoni_audio"
PYTHON_SCRIPT="${SCRIPT_DIR}/memoni_audio_creation_v3.py"
LOG_DIR="${SCRIPT_DIR}/logs"
TEST_MODE=true

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --total-videos)  TOTAL_VIDEOS="$2";  shift 2 ;;
        --chunk-size)    CHUNK_SIZE="$2";    shift 2 ;;
        --account)       ACCOUNT="$2";       shift 2 ;;
        --hf-username)   HF_USERNAME="$2";   shift 2 ;;
        --dataset-name)  DATASET_NAME="$2";  shift 2 ;;
        --base-dir)      BASE_DIR="$2";      shift 2 ;;
        --script)        PYTHON_SCRIPT="$2"; shift 2 ;;
        --hf-token)      HF_TOKEN="$2";      shift 2 ;;
        --test)          TEST_MODE=true;     shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Derived values ────────────────────────────────────────────────────────────
NUM_JOBS=$(( (TOTAL_VIDEOS + CHUNK_SIZE - 1) / CHUNK_SIZE ))   # ceiling division

if [[ "${TEST_MODE}" == true ]]; then
    NUM_JOBS=1
fi

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo " Memoni Audio Pipeline — SLURM Fan-Out Launcher"
echo "============================================================"
echo "  total videos  : ${TOTAL_VIDEOS}"
echo "  chunk size    : ${CHUNK_SIZE}"
echo "  jobs to submit: ${NUM_JOBS}"
echo "  account       : ${ACCOUNT}"
echo "  hf-username   : ${HF_USERNAME}"
echo "  dataset-name  : ${DATASET_NAME}"
echo "  base-dir      : ${BASE_DIR}"
echo "  python script : ${PYTHON_SCRIPT}"
echo "  log dir       : ${LOG_DIR}"
echo "  test mode     : ${TEST_MODE}"
echo "  hf-token      : ${HF_TOKEN:+(set)}"
echo "============================================================"

# ── Resolve HF token (arg > env var > ~/.hf_token file) ──────────────────────
if [[ -z "${HF_TOKEN}" ]]; then
    if [[ -n "${HF_TOKEN:-}" ]]; then
        : # already set via environment
    elif [[ -f "${HOME}/.hf_token" ]]; then
        HF_TOKEN=$(cat "${HOME}/.hf_token")
        echo "  Loaded HF_TOKEN from ~/.hf_token"
    else
        echo "  WARNING: HF_TOKEN not set — upload to Hugging Face will fail."
        echo "           Pass --hf-token TOKEN, set HF_TOKEN env var, or create ~/.hf_token"
    fi
fi

# ── Submit one job per slice ──────────────────────────────────────────────────
for (( JOB_IDX=0; JOB_IDX<NUM_JOBS; JOB_IDX++ )); do

    JOB_NAME="memoni_audio_${JOB_IDX}"

    sbatch \
        --job-name="${JOB_NAME}" \
        --account="${ACCOUNT}" \
        --time=08:00:00 \
        --mem=32G \
        --cpus-per-task=4 \
        --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
        --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
        <<EOF
#!/bin/bash
set -euo pipefail

module load python/3.11

# Activate your virtual-env if needed:
# source /path/to/venv/bin/activate

echo "Job index : ${JOB_IDX}"
echo "Chunk size: ${CHUNK_SIZE}"
echo "Account   : ${ACCOUNT}"

python "${PYTHON_SCRIPT}" \
    --job-index   ${JOB_IDX} \
    --chunk-size  ${CHUNK_SIZE} \
    --account     "${ACCOUNT}" \
    --hf-username "${HF_USERNAME}" \
    --dataset-name "${DATASET_NAME}" \
    --base-dir    "${BASE_DIR}/job_${JOB_IDX}" \
    --hf-token    "${HF_TOKEN}"
EOF

    echo "  Submitted job index ${JOB_IDX}  (${JOB_NAME})"
done

echo ""
if [[ "${TEST_MODE}" == true ]]; then
    echo "TEST MODE: submitted 1 job (job-index 0, first ${CHUNK_SIZE} videos)."
else
    echo "All ${NUM_JOBS} jobs submitted."
fi
echo "Monitor with:  squeue -u \${USER}"