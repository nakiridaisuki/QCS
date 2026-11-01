#!/usr/bin/env bash
#SBATCH --job-name=single-gate
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=single_gate/results/raw/slurm_container-%j.out

# Generic Slurm wrapper: sbatch slurm_container.sh <table-id>
set -euo pipefail

ID=${1:?usage: slurm_container.sh X.Y.Z}
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
SIF="$REPO/single_gate/env/single-gate-cudaq0.16.sif"
ARCH=${ARCH:-sm_90}
cd "$REPO"

if [ ! -f "$SIF" ]; then
    echo "!! SIF not found: $SIF" >&2
    exit 1
fi

exec apptainer exec --nv --pwd "$PWD" --bind "$PWD":"$PWD" "$SIF" \
    env SG_DIRECT=1 ARCH="$ARCH" make --no-print-directory single "$ID"
