#!/usr/bin/env bash
#SBATCH --job-name=single-gate
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=single_gate/results/raw/slurm_container-%j.out

# Generic Slurm wrapper: sbatch slurm_container.sh <make-target>
# (the target is a stamp path under single_gate/results/raw/)
set -euo pipefail

TARGET=${1:?usage: slurm_container.sh <make-target>}
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
SIF="$REPO/single_gate/env/single-gate-cudaq0.16.sif"
ARCH=${ARCH:-sm_90}
cd "$REPO"

if [ ! -f "$SIF" ]; then
    echo "!! SIF not found: $SIF" >&2
    exit 1
fi

# some compute nodes lack apptainer in PATH (job 670426 died on hgpn167);
# fall back to the absolute path / singularity
APPTAINER_BIN=$(command -v apptainer || command -v singularity || echo /usr/bin/apptainer)
exec "$APPTAINER_BIN" exec --nv --pwd "$PWD" --bind "$PWD":"$PWD" "$SIF" \
    env SG_DIRECT=1 ARCH="$ARCH" make --no-print-directory "$TARGET"
