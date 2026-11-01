#!/usr/bin/env bash
# Host-side dispatcher: run a table inside the single-gate container.
#   dispatch.sh build
#   dispatch.sh run  X.Y.Z
#   dispatch.sh parse X.Y.Z
set -euo pipefail

mode=${1:?usage: dispatch.sh build|run|parse <id>}
shift || true

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

IMAGE=${IMAGE:-single-gate:cudaq0.16}
SIF="$REPO/single_gate/env/single-gate-cudaq0.16.sif"
RUNNER=${RUNNER:-auto}

in_container() {
    [ -n "${APPTAINER_CONTAINER:-}" ] || [ -n "${SINGULARITY_CONTAINER:-}" ] \
        || [ -f /.dockerenv ]
}
docker_ok()  { command -v docker >/dev/null 2>&1 && docker image inspect "$IMAGE" >/dev/null 2>&1; }
sif_ok()     { command -v apptainer >/dev/null 2>&1 && [ -f "$SIF" ]; }

if [ -z "${ARCH:-}" ] && [ "${SLURM:-0}" != "1" ]; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    ARCH="sm_${cap/./}"
fi
[ "${SLURM:-0}" = "1" ] || ARCH=${ARCH:-sm_75}

RUNNER_SELECTED=
choose() {
    case "$RUNNER" in
        docker)    docker_ok || { echo "!! Docker image $IMAGE not found"; exit 1; }; RUNNER_SELECTED=docker ;;
        apptainer) sif_ok || { echo "!! SIF $SIF not found"; exit 1; }; RUNNER_SELECTED=apptainer ;;
        auto)
            if docker_ok; then RUNNER_SELECTED=docker
            elif sif_ok; then RUNNER_SELECTED=apptainer
            else
                cat >&2 <<MSG
!! no container image found
   Docker image : $IMAGE
   Apptainer SIF: $SIF

Build one first (no automatic build):
   make single-docker
   make single-apptainer
MSG
                exit 1
            fi ;;
        *) echo "!! unknown RUNNER=$RUNNER (use auto|docker|apptainer)" >&2; exit 2 ;;
    esac
}

container_make() { # <make args...>
    local runner=$1; shift
    if [ "$runner" = docker ]; then
        docker run --rm --gpus all --entrypoint /bin/bash \
            -u "$(id -u):$(id -g)" \
            -e SG_DIRECT=1 -e ARCH="$ARCH" \
            -v "$REPO":/workspace -w /workspace "$IMAGE" \
            -c "make --no-print-directory $*"
    else
        apptainer exec --nv --pwd "$REPO" --bind "$REPO":"$REPO" "$SIF" \
            env SG_DIRECT=1 ARCH="$ARCH" make --no-print-directory "$@"
    fi
}

if in_container; then
    # should only happen if someone calls the dispatcher from inside explicitly
    exec make --no-print-directory SG_DIRECT=1 "$@"
fi

case "$mode" in
    build)
        choose; container_make "$RUNNER_SELECTED" single ;;
    parse)
        [ $# -eq 1 ] || { echo "usage: dispatch.sh parse ID" >&2; exit 1; }
        choose; container_make "$RUNNER_SELECTED" single_parse ID="$1" ;;
    run)
        [ $# -eq 1 ] || { echo "usage: dispatch.sh run ID" >&2; exit 1; }
        id=$1
        if [ "${SLURM:-0}" = "1" ]; then
            mkdir -p single_gate/results/raw
            echo "submitting $id ..." >&2
            sbatch ${SBATCH_ARGS:-} --parsable single_gate/jobs/slurm_container.sh "$id"
        else
            choose; container_make "$RUNNER_SELECTED" single "$id"
        fi ;;
    *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
