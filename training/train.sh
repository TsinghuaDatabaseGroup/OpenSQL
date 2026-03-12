#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'USAGE'
Usage:
  bash training/train.sh <stage> [stage_args...]

Stages:
  global-sft    -> schema_linking/SFT/launch_global_sft.sh
  global-dpo    -> schema_linking/DPO/launch_global_dpo.sh
  local-linker  -> schema_linking/Local/launch_local.sh (alias: local)
  generator     -> generation/launch_generation.sh (alias: generation)
  selector      -> selector/launch_selector.sh

Example:
  bash training/train.sh global-sft 1e-5 3 /path/to/base-model
USAGE
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

stage="$1"
shift

case "${stage}" in
    global-sft)
        exec "${SCRIPT_DIR}/schema_linking/SFT/launch_global_sft.sh" "$@"
        ;;
    global-dpo)
        exec "${SCRIPT_DIR}/schema_linking/DPO/launch_global_dpo.sh" "$@"
        ;;
    local-linker|local)
        exec "${SCRIPT_DIR}/schema_linking/Local/launch_local.sh" "$@"
        ;;
    generator|generation)
        exec "${SCRIPT_DIR}/generation/launch_generation.sh" "$@"
        ;;
    selector)
        exec "${SCRIPT_DIR}/selector/launch_selector.sh" "$@"
        ;;
    *)
        echo "Unknown stage: ${stage}" >&2
        usage
        exit 1
        ;;
esac
