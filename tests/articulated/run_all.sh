#!/usr/bin/env bash
# Runs the articulated micro-Doppler validation: identity tests, signal-level tests, and the end-to-end example.
# Usage (from the repository root, in the WaveVerse environment):  bash tests/articulated/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tests/articulated/test_vertex_identity.py
python tests/articulated/test_articulated_burst.py
python examples/articulated_temporal_example.py
echo "articulated micro-Doppler validation: all checks passed"
