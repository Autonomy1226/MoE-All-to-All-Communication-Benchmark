#!/usr/bin/env bash
# Batch runner for MoE communication benchmark experiments.
# Usage: bash scripts/run_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR="${1:-results}"

echo "=============================================="
echo "  MoE All-to-All Communication Benchmark"
echo "  Output directory: $OUTDIR"
echo "=============================================="

# Quick comparison (small scale, fast)
echo ""
echo "[1/4] Quick comparison (3 strategies, small scale)..."
python scripts/run_benchmark.py \
    --compare-all \
    --hidden 512 \
    --tokens 512 \
    --iterations 30 \
    --output "$OUTDIR/quick"

# Full comparison at standard scale
echo ""
echo "[2/4] Standard comparison (3 strategies, 1024 tokens)..."
python scripts/run_benchmark.py \
    --compare-all \
    --hidden 1024 \
    --tokens 1024 \
    --iterations 50 \
    --output "$OUTDIR/standard"

# Token count sweep
echo ""
echo "[3/4] Token count sweep..."
python scripts/run_benchmark.py \
    --sweep-tokens \
    --hidden 1024 \
    --iterations 30 \
    --output "$OUTDIR/sweep_tokens"

# Hidden dimension sweep
echo ""
echo "[4/4] Hidden dimension sweep..."
python scripts/run_benchmark.py \
    --sweep-hidden \
    --iterations 30 \
    --output "$OUTDIR/sweep_hidden"

echo ""
echo "=============================================="
echo "  All experiments complete!"
echo "  Results saved in: $OUTDIR"
echo "=============================================="
