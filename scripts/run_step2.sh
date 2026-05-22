#!/bin/bash
# Step 2 launcher — parallel walk-forward across all 28 G10 pairs.
#
# Phase 1: 4-way parallel OANDA download (2022-01-01 → now) for all 28 pairs.
# Phase 2: 8-way parallel walk-forward (one process per pair, --skip-download).
#
# All outputs land in data/walk_forward/<PAIR>.json
# Per-pair logs in logs/wf_<PAIR>.log; launcher log in logs/step2_launcher.log

set -u

cd "$(dirname "$0")/.."

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

FROM=2022-01-01
TRAIN=365
TEST=90
STEP=90

CHUNK_A=(EUR_USD USD_JPY GBP_USD USD_CHF AUD_USD NZD_USD USD_CAD)
CHUNK_B=(EUR_GBP EUR_JPY EUR_CHF EUR_AUD EUR_NZD EUR_CAD GBP_JPY)
CHUNK_C=(GBP_CHF GBP_AUD GBP_NZD GBP_CAD AUD_JPY AUD_CHF AUD_NZD)
CHUNK_D=(AUD_CAD NZD_JPY NZD_CHF NZD_CAD CHF_JPY CAD_JPY CAD_CHF)

ALL_PAIRS=("${CHUNK_A[@]}" "${CHUNK_B[@]}" "${CHUNK_C[@]}" "${CHUNK_D[@]}")

mkdir -p logs data/walk_forward

echo "[$(date -Iseconds)] Step 2 launcher started"
echo "  Pairs:    ${#ALL_PAIRS[@]}"
echo "  Period:   ${FROM} -> now"
echo "  Window:   train=${TRAIN}d test=${TEST}d step=${STEP}d"
echo ""

# ── Phase 1: 4-way parallel download ─────────────────────────────────────────
echo "[$(date -Iseconds)] Phase 1: downloading history (4-way parallel)"

run_download() {
    local chunk_id="$1"; shift
    python -c "
import asyncio, sys
sys.path.insert(0, '.')
from scripts.backtest_all_pairs import download_all_history
pairs = sys.argv[1:]
asyncio.run(download_all_history(pairs, '${FROM}', None))
" "$@" > "logs/dl_${chunk_id}.log" 2>&1
    echo "[$(date -Iseconds)]   download chunk ${chunk_id} done"
}

run_download A "${CHUNK_A[@]}" &
run_download B "${CHUNK_B[@]}" &
run_download C "${CHUNK_C[@]}" &
run_download D "${CHUNK_D[@]}" &
wait
echo "[$(date -Iseconds)] Phase 1: all downloads complete"
echo ""

# ── Phase 2: 8-way parallel walk-forward ─────────────────────────────────────
echo "[$(date -Iseconds)] Phase 2: walk-forward grid search (8-way parallel)"

run_one_pair() {
    local pair="$1"
    local t0=$(date +%s)
    python scripts/walk_forward_optimize.py \
        -f "${FROM}" \
        --pairs "${pair}" \
        --train-days "${TRAIN}" \
        --test-days "${TEST}" \
        --step-days "${STEP}" \
        --skip-download \
        > "logs/wf_${pair}.log" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        echo "[$(date -Iseconds)]   ${pair} done in ${elapsed}s"
    else
        echo "[$(date -Iseconds)]   ${pair} FAILED (rc=${rc}, ${elapsed}s)"
    fi
}

export -f run_one_pair
export FROM TRAIN TEST STEP

printf '%s\n' "${ALL_PAIRS[@]}" | xargs -n 1 -P 8 -I {} bash -c 'run_one_pair "$@"' _ {}

echo ""
echo "[$(date -Iseconds)] Phase 2: all walk-forward jobs complete"
echo ""
ls -1 data/walk_forward/*.json 2>/dev/null | wc -l | xargs -I {} echo "JSON files produced: {} / ${#ALL_PAIRS[@]}"
echo ""
echo "Next: python scripts/deflated_sharpe_analysis.py --input-dir data/walk_forward"
