#!/bin/bash
# Step 2 — broker-cost experiment launcher.
#
# Runs TWO 28-pair walk-forwards concurrently through the same
# CostAwareSimulator (full round-trip spread + optional commission):
#
#   1. corrected-OANDA: OANDA spreads, commission = 0
#      → data/walk_forward_oanda_v2/<PAIR>.json
#   2. ECN:             ECN spreads, commission = 0.8 bps round-trip
#      → data/walk_forward_ecn/<PAIR>.json
#
# Same signal, same grid, same fold windows, same data — only the cost inputs
# differ. That is the controlled-experiment requirement.
#
# Parallelism: 6 pair-workers per cost model × 2 cost models = 12 concurrent
# workers, matching the 12 logical cores on this Windows box.
#
# Per-pair logs: logs/oanda_v2_<PAIR>.log + logs/ecn_<PAIR>.log
# Launcher log:  logs/step2_broker.log
#
# History download is assumed already cached (full 2022-01-02 → now). If not,
# run scripts/run_step2.sh first (it does Phase 1 sequentially across 28 pairs).

set -u
cd "$(dirname "$0")/.."

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

FROM=2022-01-01
TRAIN=365
TEST=90
STEP=90

PAIRS=(EUR_USD USD_JPY GBP_USD USD_CHF AUD_USD NZD_USD USD_CAD
       EUR_GBP EUR_JPY EUR_CHF EUR_AUD EUR_NZD EUR_CAD
       GBP_JPY GBP_CHF GBP_AUD GBP_NZD GBP_CAD
       AUD_JPY AUD_CHF AUD_NZD AUD_CAD
       NZD_JPY NZD_CHF NZD_CAD CHF_JPY CAD_JPY CAD_CHF)

mkdir -p logs data/walk_forward_oanda_v2 data/walk_forward_ecn

echo "[$(date -Iseconds)] Step 2 broker experiment started"
echo "  Pairs:        ${#PAIRS[@]} (per cost model)"
echo "  Period:       ${FROM} -> now"
echo "  Window:       train=${TRAIN}d test=${TEST}d step=${STEP}d"
echo "  Cost models:  oanda (full round-trip spread, commission=0)"
echo "                ecn   (ECN spreads, commission=0.8 bps)"
echo "  Concurrency:  6 workers per model x 2 models = 12 total"
echo ""

run_one_pair() {
    local model="$1"
    local pair="$2"
    local out_dir="$3"
    local t0=$(date +%s)
    python scripts/run_ecn_experiment.py \
        --cost-model "${model}" \
        -f "${FROM}" \
        --pairs "${pair}" \
        --train-days "${TRAIN}" \
        --test-days "${TEST}" \
        --step-days "${STEP}" \
        --skip-download \
        -o "${out_dir}" \
        > "logs/${model}_${pair}.log" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    if [ $rc -eq 0 ]; then
        echo "[$(date -Iseconds)]   ${model} ${pair} done in ${elapsed}s"
    else
        echo "[$(date -Iseconds)]   ${model} ${pair} FAILED rc=${rc} (${elapsed}s)"
    fi
}
export -f run_one_pair
export FROM TRAIN TEST STEP

# Launch 6 OANDA-v2 workers + 6 ECN workers concurrently. xargs -P 6 on each
# stream means each stream caps at 6 in-flight jobs.

oanda_stream() {
    printf '%s\n' "${PAIRS[@]}" | xargs -n 1 -P 6 -I {} bash -c \
        'run_one_pair oanda "$@" data/walk_forward_oanda_v2' _ {}
}
ecn_stream() {
    printf '%s\n' "${PAIRS[@]}" | xargs -n 1 -P 6 -I {} bash -c \
        'run_one_pair ecn "$@" data/walk_forward_ecn' _ {}
}
export -f oanda_stream ecn_stream

echo "[$(date -Iseconds)] Launching 6-way OANDA-v2 stream + 6-way ECN stream"
oanda_stream &
ecn_stream &
wait

echo ""
echo "[$(date -Iseconds)] Both streams complete"
echo ""
oanda_n=$(ls -1 data/walk_forward_oanda_v2/*.json 2>/dev/null | wc -l)
ecn_n=$(ls -1 data/walk_forward_ecn/*.json 2>/dev/null | wc -l)
echo "OANDA-v2 JSONs: ${oanda_n} / ${#PAIRS[@]}"
echo "ECN JSONs:      ${ecn_n} / ${#PAIRS[@]}"
echo ""
echo "Next:"
echo "  python scripts/deflated_sharpe_analysis.py --input-dir data/walk_forward_oanda_v2 --output data/walk_forward_oanda_v2/_dsr_report.json"
echo "  python scripts/deflated_sharpe_analysis.py --input-dir data/walk_forward_ecn       --output data/walk_forward_ecn/_dsr_report.json"
