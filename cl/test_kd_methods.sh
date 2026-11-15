#!/usr/bin/env bash
# Testing of the distillation rows (TMR+KD = tmr_joint_neff_kd, GCR+KD = gcr_kd, CSReL+KD = csrel_kd, MIR+KD = mir_kd):
# for every session already trained (S2, S3), the held-out loss (cl/eval_nll.py, every session's test songs) and the
# dance metrics (cl/eval_dance.py: StyleRetention, StyleAcc, DanceRate, StyleFD, FID_k, FID_g, Div, BAS, FSR) unless
# already scored, then the forgetting report of these rows next to vanilla, ER+KD and their ground-truth-replay versions
# where those have finished. Trains nothing (training: cl/run_kd_methods.sh). A method whose chain is running in
# another process is skipped: that chain scores it.
#
#   bash cl/test_kd_methods.sh > logs/test_kd.log 2>&1
#
# env: METHODS, REPORT_ORDER, FULL (1 = print the full report instead of the comparison), SAMPLES, ENCODER
set -uo pipefail
METHODS=${METHODS:-"tmr_joint_neff_kd gcr_kd csrel_kd mir_kd"}
REPORT_ORDER=${REPORT_ORDER:-"vanilla random random_kd gcr gcr_kd mir mir_kd csrel csrel_kd tmr_joint_neff tmr_joint_neff_kd"}
HERE=$(dirname "$0")
mkdir -p logs
for m in $METHODS; do
  STAGE=test REPORT=0 bash "$HERE/run_replay.sh" $m
  status=$?
  case $status in
    0) ;;
    3) echo "[test] $m is running in another process: its chain scores it" ;;
    *) echo "[test] $m FAILED with status $status" ;;
  esac
done
if [ "${FULL:-0}" = 1 ]; then
  ORDER="$REPORT_ORDER" bash "$HERE/report_replay.sh"
else
  COMPARE_ONLY=1 ORDER="$REPORT_ORDER" bash "$HERE/report_replay.sh"
fi
