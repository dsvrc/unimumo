#!/usr/bin/env bash
# Forgetting report (cl/forgetting_report.py) of every method that has finished S3, in the order of the paper's table;
# the full report goes to outputs/dance_eval/report_replay.txt.
#
#   bash cl/report_replay.sh                    print the full report
#   COMPARE_ONLY=1 bash cl/report_replay.sh     print only the comparison of the methods
# env: ORDER (methods and their order; default every method cl/run_replay.sh knows)
set -euo pipefail
ORDER=${ORDER:-"vanilla random random_star ocs gcr mir csrel tmr tmr_joint tmr_joint_neff random_kd ocs_kd gcr_kd mir_kd csrel_kd tmr_kd tmr_joint_kd tmr_joint_neff_kd"}
E=outputs/dance_eval
N=outputs/nll
METHODS=""
NLL=""
COUNT=0
for m in $ORDER; do
  if [ -f $E/${m}_after_s2.json ] && [ -f $E/${m}_after_s3.json ]; then
    METHODS="$METHODS --method $m=$E/after_s1.json,$E/${m}_after_s2.json,$E/${m}_after_s3.json"
    COUNT=$((COUNT + 1))
    if [ -f $N/fd_${m}_after_s2.json ] && [ -f $N/fd_${m}_after_s3.json ]; then
      NLL="$NLL --nll $m=$N/fd_after_s1.json,$N/fd_${m}_after_s2.json,$N/fd_${m}_after_s3.json"
    fi
  fi
done
[ -n "$METHODS" ] || { echo "[report] no method has finished S3 yet"; exit 0; }
echo "[report] $COUNT methods have finished S3"
python cl/forgetting_report.py --noise $E/noise.json --s0 $E/s0.json --gt $E/gt.json --recon $E/recon.json \
  $METHODS --nll_s0 $N/fd_s0.json $NLL > $E/report_replay.txt
L=$(grep -n "^Comparison of the methods" $E/report_replay.txt | head -1 | cut -d: -f1 || true)
if [ "${COMPARE_ONLY:-0}" = 1 ] && [ -n "$L" ]; then
  tail -n +$((L - 1)) $E/report_replay.txt
else
  cat $E/report_replay.txt
fi
echo "[report] full report: $E/report_replay.txt"
