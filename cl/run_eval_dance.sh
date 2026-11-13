#!/usr/bin/env bash
# StyleRetention + standard dance metrics (FID_k, FID_g, Div_k, Div_g, BAS, Foot Skating Ratio) for the Vanilla
# runs, and the forgetting decision - DiscoForcing's continual-learning evaluation adapted to UniMuMo.
# Run from the repo root on a GPU node, after cl/run_vanilla.sh:
#   BABEL_DIR=/path/to/DiscoForcing/raw_data/cl_sessions/_style_encoder bash cl/run_eval_dance.sh > logs/eval_dance.log 2>&1 &
# A step is skipped when its output exists (delete the output to redo it). Generated joints are kept next to each
# result (*.joints.npz), so a result can be re-scored with `cl/eval_dance.py eval --joints_in` without regenerating.
set -euo pipefail

ENCODER=${ENCODER:-pretrained/style_encoder_finedance.pt}
SAMPLES=${SAMPLES:-1}          # generated samples per 10 s clip (DiscoForcing pools 3; use 3 for final numbers)
E=outputs/dance_eval
R=runs/fd
N=outputs/nll
mkdir -p $E logs

run() {  # run <output> <log> <command ...> : run the command unless <output> already exists
  local out=$1 log=$2; shift 2
  if [ -e "$out" ]; then echo "      skipped, $out exists"; return 0; fi
  "$@" > "$log" 2>&1
}
EV="python cl/eval_dance.py eval --encoder $ENCODER --ref_dir $E --samples $SAMPLES"

echo "[1/5] style encoder: DiscoForcing's recipe trained on UniMuMo's FineDance joints"
if [ ! -e "$ENCODER" ]; then
  : "${BABEL_DIR:?set BABEL_DIR to the DiscoForcing folder raw_data/cl_sessions/_style_encoder}"
fi
run $ENCODER logs/style_encoder.log python cl/train_style_encoder.py --babel_dir "${BABEL_DIR:-}" --out $ENCODER
sed -n '/^test songs, clean/,/^$/p;/^clean:/p;/max session mass/p' logs/style_encoder.log 2>/dev/null || true

echo "[2/5] references and noise floors from the real test dance"
run $E/noise.json logs/dance_build.log python cl/eval_dance.py build --encoder $ENCODER --ref_dir $E
sed -n '/Which metric/,$p' logs/dance_build.log 2>/dev/null || true

echo "[3/5] reference rows: real dance, motion-tokenizer reconstruction"
run $E/gt.json logs/dance_gt.log $EV --gt --out $E/gt.json
run $E/recon.json logs/dance_recon.log $EV --recon --out $E/recon.json

echo "[4/5] models: pretrained (S0), after S1, after S2 vanilla [, after S2 vanilla 500 steps]"
run $E/s0.json logs/dance_s0.log $EV --out $E/s0.json
run $E/after_s1.json logs/dance_after_s1.log $EV --lm_state $R/s1/lm.pt --out $E/after_s1.json
run $E/vanilla_after_s2.json logs/dance_vanilla_s2.log $EV --lm_state $R/vanilla/s2/lm.pt --out $E/vanilla_after_s2.json
METHODS="--method vanilla=$E/after_s1.json,$E/vanilla_after_s2.json"
NLL="--nll vanilla=$N/fd_after_s1.json,$N/fd_vanilla_after_s2.json"
if [ -f $R/vanilla_500steps/s2/lm.pt ]; then
  run $E/vanilla500_after_s2.json logs/dance_vanilla500_s2.log $EV --lm_state $R/vanilla_500steps/s2/lm.pt \
    --out $E/vanilla500_after_s2.json
  METHODS="$METHODS --method vanilla_500steps=$E/after_s1.json,$E/vanilla500_after_s2.json"
  if [ -f $N/fd_vanilla500_after_s2.json ]; then
    NLL="$NLL --nll vanilla_500steps=$N/fd_after_s1.json,$N/fd_vanilla500_after_s2.json"
  fi
fi
grep -h "_test " logs/dance_gt.log logs/dance_recon.log logs/dance_s0.log logs/dance_after_s1.log logs/dance_vanilla*_s2.log || true

echo "[5/5] forgetting decision"
python cl/forgetting_report.py --noise $E/noise.json --s0 $E/s0.json --gt $E/gt.json --recon $E/recon.json \
  $METHODS --nll_s0 $N/fd_s0.json $NLL | tee $E/report.txt
echo "done"
