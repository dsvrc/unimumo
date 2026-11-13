#!/usr/bin/env bash
# Establish forgetting with plain fine-tuning: S0 (pretrained) -> S1 Street -> S2 Folk (DiscoForcing order).
# Run from the repo root on a GPU node:
#   mkdir -p logs && bash cl/run_vanilla.sh > logs/run_vanilla.log 2>&1 &
# A step is skipped when its output already exists, so rerunning only does what is missing (for example the
# StyleRetention part once the style encoder is at $ENCODER). If a step crashes, delete its output first.
#
# Session length: S1 trains up to 1000 steps and keeps the step with the lowest S1 validation loss. Every
# later session gets the same number of passes (epochs) over its own windows, with the same warm-up in
# epochs (--schedule_from), so a small session (Folk is ~1/3 of Street) is not over-trained.
set -euo pipefail

M=data/finedance/manifests
C=outputs/token_cache
R=runs/fd
ENCODER=${ENCODER:-pretrained/style_encoder_finedance.pt}
mkdir -p outputs/nll outputs/style logs $R

run() {  # run <output> <log> <command ...> : run the command unless <output> already exists
  local out=$1 log=$2; shift 2
  if [ -e "$out" ]; then echo "      skipped, $out exists"; return 0; fi
  "$@" > "$log" 2>&1
}

SETS=""
for s in s1_street:s1 s2_folk:s2 s3_classic:s3; do
  f=${s%:*}; n=${s#*:}
  SETS="$SETS --manifest ${n}_val=$M/${f}_val.jsonl --manifest ${n}_test=$M/${f}_test.jsonl"
done
TESTS="--manifest s1_test=$M/s1_street_test.jsonl --manifest s2_test=$M/s2_folk_test.jsonl --manifest s3_test=$M/s3_classic_test.jsonl"

echo "[1/5] S0: pretrained model on every val/test set (also builds the token caches)"
run outputs/nll/fd_s0.json logs/fd_s0.log \
  python cl/eval_nll.py $SETS --bf16 --out outputs/nll/fd_s0.json

echo "[2/5] S1 Street: up to 1000 steps, keep the step with the lowest S1 validation loss"
run $R/s1/lm.pt logs/fd_s1.log \
  python cl/train_session.py --run_dir $R/s1 --train_manifest $M/s1_street_train.jsonl --train_cache $C/s1_train_w5_h2.5.pt \
  --eval s1_val=$C/s1_val_w5_h5.pt --steps 1000 --eval_every 100 --save_best s1_val

echo "[3/5] S2 Folk: vanilla fine-tuning from S1, same epochs and warm-up (in epochs) as S1"
run $R/vanilla/s2/lm.pt logs/fd_vanilla_s2.log \
  python cl/train_session.py --run_dir $R/vanilla/s2 --init_lm_state $R/s1/lm.pt --train_manifest $M/s2_folk_train.jsonl \
  --train_cache $C/s2_train_w5_h2.5.pt --eval s1_val=$C/s1_val_w5_h5.pt --eval s2_val=$C/s2_val_w5_h5.pt \
  --schedule_from $R/s1 --eval_every 25
grep -E "^train windows|^schedule from" logs/fd_vanilla_s2.log || true

echo "[4/5] held-out loss on the test sets after S1 and after S2"
run outputs/nll/fd_after_s1.json logs/fd_eval_s1.log \
  python cl/eval_nll.py --lm_state $R/s1/lm.pt $TESTS --bf16 --out outputs/nll/fd_after_s1.json
run outputs/nll/fd_vanilla_after_s2.json logs/fd_eval_s2.log \
  python cl/eval_nll.py --lm_state $R/vanilla/s2/lm.pt $TESTS --bf16 --out outputs/nll/fd_vanilla_after_s2.json
EXTRA=""   # the earlier over-trained S2 (500 steps), if it was kept under runs/fd/vanilla_500steps
if [ -f outputs/nll/fd_vanilla500_after_s2.json ]; then
  EXTRA="--method vanilla_500steps=outputs/nll/fd_after_s1.json,outputs/nll/fd_vanilla500_after_s2.json"
fi
python cl/forgetting_table.py --s0 outputs/nll/fd_s0.json \
  --method vanilla=outputs/nll/fd_after_s1.json,outputs/nll/fd_vanilla_after_s2.json $EXTRA

if [ -f "$ENCODER" ]; then
  echo "[5/5] StyleRetention of generated dance: reference rows, S0, after S1, after S2"
  ST="--encoder $ENCODER --manifest s1_test=$M/s1_street_test.jsonl --manifest s2_test=$M/s2_folk_test.jsonl"
  ST="$ST --manifest s3_test=$M/s3_classic_test.jsonl"
  run outputs/style/fd_reference.json logs/fd_style_ref.log \
    python cl/eval_style.py $ST --gt --recon --no_gen --out outputs/style/fd_reference.json
  run outputs/style/fd_s0.json logs/fd_style_s0.log \
    python cl/eval_style.py $ST --out outputs/style/fd_s0.json
  run outputs/style/fd_after_s1.json logs/fd_style_s1.log \
    python cl/eval_style.py $ST --lm_state $R/s1/lm.pt --out outputs/style/fd_after_s1.json
  run outputs/style/fd_vanilla_after_s2.json logs/fd_style_s2.log \
    python cl/eval_style.py $ST --lm_state $R/vanilla/s2/lm.pt --out outputs/style/fd_vanilla_after_s2.json
  EXTRA=""
  if [ -f $R/vanilla_500steps/s2/lm.pt ]; then
    run outputs/style/fd_vanilla500_after_s2.json logs/fd_style_s2_500steps.log \
      python cl/eval_style.py $ST --lm_state $R/vanilla_500steps/s2/lm.pt --out outputs/style/fd_vanilla500_after_s2.json
    EXTRA="--method vanilla_500steps=outputs/style/fd_after_s1.json,outputs/style/fd_vanilla500_after_s2.json"
  fi
  python cl/forgetting_table.py --key StyleRetention --higher_is_better --s0 outputs/style/fd_s0.json \
    --method vanilla=outputs/style/fd_after_s1.json,outputs/style/fd_vanilla_after_s2.json $EXTRA
  for f in ref s0 s1 s2 s2_500steps; do
    if [ -f logs/fd_style_$f.log ]; then echo "--- $f"; grep "StyleRetention" logs/fd_style_$f.log || true; fi
  done
else
  echo "[5/5] skipped StyleRetention: no style encoder at $ENCODER"
fi
echo "done"
