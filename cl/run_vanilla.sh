#!/usr/bin/env bash
# Establish forgetting with plain fine-tuning: S0 (pretrained) -> S1 Street -> S2 Folk (DiscoForcing order).
# Run from the repo root on a GPU node:
#   mkdir -p logs && bash cl/run_vanilla.sh > logs/run_vanilla.log 2>&1 &
# StyleRetention runs only if the DiscoForcing FineDance style encoder is at $ENCODER.
set -euo pipefail

M=data/finedance/manifests
C=outputs/token_cache
R=runs/fd
ENCODER=${ENCODER:-pretrained/style_encoder_finedance.pt}
mkdir -p outputs/nll outputs/style logs $R

SETS=""
for s in s1_street:s1 s2_folk:s2 s3_classic:s3; do
  f=${s%:*}; n=${s#*:}
  SETS="$SETS --manifest ${n}_val=$M/${f}_val.jsonl --manifest ${n}_test=$M/${f}_test.jsonl"
done
TESTS="--manifest s1_test=$M/s1_street_test.jsonl --manifest s2_test=$M/s2_folk_test.jsonl --manifest s3_test=$M/s3_classic_test.jsonl"

echo "[1/5] S0: pretrained model on every val/test set (also builds the token caches)"
python cl/eval_nll.py $SETS --bf16 --out outputs/nll/fd_s0.json > logs/fd_s0.log 2>&1

echo "[2/5] S1 Street: pick the session length on S1 validation"
python cl/train_session.py --run_dir $R/s1 --train_manifest $M/s1_street_train.jsonl --train_cache $C/s1_train_w5_h2.5.pt \
  --eval s1_val=$C/s1_val_w5_h5.pt --steps 1000 --eval_every 100 --save_best s1_val > logs/fd_s1.log 2>&1
STEPS=$(python -c "import json; print(json.load(open('$R/s1/best.json'))['step'])")
echo "      session length = $STEPS steps"

echo "[3/5] S2 Folk: vanilla fine-tuning from S1 for $STEPS steps"
python cl/train_session.py --run_dir $R/vanilla/s2 --init_lm_state $R/s1/lm.pt --train_manifest $M/s2_folk_train.jsonl \
  --train_cache $C/s2_train_w5_h2.5.pt --eval s1_val=$C/s1_val_w5_h5.pt --eval s2_val=$C/s2_val_w5_h5.pt \
  --steps $STEPS --eval_every 50 > logs/fd_vanilla_s2.log 2>&1

echo "[4/5] held-out loss on the test sets after S1 and after S2"
python cl/eval_nll.py --lm_state $R/s1/lm.pt $TESTS --bf16 --out outputs/nll/fd_after_s1.json > logs/fd_eval_s1.log 2>&1
python cl/eval_nll.py --lm_state $R/vanilla/s2/lm.pt $TESTS --bf16 --out outputs/nll/fd_vanilla_after_s2.json > logs/fd_eval_s2.log 2>&1
python cl/forgetting_table.py --s0 outputs/nll/fd_s0.json \
  --method vanilla=outputs/nll/fd_after_s1.json,outputs/nll/fd_vanilla_after_s2.json

if [ -f "$ENCODER" ]; then
  echo "[5/5] StyleRetention of generated dance: reference rows, S0, after S1, after S2"
  ST="--encoder $ENCODER --manifest s1_test=$M/s1_street_test.jsonl --manifest s2_test=$M/s2_folk_test.jsonl"
  python cl/eval_style.py $ST --manifest s3_test=$M/s3_classic_test.jsonl --gt --recon --no_gen \
    --out outputs/style/fd_reference.json > logs/fd_style_ref.log 2>&1
  python cl/eval_style.py $ST --out outputs/style/fd_s0.json > logs/fd_style_s0.log 2>&1
  python cl/eval_style.py $ST --lm_state $R/s1/lm.pt --out outputs/style/fd_after_s1.json > logs/fd_style_s1.log 2>&1
  python cl/eval_style.py $ST --lm_state $R/vanilla/s2/lm.pt --out outputs/style/fd_vanilla_after_s2.json > logs/fd_style_s2.log 2>&1
  python cl/forgetting_table.py --key StyleRetention --higher_is_better --s0 outputs/style/fd_s0.json \
    --method vanilla=outputs/style/fd_after_s1.json,outputs/style/fd_vanilla_after_s2.json
  grep -h "StyleRetention" logs/fd_style_ref.log logs/fd_style_s0.log logs/fd_style_s1.log logs/fd_style_s2.log
else
  echo "[5/5] skipped StyleRetention: no style encoder at $ENCODER"
fi
echo "done"
