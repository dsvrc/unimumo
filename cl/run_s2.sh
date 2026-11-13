#!/usr/bin/env bash
# Session 2 (FineDance Classic) for Vanilla, ER and provisional TMR, starting from the shared S1 checkpoint.
# Run from the repo root on a GPU node:   bash cl/run_s2.sh > logs/run_s2.log 2>&1 &
set -euo pipefail

STEPS=300                                  # fixed session length (best S1 validation step)
M=data/finedance/manifests
C=outputs/token_cache
TESTS="--manifest s1_test=$M/s1_street_test.jsonl --manifest s2_test=$M/s2_classic_test.jsonl --manifest s3_test=$M/s3_folk_test.jsonl"
mkdir -p outputs/nll runs/buffers logs

echo "[1/4] test scores right after S1"
python cl/eval_nll.py --lm_state runs/s1/lm.pt $TESTS --bf16 --out outputs/nll/after_s1.json > logs/eval_after_s1.log 2>&1

echo "[2/4] replay buffers from S1 (10% of its training windows)"
python cl/make_buffer.py random --train_cache runs/s1/train_tokens.pt --out runs/buffers/er_s1.pt > logs/buf_er_s1.log 2>&1
python cl/make_buffer.py tmr --lm_state runs/s1/lm.pt --train_cache runs/s1/train_tokens.pt --out runs/buffers/tmr_s1.pt > logs/buf_tmr_s1.log 2>&1

echo "[3/4] session 2: $STEPS steps per method"
COMMON="--init_lm_state runs/s1/lm.pt --train_manifest $M/s2_classic_train.jsonl --train_cache $C/s2_train_w5_h2.5.pt \
  --eval s1_val=$C/s1_val_w5_h5.pt --eval s2_val=$C/s2_val_w5_h5.pt --steps $STEPS --eval_every 50"
python cl/train_session.py --run_dir runs/vanilla/s2 $COMMON > logs/vanilla_s2.log 2>&1
python cl/train_session.py --run_dir runs/er/s2 $COMMON --replay runs/buffers/er_s1.pt > logs/er_s2.log 2>&1
python cl/train_session.py --run_dir runs/tmr/s2 $COMMON --replay runs/buffers/tmr_s1.pt > logs/tmr_s2.log 2>&1

echo "[4/4] test scores after S2"
for m in vanilla er tmr; do
  python cl/eval_nll.py --lm_state runs/$m/s2/lm.pt $TESTS --bf16 --out outputs/nll/${m}_after_s2.json > logs/eval_${m}_after_s2.log 2>&1
done

python cl/forgetting_table.py --s0 outputs/nll/fd_s0.json \
  --method vanilla=outputs/nll/after_s1.json,outputs/nll/vanilla_after_s2.json \
  --method er=outputs/nll/after_s1.json,outputs/nll/er_after_s2.json \
  --method tmr=outputs/nll/after_s1.json,outputs/nll/tmr_after_s2.json
echo "done"
