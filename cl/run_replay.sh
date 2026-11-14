#!/usr/bin/env bash
# Continual run of one method after the shared S1 model (cl/run_vanilla.sh), as DiscoForcing scripts/cl/run_method.sh:
#   S2 + buffer(S1) ; S3 + buffer(S1) + buffer(S2)          (the pretrained model gets no buffer)
# For k = 2, 3: build the buffer of session k-1 with this method's model after k-1 (10 % of its training clips,
# cl/make_buffer.py), train S_k from this method's S_(k-1) with replay (same epochs and warm-up as S1), then score it
# (held-out loss and dance metrics). vanilla = the same chain without buffers. Finished steps are skipped.
#
#   for m in vanilla random tmr tmr_joint; do bash cl/run_replay.sh $m; done > logs/replay_all.log 2>&1 &
#
# Methods, as DiscoForcing's variants: vanilla | <selection> | <selection>_kd, with selection one of random, tmr,
# tmr_joint, tmr_joint_neff (cl/make_buffer.py). <selection>_kd replays that selection's buffer by distillation
# (cl/train_session.py --distill: teacher = the model the session starts from, kd_weight 1.0, gt_weight 0.25).
# Each method has its own buffer folder; its S1 buffer is copied from <selection>'s when that exists (same S1 model).
# Run the methods one after another: the first S3 run builds the shared S3 training cache and the first buffer
# build the shared candidate caches (parallel runs could write them at the same time).
# env: BALANCE (equal | session | uniform; equal = the current session and every earlier session weigh the same in
#      the batch and the loss), RATIO (0.10), SAMPLES (generated samples per clip for the dance metrics, 1), ENCODER
set -euo pipefail
METHOD=${1:?usage: run_replay.sh <vanilla | random | tmr | tmr_joint | tmr_joint_neff | <selection>_kd>}
case $METHOD in
  vanilla) SELECT= ; DISTILL= ;;
  random|tmr|tmr_joint|tmr_joint_neff) SELECT=$METHOD ; DISTILL= ;;
  random_kd|tmr_kd|tmr_joint_kd|tmr_joint_neff_kd) SELECT=${METHOD%_kd} ; DISTILL=--distill ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac
RATIO=${RATIO:-0.10}
BALANCE=${BALANCE:-equal}
SAMPLES=${SAMPLES:-1}
ENCODER=${ENCODER:-pretrained/style_encoder_finedance.pt}
M=data/finedance/manifests
C=outputs/token_cache
R=runs/fd
B=buffers/fd/$METHOD
N=outputs/nll
E=outputs/dance_eval
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$PWD/outputs/.numba_cache}
mkdir -p $B $N $E logs $NUMBA_CACHE_DIR
NAME=(- s1_street s2_folk s3_classic)
TESTS="--manifest s1_test=$M/s1_street_test.jsonl --manifest s2_test=$M/s2_folk_test.jsonl --manifest s3_test=$M/s3_classic_test.jsonl"

run() {  # run <output> <log> <command ...> : run the command unless <output> already exists
  local out=$1 log=$2; shift 2
  if [ -e "$out" ]; then echo "      skipped, $out exists"; return 0; fi
  "$@" > "$log" 2>&1
}
model_dir() { if [ "$1" -eq 1 ]; then echo $R/s1; else echo $R/$METHOD/s$1; fi; }
[ -f $R/s1/lm.pt ] || { echo "no S1 model at $R/s1/lm.pt: run cl/run_vanilla.sh first"; exit 1; }
[ -f $E/noise.json ] || { echo "no dance-metric references at $E: run cl/run_eval_dance.sh first"; exit 1; }

for k in 2 3; do
  j=$((k - 1))
  echo "[$METHOD] S$k ($(date))"
  REPLAY=""
  if [ -n "$SELECT" ]; then
    for i in $(seq 1 $j); do
      if [ "$i" -eq 1 ] && [ ! -f $B/s1.pt ] && [ "$SELECT" != "$METHOD" ] && [ -f buffers/fd/$SELECT/s1.pt ]; then
        cp buffers/fd/$SELECT/s1.pt $B/s1.pt  # same selection on the same shared S1 model: the same buffer
        cp buffers/fd/$SELECT/s1.json $B/s1.json 2>/dev/null || true
        echo "      buffer S1 copied from buffers/fd/$SELECT/s1.pt"
      fi
      LM=""
      [ "$SELECT" != random ] && LM="--lm_state $(model_dir $i)/lm.pt"
      run $B/s$i.pt logs/buffer_${METHOD}_s$i.log python cl/make_buffer.py --method $SELECT --session $i \
        --manifest $M/${NAME[$i]}_train.jsonl $LM --ratio $RATIO --out $B/s$i.pt
      grep -hE "^\[tmr|^session|^saved" logs/buffer_${METHOD}_s$i.log 2>/dev/null || true
      REPLAY="$REPLAY --replay $B/s$i.pt"
    done
    REPLAY="$REPLAY --replay_balance $BALANCE $DISTILL"
  fi
  EVALS=""
  for i in $(seq 1 $k); do EVALS="$EVALS --eval s${i}_val=$C/s${i}_val_w5_h5.pt"; done
  run $(model_dir $k)/lm.pt logs/fd_${METHOD}_s$k.log \
    python cl/train_session.py --run_dir $(model_dir $k) --init_lm_state $(model_dir $j)/lm.pt \
    --train_manifest $M/${NAME[$k]}_train.jsonl --train_cache $C/s${k}_train_w5_h2.5.pt $EVALS \
    --schedule_from $R/s1 --eval_every 25 $REPLAY
  grep -hE "^train windows|^replay buffer|^each step|^distillation" logs/fd_${METHOD}_s$k.log 2>/dev/null || true
  run $N/fd_${METHOD}_after_s$k.json logs/fd_eval_${METHOD}_s$k.log \
    python cl/eval_nll.py --lm_state $(model_dir $k)/lm.pt $TESTS --bf16 --out $N/fd_${METHOD}_after_s$k.json
  run $E/${METHOD}_after_s$k.json logs/dance_${METHOD}_s$k.log \
    python cl/eval_dance.py eval --encoder $ENCODER --ref_dir $E --samples $SAMPLES --lm_state $(model_dir $k)/lm.pt \
    --out $E/${METHOD}_after_s$k.json
  grep -h "_test " logs/dance_${METHOD}_s$k.log 2>/dev/null || true
done

echo "[report] every method that has finished S3"
METHODS=""
NLL=""
for m in vanilla random tmr tmr_joint tmr_joint_neff random_kd tmr_kd tmr_joint_kd tmr_joint_neff_kd; do
  if [ -f $E/${m}_after_s2.json ] && [ -f $E/${m}_after_s3.json ]; then
    METHODS="$METHODS --method $m=$E/after_s1.json,$E/${m}_after_s2.json,$E/${m}_after_s3.json"
    NLL="$NLL --nll $m=$N/fd_after_s1.json,$N/fd_${m}_after_s2.json,$N/fd_${m}_after_s3.json"
  fi
done
python cl/forgetting_report.py --noise $E/noise.json --s0 $E/s0.json --gt $E/gt.json --recon $E/recon.json \
  $METHODS --nll_s0 $N/fd_s0.json $NLL | tee $E/report_replay.txt
echo "[$METHOD] done ($(date))"
