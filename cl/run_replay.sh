#!/usr/bin/env bash
# Continual run of one method after the shared S1 model (cl/run_vanilla.sh), as DiscoForcing scripts/cl/run_method.sh:
#   S2 + buffer(S1) ; S3 + buffer(S1) + buffer(S2)          (the pretrained model gets no buffer)
# For k = 2, 3: build the buffer of session k-1 with this method's model after k-1 (10 % of its training clips,
# cl/make_buffer.py), train S_k from this method's S_(k-1) with replay (same epochs and warm-up as S1), then score it
# (held-out loss and dance metrics). vanilla = the same chain without buffers. Finished steps are skipped, so a
# crashed run restarts with the same command. Every method of the table: cl/run_all_methods.sh.
#
#   bash cl/run_replay.sh <method> > logs/replay_<method>.log 2>&1 &
#
# Methods (DiscoForcing's names; the paper's DiscoForcing table in brackets):
#   vanilla                                    no replay [Fine-tuning]
#   random | ocs | gcr | mir | csrel           baseline buffers, ground-truth replay [ER, OCS, GCR, MIR, CSReL]
#   random_star                                ER + STAR (cl/train_session.py --star) [STAR]
#   tmr | tmr_joint | tmr_joint_neff           TMR buffers, ground-truth replay [TMR = tmr_joint_neff]
#   <selection>_kd                             that selection's buffer replayed by distillation (--distill: teacher =
#                                              the model the session starts from) [ER+KD = random_kd, GCR+KD =
#                                              gcr_kd, TMR+KD = tmr_joint_neff_kd]; random_distill = random_kd
# Each method has its own buffer folder. Its S1 buffer (same selection on the same shared S1 model) is copied from a
# finished method with the same selection when there is one.
# env: BALANCE (equal | session | uniform; equal = the current session and every earlier session weigh the same in
#      the batch and the loss), RATIO (0.10), SAMPLES (generated samples per clip for the dance metrics, 1), ENCODER,
#      REPORT (1 = print the forgetting report of every finished method at the end; 0 = skip), DRY (1 = print the
#      commands without running them), BUFFER_ARGS (more cl/make_buffer.py options, e.g. "--score_bf16" if MIR /
#      CSReL run out of GPU memory in float32)
set -euo pipefail
METHOD=${1:?usage: run_replay.sh <vanilla | random | ocs | gcr | mir | csrel | random_star | tmr | tmr_joint | tmr_joint_neff | <selection>_kd>}
[ "$METHOD" = random_distill ] && METHOD=random_kd
SELECTIONS="random ocs gcr mir csrel tmr tmr_joint tmr_joint_neff"
case $METHOD in
  vanilla) SELECT= ; TRAIN= ;;
  random_star) SELECT=random ; TRAIN=--star ;;
  *_kd) SELECT=${METHOD%_kd} ; TRAIN=--distill ;;
  *) SELECT=$METHOD ; TRAIN= ;;
esac
if [ -n "$SELECT" ] && [[ " $SELECTIONS " != *" $SELECT "* ]]; then echo "unknown method $METHOD"; exit 1; fi
RATIO=${RATIO:-0.10}
BALANCE=${BALANCE:-equal}
SAMPLES=${SAMPLES:-1}
ENCODER=${ENCODER:-pretrained/style_encoder_finedance.pt}
REPORT=${REPORT:-1}
DRY=${DRY:-0}
M=data/finedance/manifests
C=outputs/token_cache
R=runs/fd
B=buffers/fd/$METHOD
N=outputs/nll
E=outputs/dance_eval
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$PWD/outputs/.numba_cache}
[ "$DRY" = 1 ] || mkdir -p $B $N $E logs $NUMBA_CACHE_DIR
NAME=(- s1_street s2_folk s3_classic)
TESTS="--manifest s1_test=$M/s1_street_test.jsonl --manifest s2_test=$M/s2_folk_test.jsonl --manifest s3_test=$M/s3_classic_test.jsonl"

run() {  # run <output> <log> <command ...> : run the command unless <output> already exists
  local out=$1 log=$2; shift 2
  if [ -e "$out" ]; then echo "      skipped, $out exists"; return 0; fi
  if [ "$DRY" = 1 ]; then echo "      would run: $* > $log"; return 0; fi
  "$@" > "$log" 2>&1
}
model_dir() { if [ "$1" -eq 1 ]; then echo $R/s1; else echo $R/$METHOD/s$1; fi; }
[ -f $R/s1/lm.pt ] || { echo "no S1 model at $R/s1/lm.pt: run cl/run_vanilla.sh first"; exit 1; }
[ -f $E/noise.json ] || { echo "no dance-metric references at $E: run cl/run_eval_dance.sh first"; exit 1; }
echo "[$METHOD] selection ${SELECT:-none}, training ${TRAIN:-ground-truth replay}, balance $BALANCE ($(date))"

for k in 2 3; do
  j=$((k - 1))
  echo "[$METHOD] S$k ($(date))"
  REPLAY=""
  if [ -n "$SELECT" ]; then
    for i in $(seq 1 $j); do
      if [ "$i" -eq 1 ] && [ ! -f $B/s1.pt ]; then  # same selection, same shared S1 model: the same buffer
        for src in $SELECT ${SELECT}_kd; do
          [ "$src" != "$METHOD" ] && [ -f buffers/fd/$src/s1.pt ] || continue
          if [ "$DRY" = 1 ]; then echo "      would copy buffers/fd/$src/s1.pt"; break; fi
          cp buffers/fd/$src/s1.pt $B/s1.pt
          cp buffers/fd/$src/s1.json $B/s1.json 2>/dev/null || true
          echo "      buffer S1 copied from buffers/fd/$src/s1.pt"
          break
        done
      fi
      BUF_ARGS="${BUFFER_ARGS:-}"
      [ "$SELECT" != random ] && BUF_ARGS="$BUF_ARGS --lm_state $(model_dir $i)/lm.pt"
      [ "$SELECT" = mir ] && BUF_ARGS="$BUF_ARGS --train_cache $C/s${i}_train_w5_h2.5.pt"
      [ "$SELECT" = csrel ] && [ "$i" -gt 1 ] && BUF_ARGS="$BUF_ARGS --prev_lm_state $(model_dir $((i - 1)))/lm.pt"
      run $B/s$i.pt logs/buffer_${METHOD}_s$i.log python cl/make_buffer.py --method $SELECT --session $i \
        --manifest $M/${NAME[$i]}_train.jsonl $BUF_ARGS --ratio $RATIO --out $B/s$i.pt
      grep -hE "^\[|^session|^saved" logs/buffer_${METHOD}_s$i.log 2>/dev/null || true
      REPLAY="$REPLAY --replay $B/s$i.pt"
    done
    REPLAY="$REPLAY --replay_balance $BALANCE $TRAIN"
  fi
  EVALS=""
  for i in $(seq 1 $k); do EVALS="$EVALS --eval s${i}_val=$C/s${i}_val_w5_h5.pt"; done
  run $(model_dir $k)/lm.pt logs/fd_${METHOD}_s$k.log \
    python cl/train_session.py --run_dir $(model_dir $k) --init_lm_state $(model_dir $j)/lm.pt \
    --train_manifest $M/${NAME[$k]}_train.jsonl --train_cache $C/s${k}_train_w5_h2.5.pt $EVALS \
    --schedule_from $R/s1 --eval_every 25 $REPLAY
  grep -hE "^train windows|^replay buffer|^each step|^distillation|^STAR" logs/fd_${METHOD}_s$k.log 2>/dev/null || true
  run $N/fd_${METHOD}_after_s$k.json logs/fd_eval_${METHOD}_s$k.log \
    python cl/eval_nll.py --lm_state $(model_dir $k)/lm.pt $TESTS --bf16 --out $N/fd_${METHOD}_after_s$k.json
  run $E/${METHOD}_after_s$k.json logs/dance_${METHOD}_s$k.log \
    python cl/eval_dance.py eval --encoder $ENCODER --ref_dir $E --samples $SAMPLES --lm_state $(model_dir $k)/lm.pt \
    --out $E/${METHOD}_after_s$k.json
  grep -h "_test " logs/dance_${METHOD}_s$k.log 2>/dev/null || true
done

if [ "$REPORT" = 1 ] && [ "$DRY" != 1 ]; then bash "$(dirname "$0")/report_replay.sh"; fi
echo "[$METHOD] done ($(date))"
