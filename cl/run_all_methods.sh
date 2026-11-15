#!/usr/bin/env bash
# Every method of the paper's DiscoForcing table on UniMuMo / FineDance (S1 Street -> S2 Folk -> S3 Classic), after
# cl/run_vanilla.sh (the shared S1 model) and cl/run_eval_dance.sh (the dance-metric references): one
# cl/run_replay.sh chain per method (buffers, S2 and S3 training, held-out loss and dance metrics). A method whose S3
# is already scored is skipped; a half-finished one resumes at its first missing step; one that is running in another
# process is left to it.
#
#   bash cl/run_all_methods.sh > logs/run_all.log 2>&1 &                    methods one after another
#   GPUS="0 1 2 3" bash cl/run_all_methods.sh > logs/run_all.log 2>&1 &     one queue per GPU
#
# Paper name = method: Fine-tuning = vanilla, ER = random, STAR = random_star (ER + STAR), OCS = ocs, GCR = gcr,
# MIR = mir, CSReL = csrel, TMR = tmr_joint_neff, ER+KD = random_kd, GCR+KD = gcr_kd, MIR+KD = mir_kd,
# CSReL+KD = csrel_kd, TMR+KD = tmr_joint_neff_kd; also tmr and tmr_joint (TMR on music only, and the joint
# trajectory with the BIC's N x D).
# env: METHODS (default: all of the above, in this order), REPORT_ORDER (rows of the final comparison; default
#      METHODS), GPUS (CUDA device ids, as CUDA_VISIBLE_DEVICES takes them; unset = one queue on the GPU(s) this shell
#      already sees), STAGE (all | train | test, passed to cl/run_replay.sh) and cl/run_replay.sh's BALANCE, RATIO,
#      SAMPLES, ENCODER, BUFFER_ARGS.
# Logs: logs/replay_<method>.log per method (and the step logs it names). At the end: the comparison of every
# finished method; the full report is in outputs/dance_eval/report_replay.txt.
set -uo pipefail
METHODS=${METHODS:-"vanilla random random_star ocs gcr mir csrel tmr tmr_joint tmr_joint_neff random_kd gcr_kd mir_kd csrel_kd tmr_joint_neff_kd"}
REPORT_ORDER=${REPORT_ORDER:-$METHODS}
E=outputs/dance_eval
N=outputs/nll
HERE=$(dirname "$0")
mkdir -p logs
finished() { [ -f $E/${1}_after_s3.json ] && [ -f $N/fd_${1}_after_s3.json ]; }
selection() { case $1 in vanilla) echo - ;; random_star) echo random ;; *_kd) echo ${1%_kd} ;; *) echo $1 ;; esac; }

TODO=()
for m in $METHODS; do
  if finished $m; then echo "[all] $m: finished, skipped"; else TODO+=("$m"); fi
done
if [ ${#TODO[@]} -eq 0 ]; then
  echo "[all] every method has finished"
else
  if [ -n "${GPUS:-}" ]; then read -r -a GPU_LIST <<< "$GPUS"; else GPU_LIST=(""); fi
  NQ=${#GPU_LIST[@]}
  # queues: methods with the same buffer selection share a queue (the later ones copy its S1 buffer), the rest go to
  # the shortest queue
  QUEUE=(); SELS=(); COUNT=()
  for ((q = 0; q < NQ; q++)); do QUEUE[q]=""; SELS[q]=" "; COUNT[q]=0; done
  for m in "${TODO[@]}"; do
    s=$(selection $m)
    pick=-1
    for ((q = 0; q < NQ; q++)); do
      if [ "$s" != - ] && [[ "${SELS[q]}" == *" $s "* ]]; then pick=$q; break; fi
    done
    if [ $pick -lt 0 ]; then
      pick=0
      for ((q = 1; q < NQ; q++)); do [ ${COUNT[q]} -lt ${COUNT[pick]} ] && pick=$q; done
    fi
    QUEUE[pick]="${QUEUE[pick]} $m"; SELS[pick]="${SELS[pick]}$s "; COUNT[pick]=$((COUNT[pick] + 1))
  done
  for ((q = 0; q < NQ; q++)); do
    [ -n "${QUEUE[q]}" ] || continue
    gpu=${GPU_LIST[q]}
    echo "[all] queue ${gpu:-default GPU}:${QUEUE[q]}"
    (
      for m in ${QUEUE[q]}; do
        echo "[all] $m started${gpu:+ on GPU $gpu} ($(date)), log logs/replay_$m.log"
        if [ -n "$gpu" ]; then
          CUDA_VISIBLE_DEVICES=$gpu REPORT=0 bash "$HERE/run_replay.sh" $m > logs/replay_$m.log 2>&1
        else
          REPORT=0 bash "$HERE/run_replay.sh" $m > logs/replay_$m.log 2>&1
        fi
        status=$?
        case $status in
          0) echo "[all] $m finished ($(date))" ;;
          3) echo "[all] $m is running in another process: left to it" ;;
          *) echo "[all] $m FAILED with status $status ($(date)): tail logs/replay_$m.log" ;;
        esac
      done
    ) &
  done
  wait
fi
for m in $METHODS; do finished $m || echo "[all] $m has not finished S3"; done
COMPARE_ONLY=1 ORDER="$REPORT_ORDER" bash "$HERE/report_replay.sh"
