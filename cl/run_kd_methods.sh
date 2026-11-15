#!/usr/bin/env bash
# The distillation rows only, on UniMuMo / FineDance (S1 Street -> S2 Folk -> S3 Classic), after cl/run_vanilla.sh and
# cl/run_eval_dance.sh:
#   TMR+KD = tmr_joint_neff_kd, GCR+KD = gcr_kd, CSReL+KD = csrel_kd, MIR+KD = mir_kd
# Each: buffer of S1 (the shared S1 model) -> train S2 -> test -> buffer of S2 (its own S2 model) -> train S3 -> test,
# every buffer replayed by distillation (teacher = the model the session starts from; replay term
# 1 x (0.25 x ground truth + 1.0 x KL), equal weights current : S1 : S2). Same chain as every method
# (cl/run_replay.sh through cl/run_all_methods.sh): a method whose S3 is already scored is skipped, a half-finished one
# resumes at its first missing step, one running in another process is left to it. Ends with the comparison of these
# rows next to vanilla, ER+KD and their ground-truth-replay versions, where those have finished.
#
#   bash cl/run_kd_methods.sh > logs/run_kd.log 2>&1 &                  one after another
#   GPUS="0 1 2" bash cl/run_kd_methods.sh > logs/run_kd.log 2>&1 &     one queue per GPU
#
# Testing only (score what is trained but not yet scored, then the comparison): bash cl/test_kd_methods.sh
# env: METHODS, REPORT_ORDER, GPUS and the cl/run_all_methods.sh / cl/run_replay.sh options (BUFFER_ARGS, ...)
export METHODS=${METHODS:-"tmr_joint_neff_kd gcr_kd csrel_kd mir_kd"}
export REPORT_ORDER=${REPORT_ORDER:-"vanilla random random_kd gcr gcr_kd mir mir_kd csrel csrel_kd tmr_joint_neff tmr_joint_neff_kd"}
exec bash "$(dirname "$0")/run_all_methods.sh"
