#!/usr/bin/env bash
# DiscoForcing's tmr_joint_neff_kd on UniMuMo's FineDance sessions, after the shared S1 model (cl/run_vanilla.sh) and
# the dance-metric references (cl/run_eval_dance.sh):
#   buffers  tmr_joint_neff: TMR on the joint music + dance trajectory, BIC with the effective number of independent
#            temporal vectors, pursuit fill; 10 % of each session's clips, built after S1 (shared S1 model) and after
#            this variant's own S2 model -> buffers/fd/tmr_joint_neff_kd/
#   replay   distillation: teacher = the model the session starts from (frozen), kd = KL(teacher || student) on the
#            replay clips, replay term = 1.0 x (0.25 x ground truth + 1.0 x kd), TMR weights inside; every earlier
#            session its own batch of 16 and its own term (current : S1 : S2 = 1 : 1 : 1)
#   scoring  held-out loss and dance metrics after S2 and S3, then the forgetting report with every finished method
# Same chain as the other methods (cl/run_replay.sh); finished steps are skipped on a rerun.
#
#   bash cl/run_tmr_joint_neff_kd.sh > logs/replay_tmr_joint_neff_kd.log 2>&1 &
exec bash "$(dirname "$0")/run_replay.sh" tmr_joint_neff_kd "$@"
