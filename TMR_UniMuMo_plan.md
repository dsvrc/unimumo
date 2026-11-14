# Adding UniMuMo to the TMR benchmark: data, sessions, protocol

> **Protocol as decided (supersedes the Session-0 parts below):**
> - **Session 0** is the pretrained UniMuMo as downloaded. No training, and it is not an evaluation target.
> - **Three fine-tuning sessions, aligned with the DiscoForcing `finedance_style4` benchmark** (Mix dropped, as that plan recommends):
>   - **Order: S1 = Street → S2 = Folk → S3 = Classic.** This is DiscoForcing's escalating-drift order; distance from S1 rises 1.10 → 1.64.
>   - **Splits use DiscoForcing's algorithm** (`cl/finedance_sessions.py`): test = official cross_genre songs per family; val = round(10%) of train from sub-genres with ≥ 3 songs, `random.Random(0)`; `--disco_dir` imports its split files directly.
>   - After S2 we measure forgetting of S1; after S3, forgetting of S1 and S2. F_avg and A_avg use the paper's formulas.
> - **Metrics:**
>   - **StyleRetention** (DiscoForcing's validated style encoder on *generated* dance, `cl/eval_style.py`) is the headline forgetting number.
>   - **Held-out motion CE** (`cl/eval_nll.py`, the Eq. 6 quantity) is the cheap per-step diagnostic.
>   - Ground-truth and tokenizer-reconstruction rows are the reference levels.
> - Forgetting is measured only on the fine-tuned sessions, as in the GestureLSM, TalkSHOW and MOSPA tables.
> - Replay buffers are built only from data of fine-tuned sessions. There is **no D0 replay pool**, so Music4All is not needed.
> - The tests of pretrained skills (AIST++, MusicCaps, captioning) are at most an appendix extra.
> - The rest of §§4.4–4.6 and §8 that concerns Session 0 is optional.

**The short answer.** Start from the released UniMuMo checkpoint and treat it as Session 0. Build Sessions 1–4 from FineDance, grouped by coarse dance style. UniMuMo has never seen FineDance. Test each session on its held-out songs. Also measure the capabilities UniMuMo had before you touched it: AIST++ music→dance, text→music and captioning. Do not use AIST++ genres as sessions, because UniMuMo already trained on those motions. Give Session 0 a small replay pool built with UniMuMo's own pretraining recipe, so that replay methods can protect what was pretrained.

---

## 0. Decisions at a glance

| Question | Recommendation |
|---|---|
| Continual axis | Behavioural mode: dance-genre families with **real** paired music and dance |
| New-session data | FineDance public subset (7.7 h, 30 fps, SMPL-H), not in UniMuMo's pretraining |
| Session 0 | Released `full.ckpt`, no retraining |
| Sessions 1–4 | FineDance coarse-style groups, ordered from closest to pretraining to furthest: Street → Chinese classical → Folk/ethnic → Modern/other |
| Test data per session | FineDance@Genre test songs of that group, 5 s slices |
| Probes of pretrained capability | Optional appendix only (AIST++, MusicCaps, captioning) |
| Session-0 replay data | None. Session 0 is just the starting weights; buffers come from fine-tuned sessions |
| Conditioning in sessions | Audio only (null text), so the model gets no task identity; genre text is an ablation |
| What is trained | The LM only. Encodec, the motion VQ-VAE, T5 and the captioner stay frozen |
| TMR feature φ | Music-stream hidden state at LM layer 12 of 24, motion stream masked, pooled to 25 Hz (T=250, D=1024) |
| Primary metric | FID_k as m_{i,t}, giving F_avg and A_avg exactly as in the paper; FID_g, Div_k and BAS as secondary metrics |

---

## 1. What UniMuMo is, in continual-learning terms

- **Three parts in `full.ckpt`** ([unimumo/models/unimumo.py](unimumo/models/unimumo.py)):
  - an Encodec 32 kHz **mono** music tokenizer (4 codebooks × 2048 entries, 50 Hz), frozen;
  - a motion VQ-VAE that maps 263-d HumanML3D-format features at **60 fps** into the *same* Encodec codebook at 50 Hz ([configs/train_motion_vqvae.yaml](configs/train_motion_vqvae.yaml)). Its encoder and decoder also take the music embedding ([unimumo/models/motion_vqvae.py:111](unimumo/models/motion_vqvae.py:111));
  - the music–motion LM `MusicMotionTransformer`. It uses the MusicGen-small backbone (24 layers, d=1024) plus motion embeddings, a motion FFN per layer and motion heads. Measured with `cl/check_setup.py`: 848.9M LM parameters in total, of which 637.8M are trained in a session (≈420M backbone + 218.2M motion-specific). The rest are the caption-stage self-attention (100.7M) and the T5 conditioner (110.4M); the captioner decoder adds 223.7M. Text enters through cross-attention to a frozen T5-base encoder, with separate music and motion prompts split by `<separation>`.
  - a captioner: a T5-base decoder that reads LM features through an extra `captioning_self_attn` in every layer ([transformer.py:554](unimumo/audio/audiocraft_/modules/transformer.py:554)).
- **One training task.** The LM does joint next-token prediction of both token streams over 10 s windows, with loss = 0.85·CE_music + 0.15·CE_motion ([transformer_model.py:229](unimumo/models/transformer_model.py:229), [configs/train_music_motion.yaml](configs/train_music_motion.yaml)). Attention is causal both within and across the two streams ([mm_lm.py:331](unimumo/audio/audiocraft_/models/mm_lm.py:331)). Music→dance at test time therefore just teacher-forces the music stream.
- Pretraining used 48 GPUs (6 × 8 nodes). You never repeat it.

**What this means for the experiments**
1. UniMuMo has no separate audio encoder: the LM's music stream *is* the encoder. φ is therefore defined at an LM layer (§5).
2. Captioning shares the LM weights you fine-tune, but sessions never train it. Any drop in captioning quality is collateral forgetting, so it is a free probe.
3. Both tokenizers stay frozen. Forgetting therefore happens only in the LM, and all sessions share one token space.

## 2. What UniMuMo has already seen

Facts below are from the training code: [music_motion_text_dataset.py](unimumo/data/music_motion_text_dataset.py) and [get_aligned_motion_code.py](preprocessing/get_aligned_motion_code.py).

| Data | Role in UniMuMo pretraining | Use as a new session? |
|---|---|---|
| Music4All (~109K clips) | All music the LM trained on, with captions from ChatGPT, MU-LLaMA and tag templates | No. It *is* Session 0 |
| HumanML3D (23,384 train IDs) | Text–motion data. Motions were beat-aligned with DTW onto Music4All tracks | No |
| AIST++ motions (867 train sequences, 60 fps) | DTW-warped onto Music4All tracks, oversampled 30×, with genre text | **No.** Use it only as a probe of Session 0 |
| AIST++ music | Never used to train the LM | Test-only probe |
| DanceDB (126 train) | Emotion and folk dances, oversampled 30× | No |
| **FineDance** | Unseen | **Yes (main)** |
| PopDanceSet | Unseen | Optional extra session |
| BEAT2 / TalkSHOW / SAM | Unseen | Not suitable (see §3) |

Each Music4All track was paired with 5 motions (`num_pair_each_music = 5`, `repeat_time = 30`). **The LM has never seen a genuinely paired music–dance clip.** Every dance it saw had been warped onto unrelated pop tracks. Real paired music–dance sessions are therefore a new audio–motion relationship, not only new motions. That is the right setting for TMR.

Note: this copy of the repo contains only stub metadata. `data/music/music4all_train.txt` has 1 line and the caption JSONs are placeholders. Rebuilding pretraining pairs needs the real Music4All metadata.

## 3. Why FineDance genre families, and why not the alternatives

**Why FineDance:**
- Music→dance is UniMuMo's native task, so the model can actually learn each session (plasticity), and forgetting is measured on the same metric.
- FineDance is unseen and has real pairs. Its music also shifts: Chinese classical and folk music differ from Music4All's mostly Western catalogue. That exercises TMR's audio-trajectory covariance, not just motion novelty.
- It adds a fourth generator family on the behavioural-mode axis, next to MOSPA's genre sessions: a pretrained discrete-token LM, alongside flow matching, VQ-VAE+AR and diffusion.

**Why not the alternatives:**
- **AIST++ genres as sessions:** UniMuMo trained on these motions, oversampled 30×. Sessions would not be new, and plasticity and forgetting numbers would be inflated.
- **BEAT2 or TalkSHOW:** speech is out of domain for a music codec, and UniMuMo is body-only (22 joints). It has no hands or face, which carry most of the gesture and talking-head signal. The FGD evaluators also expect SMPL-X rotations.
- **SAM (MOSPA):** Encodec here is mono, so the binaural cues that define the task are lost.
- **PopDanceSet** (HybrIK reconstructions from Bilibili videos): usable, but it mixes a capture-domain shift into the genre shift. Keep it optional (§12).

## 4. Protocol

### 4.1 Sessions
- **S0**: the released UniMuMo. Evaluate everything; train nothing.
- **S1–S4**: the FineDance public subset, grouped by the **coarse style label** in FineDance's label files. The paper defines 5 coarse styles over 22 genres; the public part covers about 16 genres. Count hours per group after you download it, and merge any group under about 1 h. Proposed order:
  - **S1 Street** (breaking, locking, popping, hip-hop, …). Closest to AIST++, so it measures plasticity where styles overlap.
  - **S2 Chinese classical** (e.g. Dunhuang, ShenYuan, …). The largest jump in both movement vocabulary and music.
  - **S3 Folk/ethnic** (e.g. Hmong, Uighur, Korean, …).
  - **S4 Modern/jazz/other.**
- Appendix experiments: the reversed order (does the order matter?), and an 8-session sequence of fine genres (§12).

### 4.2 Train and test data per session
- **Train:** FineDance@Genre train and val songs of the group. **Test:** FineDance@Genre test songs of the group. The commonly used test split has 20 songs, and a song never appears in two splits. If a group ends up with fewer than 3 test songs, move whole songs from train to test until at least 15% of the group's songs are in test, and publish the split files.
- **Windows:** training uses 10 s random crops, like `MusicMotionTextDataset`. Testing uses 5 s slices with a 2.5 s hop; only the music is given as input.
- **Text:** use the null prompt (`'<separation>'`) for both train and test. Audio-only conditioning means no task identity, matching TMR's claim. Training and testing with genre text ("The style of the dance is X.") becomes an ablation for the setting where a task ID is available.

### 4.3 Converting FineDance to UniMuMo's format (get this right)
1. SMPL-H parameters (translation + 52×3 axis-angle) → forward kinematics with the `smplx` body model → keep joints 0–21, the 22 SMPL body joints. Hands are dropped because UniMuMo is body-only. State this in the paper; FineDance's hand FID cannot be reported.
2. Resample **joint positions** from 30 to 60 fps (or rotations, with slerp). *Then* compute the 263-d features with the HumanML3D pipeline: same target skeleton, y-up, floor at 0, facing +Z. **Do not interpolate the 263-d features themselves.** Their velocity channels would stay on a 30-fps-per-frame scale, unlike the 60 fps AIST++ features the tokenizer knows ([get_aligned_motion_code.py:186](preprocessing/get_aligned_motion_code.py:186)).
3. Normalize with UniMuMo's own [data/motion/Mean.npy](data/motion/Mean.npy) and `Std.npy`. Never recompute them.
4. Tokens:
   - music → `UniMuMo.encode_music` ([unimumo.py:113](unimumo/models/unimumo.py:113)), plus beats from [preprocessing/extract_music_code_beat.py](preprocessing/extract_music_code_beat.py);
   - motion → `UniMuMo.encode_motion` ([unimumo.py:90](unimumo/models/unimumo.py:90)). This is the frozen VQ-VAE with a zero waveform, the same way the repo encodes motion.
5. Sanity check: decode the ground-truth tokens to joints and compare them with the ground truth (MPJPE and FID_k), per group. This is the tokenizer ceiling (gate G2).

### 4.4 Session-0 replay data (D0)
A foundation model's Session 0 must be something replay methods can replay. Without that, every replay method is blind to pretraining, and regularizers win Session-0 retention by construction.

- Build **D0 = 2,000 windows of 10 s** with UniMuMo's own recipe:
  1. Take Music4All tracks. Apply for access; you only need a few thousand.
  2. Extract Encodec codes and beats.
  3. Run `preprocessing/get_aligned_motion_code.py`, which pairs each track with HumanML3D, AIST++ and DanceDB train motions. The authors' preprocessed `motion_data.tar.gz` on Hugging Face provides these motions.
  4. Build the text with the same captions and templates the dataset class uses.
- **If you cannot get Music4All:** run the same pipeline on a public tagged music corpus (e.g. FMA) and call it a pretraining-recipe proxy. Do **not** use real AIST++ pairs for D0. The model never trained on them, so replaying them would also improve the AIST++ probe, and retention would be confounded with extra supervision.
- Apply the paper's 10% rule: each method selects |B0| = 200 from D0. ER picks at random, TMR as in §5, GCR, MIR and the others by their own criteria.
- Ablations:
  - |B0| ∈ {50, 100, 200, 400}: Session-0 retention against how much pretraining data is kept.
  - A no-D0 setting, as if pretraining data were inaccessible; GenRep is then the only Session-0 replay.

### 4.5 Training recipe per session
- **Model surgery.** Load `full.ckpt` and take `music_motion_lm`. The released LM must contain the caption-stage `captioning_self_attn` weights, because `UniMuMo.generate_text` runs through them. It is therefore built with `stage='train_caption'`, and its `training_step` and freezing code ([transformer_model.py:151](unimumo/models/transformer_model.py:151)) take the captioning path. Subclass it:
  - override `training_step` to use the music–motion branch;
  - train `model.*` except `captioning_self_attn`;
  - freeze `text_model`, the T5 encoder, Encodec and the VQ-VAE.
  The layers already pick the generation path when `get_feature=False`.
- **Objective:** UniMuMo's own loss, unchanged (0.85 / 0.15). This matches the paper's "default backbone configuration". If gate G4 shows weak plasticity, raise the motion weight to 0.5 **for all methods**.
- **Optimizer:**
  - AdamW with β=(0.9, 0.95) and gradient clipping at 1.0.
  - Learning rate: sweep {5e-6, 1e-5, 2e-5} **once**, with Vanilla on S1 validation, then freeze it (S1 used 1e-5). Linear warm-up (200 steps in S1, the same number of epochs in later sessions), then a constant rate.
  - Weight decay 0.0–0.01. Pretraining's 0.1 pulls weights towards zero, which is drift that has nothing to do with the data.
- **Steps and batches:** bf16 autocast, 16 windows per step (use gradient accumulation if needed). Session length is fixed in **epochs** (passes over the session's own training windows): run Vanilla on S1, evaluate the S1 **validation** songs every 100 steps, and take the step with the **lowest** validation motion CE. Every later session then gets the same number of epochs and the same warm-up in epochs (`train_session.py --schedule_from runs/fd/s1`). A fixed *step* count over-trains small sessions: Folk has 58% of Street's training windows, so 500 steps were 1.7× S1's epochs and Folk validation loss was already rising (see §7). Use the identical schedule, batch and steps per session for every method.
- **Replay (ported from DiscoForcing: `scripts/build_replay_buffer.py`, `utils/cl_replay.py`, `train_ldf.py`):**
  - Buffers start at S1 (the pretrained model gets none): S2 replays buffer(S1), S3 replays buffer(S1) + buffer(S2). Each buffer is built right after its session, with that method's model after the session (`cl/make_buffer.py`, chain `cl/run_replay.sh <vanilla|random|tmr|tmr_joint>`).
  - Candidates = the session's training clips (non-overlapping training-length windows tiling every song, plus an end-aligned one when the remainder is ≥ half a window); buffer = 10 % of them.
  - ER = uniform sample; TMR = music trajectories (the LM's music input embedding, DiscoForcing's `music_embedding` analogue) → temporal operator → PPCA-BIC modes → whitened atoms → simplex OMP (stops early if no clip helps); TMR-joint = the same on [music, motion-tokenizer latents], each block scaled to equal weight, BIC with N·D, k-center fill. Selection checked against DiscoForcing's own functions on the same trajectories: identical clips and weights (`cl/tmr_select.py` is copied unchanged).
  - Training: each step = current batch (16) + replay; loss = L_current + λ·mean_i(w_i·l_i) per replay batch, λ = 1, w_i = TMR weights (mean 1), 1 for ER; buffer clips drawn without replacement. Balance `equal` (default, as required): every earlier session gets its own batch of 16 and its own term, so current : S1 : S2 = 1 : 1 : 1 in S3 (S2: 1 : 1, identical to DiscoForcing). DiscoForcing's `session` (2 : 1 : 1) and `uniform` (union) are options.
  - First replay results (1 seed, ground-truth replay): TMR had the lowest StyleRetention forgetting (F_avg 0.265 vs ER 0.293, TMR-joint 0.327, vanilla 0.416) and kept Folk through S3, but every replay method ended with *worse* held-out Street loss than vanilla (≈3.5 vs 2.47; pretrained 3.32): the 10 % buffers are most likely memorised (each S1 buffer clip is replayed ~28× in S2 and ~40× more in S3).
  - `tmr_joint_neff` (DiscoForcing): tmr_joint with the BIC on the effective number of independent temporal vectors and pursuit fill; checked against DiscoForcing's function (identical clips and weights, pursuit fill included).
  - `<selection>_kd` (DiscoForcing `replay.distill`, e.g. `tmr_joint_neff_kd`, `cl/run_tmr_joint_neff_kd.sh`): teacher = the model the session starts from, frozen, same random draws as the student; each replay term = λ·(0.25·mean(w·l_gt) + 1.0·mean(w·kd)), with kd = KL(teacher ‖ student) of the next-token distributions (UniMuMo's counterpart of DiscoForcing's squared error to the teacher's prediction; same masks and 0.85/0.15 weighting; zero at the session start).
- **Seeds:** 3 seeds for Vanilla, Joint, TMR and the two strongest baselines; 1 seed for the rest, and state that.

### 4.6 Evaluation
**After every session t (DiscoForcing's CL evaluation, ported in `cl/eval_dance.py`; run `cl/run_eval_dance.sh`):**
- Every session's test songs are cut into non-overlapping 10 s clips; the model dances each clip from its music with null text (as in fine-tuning; guidance only acts through text, so cfg 1.0), top-k 250, temperature 1. Joints recovered per clip, taken at 30 fps; the real-dance references are the same clips recovered the same way.
- **Forgetting metric: StyleRetention** (mass on the session's family from the style encoder, `cl/train_style_encoder.py` = DiscoForcing's recipe on UniMuMo's skeleton, with reject class). Precedent: FineDance's own Genre Matching Score. Also StyleAcc, DanceRate, StyleFD.
- **Standard metrics (FineDance benchmark = LODGE):** FID_k, FID_g on 5 s windows (1 s stride) with one frozen normalisation over all sessions (DiscoForcing), Div_k, Div_g, BAS and Foot Skating Ratio per clip. The metric code is copied from DiscoForcing (LODGE-derived) and checked against it (kinetic 1e-7, geometric/FID/Div/FSR identical; BAS checked on the cluster via real-dance BAS ≈ LODGE's 0.212).
- **Decision (`cl/forgetting_report.py`):** per metric and old session, change vs the split-half noise of that metric on real dance (SNR ≥ 3 = forgetting) and as a share of "complete forgetting" (another session's real dance scored as this session's). The build step prints each metric's SNR for complete forgetting first, i.e. which metrics can show forgetting at all on these test sets. Held-out loss is a supporting row with a paired 95% CI.
- Reference rows: real dance, and the motion tokenizer's reconstruction (ceiling of any UniMuMo model).
- *Superseded:* FID_k as the primary metric (no signal for genre forgetting on FineDance in DiscoForcing's validation; UniMuMo's authors do not report FID).
- The AIST++ music→dance probe, with the authors' script [test_model/test_music2motion_aist.py](test_model/test_music2motion_aist.py). It uses EDGE-style sliced test music, genre text and guidance 3.0. Report FID_k, FID_g, Div_k, Div_g and BAS. **At S0, Div_k, Div_g and BAS must match the UniMuMo paper.** It reports 10.68 / 10.35 / 0.24 for music-conditioned generation, against 10.61 / 7.48 / 0.24 for real dance. The paper deliberately reports no FID_k/FID_g, because the authors consider them unreliable for dance, so for FID there is no published number to match.

**At S0 and after the last session only** (generating 10 s music is costly):
- Text→music on MusicCaps with [test_model/test_musiccaps.py](test_model/test_musiccaps.py): FAD (VGGish), KL (PaSST) and CLAP, on the same caption subset for every method.
- Music captioning with [test_music2text_mullama.py](test_model/test_music2text_mullama.py) on MusicQA's 552 test pairs (the paper's protocol): BLEU, METEOR, ROUGE-L and BERTScore. The paper reports 0.261 / 0.291 / 0.369 / 0.892.
- Motion captioning with [test_motion2text_pad.py](test_model/test_motion2text_pad.py) is optional. The authors' `motion_data.tar.gz` has no `test/` split, so HumanML3D test motions would have to be rebuilt from the official release.
- The paper's text→music numbers to match at S0 are FAD_VGG 5.93, KL 1.99 and CLAP 0.27, computed with the AudioLDM eval repo.
- Optional: motion→music with [test_motion2music_aist.py](test_model/test_motion2music_aist.py).

**Report:**
- the matrix m_{i,t};
- F_avg and A_avg exactly as in the paper's §4, over the FineDance sessions;
- separately, **F_S0** for each probe: the probe at the last session minus the probe at S0, with signs adjusted so that higher means more forgetting.

**Helper:** copy `full.ckpt` and replace `music_motion_lm_weight` with θ_t's state_dict. Every existing `test_model/` script then runs unchanged on every session checkpoint.

## 5. TMR on UniMuMo

**Defining φ_i** (at θ_{k−1}): the music-stream output of transformer layer ℓ, computed with
- the motion stream replaced by the special token, so φ depends on the audio only;
- the null text prompt, eval mode, and the training-time causal mask (`get_self_attn_mask(..., 'music_motion')`);
- the delay-pattern padding dropped, then 2× average pooling in time, giving φ_i ∈ R^{250×1024} for a 10 s window.

**Which layer:**
- The default is **ℓ = 12** (of 24), fixed in advance. Ablate ℓ ∈ {0, 6, 12, 18, 24}.
- At ℓ = 0 (the summed token embeddings), assumption (A1) holds exactly. For ℓ > 0 it holds only approximately, because in joint training the music stream also attends causally to the motion stream. Say so in the paper and let the layer ablation carry the argument.

**Why pool to 25 Hz:** Theorem 2 needs N ≳ T²/(r*·log T) samples. With T=500 and r*≈20, that is about 2K independent windows. With T=250 it is about 0.6K, which a 1–2 h session provides. Report the estimated r* for each session.

The positional embedding is the same for every sample, so centering (φ̃ = φ − φ̄) removes it; no special handling is needed. Everything else follows the paper: C_k (T×T), the rank from Eq. 12, whitening (Eq. 13), OMP by default with CoSaMP and SSP variants, and weights normalized to mean 1. TMR adds no parameters, no inference cost and no change to the backbone.

Sketch using a forward hook, with no edits to the backbone:

```python
@torch.no_grad()
def audio_trajectory(lm, music_codes, null_cond, layer=12, pool=2):
    # lm: MusicMotionTransformer.model (LMModel); music_codes: [B, 4, T], T=500 for 10 s
    B, K, T = music_codes.shape
    pat = lm.pattern_provider.get_pattern(T)
    mus, _, _ = pat.build_pattern_sequence(music_codes, lm.special_token_id, keep_only_valid_steps=True)
    mot = torch.full_like(mus, lm.special_token_id)                     # mask the motion stream
    seq = torch.cat([mus, mot], dim=-1)
    src_mask = lm.get_self_attn_mask(mus.shape[-1], mot.shape[-1], 'music_motion')
    cross_mask = torch.where(null_cond['description'][-1], 0., float('-inf'))
    out = {}
    hook = lm.transformer.layers[layer - 1].register_forward_hook(lambda m, i, o: out.update(h=o))
    lm(seq, [], src_mask=src_mask, condition_tensors=null_cond, cross_attn_mask=cross_mask)
    hook.remove()
    h = out['h'][:, :mus.shape[-1]]          # music half of the joint sequence
    h = h[:, K - 1:K - 1 + T]                # drop delay padding (check the offset in codebooks_patterns.py)
    return F.avg_pool1d(h.transpose(1, 2), pool).transpose(1, 2)   # [B, T/pool, 1024]
# null_cond = MusicMotionTransformer.prepare_text_condition(['<separation>'] * B) in eval mode
# layer=0: use sum(lm.emb[k](music_codes[:, k]) for k in range(K)) instead of the hook
```

## 6. Baselines

**Must run** (same budget, steps and schedule for all):
- **Vanilla FT.**
- **Joint:** fine-tune θ0 on D0 ∪ D1…D4 for the same total number of steps. This is the upper bound and the "lossless" reference.
- **Replay:** ER, MIR, GCR, CSReL and OCS, your strongest non-TMR methods in Tables 1–3. On a model with 0.64B trainable parameters, give the gradient-based selectors last-layer (head) gradients, and say so.
- **Regularization:** EWC, with a diagonal Fisher computed on D0 and past buffers.
- **Methods specific to foundation models:**
  - LoRA fine-tuning (rank 16 on attention and both FFNs, lr 1e-4);
  - SD-LoRA;
  - GenRep (joint music–motion samples drawn from θ_{k−1}, same budget).
- **TMR** with OMP (main), CoSaMP and SSP, plus **TMR + LoRA**.

**Appendix, 1 seed:** UCL, STAR, C-Flat, POCL, and WiSE-FT (interpolating weights between θ_k and θ0), a common baseline for adapting foundation models.

Why LoRA matters: reviewers will ask whether parameter-efficient tuning alone avoids forgetting in a pretrained model. It reduces forgetting, but it also reduces plasticity. Showing TMR+LoRA beating LoRA answers that.

## 7. Pilot gates (run before the full grid)

- **Decision (user):** G0 is **not a gate**. The protocol is S0 = checkpoint → S1 fine-tune → S2 fine-tune (forgetting of S1) → S3. d_audio is only *logged* during training, to help explain results. The one lesson kept from G0: fine-tuning can overfit, so **choose the S1 length as the step with the lowest S1 validation-song loss**, checked every 100 steps with Vanilla, and give every later session and method the same number of **epochs** (not steps).
- **Sessions built, first version (superseded by the DiscoForcing-aligned splits above; S1 is retrained under `runs/fd/`):**
  - S1 Street ≈ 2.65 h, S2 Classic ≈ 2.06 h, S3 Folk ≈ 1.49 h in total;
  - FineDance converted with SMPL-X neutral rest joints; the 6D convention is the pytorch3d "rows" one (knee check 1.000 against 0.000).
- **S1 result (`runs/s1`, lr 1e-5, batch 16, 5 s windows, default 0.85/0.15 loss):**
  - S1 validation motion CE fell from 3.30 (pretrained) to **2.32 at step 300**, then overfit (3.19 by step 2000).
  - **Fixed session length = 300 steps** for every session and method. Vanilla, ER and TMR share this S1 checkpoint.
  - d_audio stayed ≈ 0 throughout (−0.0002 ± 0.0002 at the best step).
- **Vanilla forgetting, DiscoForcing-aligned splits (`runs/fd`, `cl/run_vanilla.sh`, first run 2026-11-13). Forgetting is established in held-out CE:**
  - S0 test motion CE: S1 Street 3.32, S2 Folk 2.49, S3 Classic 2.60.
  - S1 Street: S1 val 3.16 → **2.174 best at step 500**, then overfits (2.30 at step 1000). S1 test 3.32 → 2.28.
  - S2 Folk, Vanilla, 500 steps from S1: S1 test 2.28 → **2.57 (F = +0.29; 28% of S1's gain lost)**. During S2, S1 val rose steadily from 2.164 (step 50) to 2.438 (step 500).
  - **But S2 was over-trained:** S2 val was best at **step 150 (2.030)** and climbed back to 2.137 by step 500. Train windows: S1 Street 3141, S2 Folk 1817, so 500 steps were 4.4 epochs on Folk against S1's 2.55 (1.7×). Fixed with the epoch-matched schedule (§4.5); the 500-step run is kept as `runs/fd/vanilla_500steps` as a "longer session → more forgetting" point.
  - **Epoch-matched S2 (the protocol; 289 steps = 2.54 epochs, warm-up 116 steps):** S1 test 2.28 → **2.46 (F = +0.18; 17% of S1's gain lost)**; S1 val rose at every evaluation after step 25 (2.159 → 2.336). S2 val best 2.036 at step 175 (1.5 epochs), 2.069 at the end. S2 test 1.70 → 1.57. A_avg 1.926 (vs 1.956 for the 500-step run).
  - **Most of what a session teaches, by CE, is generic FineDance, not style:** S1 (Street) alone cut the *untrained* Folk and Classic test CE from 2.49 / 2.60 to 1.70 / 1.65, the same ~1/3 as on Street itself (3.32 → 2.28). So "share of the gain lost" understates style forgetting, and StyleRetention (which only measures style) is the headline. Optional diagnostic: an S0 → Folk-only model scored on Street test gives the generic part, so F can be expressed as a share of the Street-specific gain.
  - S2 test after S2 was 1.57 while S2 val was 2.07. Folk val is only ~2 songs, and test may be closer to the training songs; check per song (the `*_per_window.npz` files) before relying on S2 test numbers.
  - d_audio ≤ 0.0024 at every evaluation: the model still ignores the music, so generated style = current motion prior.
  - **StyleRetention encoder:** only the AIST++ encoder exists on the cluster. DiscoForcing's `style_encoder train --dataset finedance` would need FineDance in its 272-d format plus the `finedance_style4` folder (4 classes including Mix), and its reject class cannot be trained for FineDance (`NEG_VAE` defines only the AIST++ VAE). So the same recipe is ported in `cl/train_style_encoder.py` and trained on UniMuMo's own FineDance joints (HumanML3D skeleton, the one UniMuMo generates on): classes Street / Folk / Classic + `other`; negatives = DiscoForcing's BABEL npz retargeted to the HumanML3D skeleton, UniMuMo-tokenizer garbage (random tokens decoded), static poses, time-shuffled dance. It writes a validation report (swap matrix clean / smoothed / mirrored; mass that non-dance probes put on any session). Smoke-tested end to end on synthetic data.
- **Style encoder (cluster, 2026-11-14):** test songs: right family 0.95-0.99, wrong <= 0.015; smoothing drop 0.001, mirror 0.072; non-dance probes <= 0.004 on any family (constant-token garbage 0.137). Passes every DiscoForcing criterion.
- **Full Vanilla evaluation, generated dance (`cl/run_eval_dance.sh`, 10 s clips of every test song; 65 / 39 / 31 clips):**
  - *Which metric can see forgetting* (real dance, split-half noise vs another session's real dance): StyleRetention SNR 1508 / 246 / 22, StyleFD 11421 / 597 / 59; **FID_k -0.8 / -2.6 / -4.2, FID_g 3.2 / 0.7 / -1.7** (FID cannot tell the sessions' real dance apart, as DiscoForcing found). Real-dance BAS 0.242 / 0.217 / 0.161 (LODGE GT 0.212: BAS port confirmed), FSR 0.035 / 0.017 / 0.012.
  - **Forgetting established.** Street music: StyleRetention 0.09 (S0) -> **0.73 after S1 -> 0.02 after S2** (change 0.71, SNR ~1000, 98% of complete forgetting); StyleFD 0.05 -> 1.34 (86%); held-out loss +0.18 (95% CI +0.16..+0.19). 500-step S2: StyleRetention also 0.02 (style forgetting saturates), loss +0.29.
  - Plasticity: Folk music 0.00 -> 0.36 after S2 (0.46 with 500 steps). The model ignores music: after S1 it dances Street to Folk and Classic music (mass 0.67 / 0.66); after S2, Street music gets Folk 0.42 + other 0.49.
  - **FID_k / FID_g say "no forgetting"** (they fall after S2: FID_k 2975 -> 1174 on Street) because they track motion artefacts, not style: after S1 the dance is jittery (Div_k 47 vs real 9.7), after S2 less so (29). Report them only as quality columns.
  - **Quality caveat (UniMuMo's motion tokenizer on FineDance):** the tokenizer's reconstruction of real test dance already scores StyleRetention 0.69 / 0.24 / 0.35 (DanceRate 0.69 / 0.27 / 0.55), FID_k 529-2092 vs a real floor of 50-63, FSR 0.16-0.32 vs real <= 0.035. Generated dance: FID_k 1.2k-3.3k, FSR 0.32-0.40. So absolute quality is low and Folk/Classic retention is capped well below 1; generation can exceed the reconstruction (Folk 0.36-0.46 vs 0.24), so it is a reference, not a strict ceiling. Pretrained S0 with null text produces mostly non-dance (DanceRate 0.10).
  - To state in the paper: the motion stream ignores the music (d_audio ~ 0), so a session's style lives in the motion prior and Vanilla forgetting is a complete style switch; a Joint (S1 + S2 together) row gives the attainable level for any CL method.
- **G0. Does the model use the audio? (diagnostic only)**
  - **Measure:** audio reliance Δ_audio = CE_motion(music shuffled across windows) − CE_motion(matched music), with `cl/eval_nll.py --shuffle_music`.
  - **S0 result on AIST++ val (152 windows): Δ_audio ≈ 0.0002 nats** (3.2926 against 3.2924). Teacher-forced, pretrained UniMuMo predicts dance from the dance history alone. A likely cause is that its pretraining pairs were random tracks with only the beats DTW-aligned.
  - **S0 held-out motion CE (bf16, 5 s windows):**
    - AIST++ val: 3.29 (152 windows);
    - the val sequences on held-out tracks m??0: 3.04 (52 windows);
    - the *train* sequences on those tracks: **1.88** (479 windows).

    UniMuMo has memorized the AIST++ train motions it saw in pretraining. This confirms that AIST++ cannot be a session, and that only unseen motions (FineDance) give a real learning signal. Judge G0 on `heldout_val`.
  - **Converter verified** (`cl/motion_convert.py selftest`): UniMuMo features → joints → features reproduces every channel exactly, on AIST++ at 60 fps and on HumanML3D at 20 fps. The foot-contact threshold is HumanML3D's default 0.002, even at 60 fps.
  - **Full pipeline verified** (`cl/motion_convert.py aist`): AIST++ SMPL (EDGE's SMPL offsets, y-up, no mirror, threshold 0.002) → features matches UniMuMo's AIST++ features exactly on 5/5 sequences. The 60→30→60 fps round trip costs relative RMSE ≤ 0.002 on positions, ≤ 0.012 on rotations and ~0.11 on velocities, with 98–99% contact agreement. That velocity loss is inherent to 30 fps data such as FineDance.
  - **G0 run 1 (`runs/g0_aist`): failed.** It trained on AIST++ train sequences without the held-out tracks, for 2000 steps at lr 1e-5 with the default 0.85/0.15 loss.
    - d_audio stayed ≈ 0 on every held-out set at every evaluation (final `heldout_val` −0.004 ± 0.003).
    - It overfit: training CE fell (motion ~0.9 → 0.56; music → 0.04, i.e. the 50 training songs were memorized) while held-out motion CE *rose* (`heldout_val` 3.04 → 4.09, `aist_val` 3.29 → 4.37).
    - Caveats: those training motions were already memorized in pretraining, and the loss was 85% music.
  - **G0 run 2 (fair retest):**
    - Train on the **AIST++ test split**: 469 sequences whose motions UniMuMo never trained on, converted with the verified pipeline, minus the held-out tracks.
    - Use a **motion-only loss** (`--motion_weight 1.0`) and evaluate every 100 steps on held-out tracks with unseen motions.
    - Pass = held-out motion CE drops below step 0 (real plasticity) **and** d_audio becomes clearly positive.
    - **If it fails too,** UniMuMo's motion stream does not use audio even when trained on real pairs. Then either report it as a boundary case, or switch to a pretrained model that verifiably uses audio.
  - **Test:** fine-tune briefly on real AIST++ train pairs, then re-measure Δ_audio on AIST++ val.
    - **If Δ_audio becomes clearly positive** (paired over windows, well above its SEM), real-pair sessions teach an audio–motion relationship. UniMuMo is then a valid TMR test bed. Report Δ_audio for every session and method: shrinkage of Δ_audio on old sessions is forgetting of the audio–motion relationship itself, which is a sharper and more TMR-specific number than CE.
    - **If it stays ≈ 0**, forgetting in UniMuMo would be about the motion prior, not the audio pathway. Either present UniMuMo as a boundary case where TMR's assumption fails, or drop it.

- **G1. Reproduce S0.** AIST++ music→dance with the authors' script must match the UniMuMo paper within noise: Div_k ≈ 10.68, Div_g ≈ 10.35 and BAS ≈ 0.24, using Bailando's diversity code and Librosa beats for BAS. Fix the metric pipeline before doing anything else. Record S0's FID_k and FID_g as well; they have no published reference but are the starting point for measuring forgetting.
- **G2. Tokenizer ceiling.** For each FineDance group, measure FID_k and MPJPE of decode(encode(ground truth)) against the ground truth. If a group's ceiling is far worse than AIST++'s, merge or drop that group. Never retrain the tokenizer partway through the sequence.
- **G3. Does forgetting happen?** Run Vanilla S1→S2 (`cl/run_vanilla.sh`). You need a clear rise in S1's held-out CE and a clear drop in S1's StyleRetention. CE: yes (+0.18 with the epoch-matched schedule, +0.29 at 500 S2 steps). StyleRetention: pending the encoder (`cl/train_style_encoder.py`). Pretrained models often forget less. If the rise is small, either:
  - use more contrasting sessions;
  - or run longer sessions.
  Decide this before running baselines.
- **G4. Plasticity.** Vanilla's A_avg should be close to Joint's. If not, raise the motion weight or the number of steps, for all methods.
- **G5. Noise floor.** Train Vanilla on S1 with 3 seeds to get the seed-to-seed spread of FID_k. Only claim differences larger than that.

## 8. What to add to the paper

1. **Main table:** "Continual genre adaptation of UniMuMo on FineDance (FID_k↓)", in the layout of Tables 1–3, with F_avg and A_avg.
2. **Table, retention of pretrained capabilities after S4:** AIST++ FID_k and BAS, MusicCaps FAD and CLAP, captioning CIDEr. One row per method; the S0 row is the reference.
3. **Figure, Session-0 retention against |B0|:** how many pretraining windows each method needs. This is the foundation-model version of your Fig. 2.
4. **Figure, φ-layer ablation** (F_avg and coverage against ℓ), plus the C_k spectrum and the selected M_k for each session. It tests whether the effective rank grows with sessions, as your §5 claims.
5. **Theory check (cheap and strong):** for every method's buffer, compute ε_B from Proposition 1 using TMR's φ. Plot it against measured forgetting across methods and seeds, and report Spearman ρ. This tests Proposition 1 directly on a pretrained model with 0.64B trainable parameters.
6. **Overhead table:** add the time and VRAM to build each buffer on UniMuMo.

**Text changes:**
- Add the fourth family to the abstract, §1 and §4 ("…diffusion-based motion generators, and a pretrained text–music–motion foundation model").
- Describe S0 as the released checkpoint.
- State the scale (an 849M-parameter LM on a MusicGen-small backbone, 638M trained per session) so "foundation model" is not overclaimed.

## 9. Implementation checklist (new `cl/` folder)

| File | Purpose |
|---|---|
| `cl/convert_finedance.py` | SMPL-H → 22 joints → 60 fps → 263-d features; split lists for each session |
| `cl/encode_tokens.py` | Music and motion tokens plus beats for any manifest (sessions, D0) |
| `cl/session_data.py` | Windowed session dataset, weighted replay dataset, batch mixing |
| `cl/continual_lm.py` | Subclass of `MusicMotionTransformer`: load from `full.ckpt`, freezing, loss with replay, EWC and LoRA hooks |
| `cl/trajectory.py` | φ extraction (§5) and streaming C_k |
| `cl/tmr_select.py` | Eqs. 12–14 with OMP, CoSaMP and SSP (port from your existing TMR code) |
| `cl/run_sequence.py` | S1→S4 loop: train → extract φ → select → save θ_k and the buffer |
| `cl/to_unimumo_ckpt.py` | Put θ_k into `full.ckpt` so every `test_model/` script works |
| `cl/eval_m2d.py` | Batched music→dance on slices, then FID_k, FID_g, Div_k and BAS |

**Reuse unchanged:** `test_model/test_music2motion_aist.py`, `test_musiccaps.py`, `test_music2text_mullama.py`, `test_motion2text_pad.py`, `preprocessing/get_aligned_motion_code.py` and `preprocessing/extract_music_code_beat.py`.

**Speed-up:** `LMModel.generate` recomputes the whole sequence at every step and has no KV cache ([mm_lm.py:503](unimumo/audio/audiocraft_/models/mm_lm.py:503)). Classifier-free guidance doubles the batch even with null text. Batching 64 slices per call helps; adding streaming KV caches would cut evaluation cost the most.

## 10. Compute (rough, single A100-40GB)

| Item | Estimate |
|---|---|
| Training | About 1 GPU-hour per session per method (~6K steps at batch 16, bf16). 4 sessions × ~14 methods ≈ 55 GPU-h per seed; extra seeds for 5 methods add ≈ 45 GPU-h |
| Evaluation | FineDance plus the AIST++ probe ≈ 0.5–0.75 GPU-h per checkpoint. The final MusicCaps and captioning probes ≈ 2 GPU-h per method. Total ≈ 60–70 GPU-h |
| **Total** | **≈ 150–200 A100-hours** |
| Minimum version | Vanilla, Joint, ER, EWC, LoRA, GCR, CSReL and TMR; 1 seed, 3 seeds for Vanilla and TMR: **≈ 60–90 A100-hours** |

## 11. Risks and how to handle them

| Risk | Mitigation |
|---|---|
| A pretrained model forgets too little | Gate G3. Use more contrasting order or longer sessions. If forgetting stays small, report it honestly; that is a finding too |
| The tokenizer cannot represent new genres | Gate G2. Add the ceiling as a row in the tables |
| Noisy FID_k on small test sets | Use 5 s slices with a hop, 2 samples per slice and 3 seeds; report std; add FID_g and BAS |
| UniMuMo's authors call FID_k/FID_g unreliable for dance | Use FID only as a relative measure (each method against itself at S0, same reference set). Report Div and BAS next to it, and say so in the paper before a reviewer does |
| The 0.85 music loss weight dominates | Gate G4 |
| Licences | Music4All on request; FineDance research licence (public part only); AIST++ CC BY 4.0; HumanML3D/AMASS non-commercial. Research use only; do not redistribute converted data |

## 12. Optional extensions

- **Identity axis in dance:** use FineDance@Dancer (dancer-disjoint) groups as sessions, if the label files include dancer IDs.
- **Capture-domain shift:** add PopDanceSet (263 videos, 19 genres, about 3.6 h, HybrIK SMPL) as S5.
- **Long horizon:** 8 sessions of FineDance genres taken in pairs, mirroring your 9-session GestureLSM test.
- **Cross-domain stress test:** BEAT2 speakers with UniMuMo, body only, with speech going through the music codec. Only if time remains; expect low plasticity.

## Sources
- FineDance repository (public 7.7 h subset, SMPL-H, @Genre / @Dancer splits): https://github.com/THUSIGSICLAB/FineDance
- FineDance paper (14.6 h, 346 songs, 22 genres in 5 coarse styles, 27 dancers, 52 joints): https://arxiv.org/abs/2212.03741
- MEGADance (public FineDance at 30 fps, 16 genres, 20 test songs): https://arxiv.org/html/2505.17543
- PopDanceSet / POPDG: https://arxiv.org/abs/2405.03178
- UniMuMo paper: https://arxiv.org/abs/2410.04534
