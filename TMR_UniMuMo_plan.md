# Adding UniMuMo to the TMR benchmark: data, sessions, protocol

> **Protocol as decided (supersedes the Session-0 parts below):**
> - **Session 0** is the pretrained UniMuMo as downloaded. No training, and it is not an evaluation target.
> - **Three fine-tuning sessions** (decided):
>   - **S1 = Street:** close to what UniMuMo already knows.
>   - **S2 = Chinese classical:** the largest jump in both movement and music.
>   - **S3 = Folk/ethnic.**
>   - After S2 we measure forgetting of S1; after S3, forgetting of S1 and S2. F_avg and A_avg use the paper's formulas over the three sessions.
>   - FineDance's modern/jazz/other songs are left out of the sequence. They can serve as a held-out appendix set.
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
  - Learning rate: sweep {5e-6, 1e-5, 2e-5} **once**, with Vanilla on S1 validation, then freeze it. Use 500 warm-up steps, then a constant rate.
  - Weight decay 0.0–0.01. Pretraining's 0.1 pulls weights towards zero, which is drift that has nothing to do with the data.
- **Steps and batches:** bf16 autocast, 16 windows per step (use gradient accumulation if needed). Use a fixed number of steps per session: pick it where Vanilla's S1 validation motion CE plateaus, probably 4–8K. Use identical steps, batch and schedule for every method.
- **Replay:** every step draws a replay batch of the same size from B_{<k}, sampling each item with probability proportional to w_i. Loss = L_new + λ·L_replay with λ = 1 (the paper's Eq. 7).
- **Seeds:** 3 seeds for Vanilla, Joint, TMR and the two strongest baselines; 1 seed for the rest, and state that.

### 4.6 Evaluation
**After every session t:**
- Each seen FineDance session i ≤ t: music→dance on its 5 s test slices, 2 samples per slice, using the repo's sampling defaults (top-k 250, temperature 1.0).
  - Metrics: **FID_k** (primary, m_{i,t}), FID_g, Div_k and BAS, computed on 22 joints at 60 fps.
  - Run ground truth and generated motion through the **same** 263-d → joints path (`recover_from_ric`). The Bailando/EDGE feature code assumes the SMPL-24 layout, so either duplicate the wrists for the hand joints or adapt it, but do the same for both sides.
  - Fix the reference set per group (all of the group's ground-truth windows) and use it for every method.
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

- **G0. Does the model use the audio? (decisive for UniMuMo)**
  - **Measure:** audio reliance Δ_audio = CE_motion(music shuffled across windows) − CE_motion(matched music), with `cl/eval_nll.py --shuffle_music`.
  - **S0 result on AIST++ val (152 windows): Δ_audio ≈ 0.0002 nats** (3.2926 against 3.2924). Teacher-forced, pretrained UniMuMo predicts dance from the dance history alone. A likely cause is that its pretraining pairs were random tracks with only the beats DTW-aligned.
  - **Test:** fine-tune briefly on real AIST++ train pairs, then re-measure Δ_audio on AIST++ val.
    - **If Δ_audio becomes clearly positive** (paired over windows, well above its SEM), real-pair sessions teach an audio–motion relationship. UniMuMo is then a valid TMR test bed. Report Δ_audio for every session and method: shrinkage of Δ_audio on old sessions is forgetting of the audio–motion relationship itself, which is a sharper and more TMR-specific number than CE.
    - **If it stays ≈ 0**, forgetting in UniMuMo would be about the motion prior, not the audio pathway. Either present UniMuMo as a boundary case where TMR's assumption fails, or drop it.

- **G1. Reproduce S0.** AIST++ music→dance with the authors' script must match the UniMuMo paper within noise: Div_k ≈ 10.68, Div_g ≈ 10.35 and BAS ≈ 0.24, using Bailando's diversity code and Librosa beats for BAS. Fix the metric pipeline before doing anything else. Record S0's FID_k and FID_g as well; they have no published reference but are the starting point for measuring forgetting.
- **G2. Tokenizer ceiling.** For each FineDance group, measure FID_k and MPJPE of decode(encode(ground truth)) against the ground truth. If a group's ceiling is far worse than AIST++'s, merge or drop that group. Never retrain the tokenizer partway through the sequence.
- **G3. Does forgetting happen?** Run Vanilla S1→S2. You need a clear rise in S1's FID_k and in the AIST++ probe. Pretrained models often forget less. If the rise is small, either:
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
