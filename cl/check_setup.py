"""Setup check for the TMR x UniMuMo experiments.

Run from the repo root:
    python cl/check_setup.py                  # data + checkpoint + a 5 s music->dance generation
    python cl/check_setup.py --skip_generation

Checks
  1. motion features listed in the split files exist (minus ignore_list), shapes are (T, 263)
  2. pretrained/ contains the unified and the separate weights
  3. full.ckpt loads; prints LM stage, parameter counts, whether caption-stage modules exist
  4. (optional) generates 5 s of dance from the sample wav and saves the joints
"""
import argparse
import os
import sys
import time
from os.path import join as pjoin
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch


def check_motion_data(motion_dir):
    print('\n=== 1. motion data ===')
    with open(pjoin(motion_dir, 'ignore_list.txt')) as f:
        ignore = {l.strip() for l in f if l.strip()}
    for split in ['train', 'val', 'test']:
        vec_dir = pjoin(motion_dir, split, 'joint_vecs')
        on_disk = set(s[:-4] for s in os.listdir(vec_dir)) if os.path.isdir(vec_dir) else set()
        listed_total, missing = 0, {}
        for name in ['humanml3d', 'aist', 'dancedb']:
            with open(pjoin(motion_dir, f'{name}_{split}.txt')) as f:
                ids = [l.strip() for l in f if l.strip()]
            listed_total += len(ids)
            miss = [i for i in ids if i not in on_disk and i not in ignore]
            missing[name] = miss
        print(f'[{split}] on disk: {len(on_disk)}, listed: {listed_total}, '
              + ', '.join(f'{k} missing (not ignored): {len(v)}' for k, v in missing.items()))
        for k, v in missing.items():
            if v:
                print(f'    e.g. {k}: {v[:5]}')

    # shapes: one sample per source from train
    vec_dir = pjoin(motion_dir, 'train', 'joint_vecs')
    for name in ['humanml3d', 'aist', 'dancedb']:
        with open(pjoin(motion_dir, f'{name}_train.txt')) as f:
            ids = [l.strip() for l in f if l.strip()]
        for i in ids:
            p = pjoin(vec_dir, i + '.npy')
            if os.path.exists(p):
                x = np.load(p)
                print(f'  {name:9s} {i}: shape {x.shape}, finite={np.isfinite(x).all()}')
                break
    mean, std = np.load(pjoin(motion_dir, 'Mean.npy')), np.load(pjoin(motion_dir, 'Std.npy'))
    print(f'  Mean/Std shapes: {mean.shape} {std.shape}, min std {std.min():.2e}')


def check_weights(pre_dir):
    print('\n=== 2. weights ===')
    for f in ['full.ckpt', 'music_vqvae.bin', 'motion_vqvae.ckpt', 'music_motino_lm.ckpt']:
        p = pjoin(pre_dir, f)
        print(f'  {f:24s} {"OK " + format(os.path.getsize(p) / 2**30, ".2f") + " GiB" if os.path.exists(p) else "MISSING"}')


def count(params):
    return sum(p.numel() for p in params) / 1e6


def check_checkpoint(ckpt, skip_generation, out_dir):
    print('\n=== 3. load full.ckpt ===')
    print(f'  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}',
          f'({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else '')
    from unimumo.models import UniMuMo
    t0 = time.time()
    model = UniMuMo.from_checkpoint(ckpt)
    print(f'  loaded in {time.time() - t0:.0f}s, motion fps = {model.motion_fps}')

    mm = model.music_motion_lm
    lm = mm.model
    print(f'  LM stage: {mm.stage}')
    caption_params = [p for n, p in lm.named_parameters() if 'captioning_self_attn' in n]
    t5_params = list(lm.condition_provider.parameters())
    motion_params = [p for n, p in lm.named_parameters() if 'motion' in n]
    print(f'  LM params total: {count(lm.parameters()):.1f}M')
    print(f'    caption-stage self-attn: {count(caption_params):.1f}M (present: {len(caption_params) > 0})')
    print(f'    text conditioner (T5 enc + proj): {count(t5_params):.1f}M')
    print(f'    motion-specific (emb/FFN/norm/heads): {count(motion_params):.1f}M')
    print(f'    trainable in a session (LM minus caption attn minus T5 encoder): '
          f'{count(lm.parameters()) - count(caption_params) - count(t5_params):.1f}M (+ T5 output_proj)')
    print(f'  captioner (text_model) params: {count(mm.text_model.parameters()):.1f}M')
    print(f'  transformer layers: {len(lm.transformer.layers)}, d_model: {lm.dim}, n_q: {lm.n_q}, card: {lm.card}')

    if skip_generation:
        return
    print('\n=== 4. 5 s music->dance generation ===')
    import librosa
    wav_path = 'data/music/audios/aajYt64GBaxOqJlt.wav'
    wav, _ = librosa.load(wav_path, sr=32000)
    wav = wav[None, None, :5 * 32000]
    t0 = time.time()
    out = model.generate_motion_from_music(
        waveform=wav, text_description=['<separation> The style of the dance is break.'],
        conditional_guidance_scale=3.0)
    joints = out['joint']
    print(f'  generated joints {joints.shape} in {time.time() - t0:.0f}s, finite={np.isfinite(joints).all()}')
    os.makedirs(out_dir, exist_ok=True)
    np.save(pjoin(out_dir, 'gen_joints.npy'), joints)
    print(f'  saved {pjoin(out_dir, "gen_joints.npy")}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='pretrained/full.ckpt')
    ap.add_argument('--motion_dir', default='data/motion')
    ap.add_argument('--pretrained_dir', default='pretrained')
    ap.add_argument('--out_dir', default='outputs/check_setup')
    ap.add_argument('--skip_generation', action='store_true')
    ap.add_argument('--skip_checkpoint', action='store_true')
    args = ap.parse_args()

    check_motion_data(args.motion_dir)
    check_weights(args.pretrained_dir)
    if not args.skip_checkpoint:
        check_checkpoint(args.ckpt, args.skip_generation, args.out_dir)
