"""Held-out loss (teacher-forced cross-entropy) of UniMuMo on paired music-motion sets.

Forgetting on set j after session k is  CE_j(theta_k) - CE_j(theta_{k-1})  (Eq. 6 of the TMR paper).

Step 1 (once per set): tokenize music + motion windows and cache them. Tokenizers are frozen,
so the cache is valid for every checkpoint.
Step 2 (per checkpoint): forward passes only.

Examples (run from the repo root):
    # AIST++ val probe: 69 sequences, never trained on, with their real music
    python cl/eval_nll.py --aist_split data/motion/aist_val.txt \
        --aist_motions data/motion/val/joint_vecs --aist_wavs ../EDGE/edge_aistpp/wavs \
        --out outputs/nll/s0.json

    # any other set: JSONL manifest, one {"id": ..., "music": wav, "motion": npy (T,263) @ 60 fps} per line
    python cl/eval_nll.py --manifest finedance_s1_test=manifests/s1_test.jsonl --out ...

    # a fine-tuned LM: pass its state_dict (or a full.ckpt-style file)
    python cl/eval_nll.py ... --lm_state runs/vanilla/s1/lm.pt --out outputs/nll/vanilla_s1.json
"""
import argparse
import json
import os
import re
import sys
import time
from os.path import join as pjoin
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

MUSIC_SR = 32000
MOTION_FPS = 60
TOKEN_RATE = 50


def read_manifest(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def aist_music_id(seq):
    """gBR_sBM_cAll_d04_mBR0_ch01 -> mBR0 (AIST++ has 60 tracks, 6 per genre)."""
    m = re.search(r'_(m[A-Z]{2}\d)_', seq)
    return m.group(1) if m else None


def parse_music_filter(spec):
    """'heldout0' -> the 10 tracks m??0 (one per genre); otherwise a comma list like 'mBR0,mPO1'."""
    if not spec:
        return None
    if spec == 'heldout0':
        return {f'm{g}0' for g in ['BR', 'PO', 'LO', 'MH', 'LH', 'HO', 'WA', 'KR', 'JS', 'JB']}
    return set(spec.split(','))


def aist_manifest(split_file, motion_dir, wav_dir, include_music=None, exclude_music=None):
    items, no_motion, no_wav = [], [], []
    include, exclude = parse_music_filter(include_music), parse_music_filter(exclude_music)
    n_filtered = 0
    with open(split_file) as f:
        for seq in (l.strip() for l in f):
            if not seq:
                continue
            mid = aist_music_id(seq)
            if (include is not None and mid not in include) or (exclude is not None and mid in exclude):
                n_filtered += 1
                continue
            m, w = pjoin(motion_dir, seq + '.npy'), pjoin(wav_dir, seq + '.wav')
            if not os.path.exists(m):
                no_motion.append(m)
            if not os.path.exists(w):
                no_wav.append(w)
            if os.path.exists(m) and os.path.exists(w):
                items.append({'id': seq, 'music': w, 'motion': m})
    print(f'AIST++ manifest from {split_file}: {len(items)} pairs; '
          f'{len(no_motion)} missing motion, {len(no_wav)} missing wav, {n_filtered} removed by music filter')
    for name, miss in [('motion', no_motion), ('wav', no_wav)]:
        if miss:
            print(f'  e.g. missing {name}: {miss[:3]}')
    if not items:
        sys.exit('No pairs found: fix --aist_motions / --aist_wavs (see the missing paths above).')
    return items


@torch.no_grad()
def tokenize_set(model, items, window, hop, batch_size):
    """Cut each pair into aligned windows and return music/motion codes [N, 4, window*50]."""
    import librosa
    n_tok = int(window * TOKEN_RATE)
    music_w, motion_w, ids = [], [], []
    for it in items:
        wav, _ = librosa.load(it['music'], sr=MUSIC_SR)
        mot = np.load(it['motion']).astype(np.float32)
        dur = min(len(wav) / MUSIC_SR, len(mot) / MOTION_FPS)
        start = 0.0
        while start + window <= dur + 1e-6:
            a, b = int(round(start * MUSIC_SR)), int(round(start * MOTION_FPS))
            music_w.append(wav[a:a + int(window * MUSIC_SR)])
            motion_w.append(mot[b:b + int(window * MOTION_FPS)])
            ids.append(f"{it['id']}@{start:.1f}")
            start += hop
    print(f'  {len(items)} pairs -> {len(ids)} windows of {window}s (hop {hop}s)')
    if not ids:
        sys.exit(f'No {window}s windows: every pair is shorter than the window (try --window 2.5).')

    music_codes, motion_codes = [], []
    for i in range(0, len(ids), batch_size):
        mu = np.stack(music_w[i:i + batch_size])[:, None, :]          # [B, 1, L]
        mo = np.stack(motion_w[i:i + batch_size])                      # [B, T, 263]
        c_mu = model.encode_music(mu).cpu()
        c_mo = model.encode_motion(mo).cpu()
        t = min(c_mu.shape[-1], c_mo.shape[-1], n_tok)
        music_codes.append(c_mu[..., :t])
        motion_codes.append(c_mo[..., :t])
    return {'ids': ids, 'music_code': torch.cat(music_codes), 'motion_code': torch.cat(motion_codes),
            'window': window, 'hop': hop}


def shuffle_music(cache, seed=0):
    """Control: pair every motion window with music from a different window (no fixed points)."""
    g = torch.Generator().manual_seed(seed)
    n = cache['music_code'].shape[0]
    perm = (torch.arange(n) + torch.randint(1, n, (1,), generator=g)) % n
    return dict(cache, music_code=cache['music_code'][perm])


def load_lm_state(model, path):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'music_motion_lm_weight' in sd:
        sd = sd['music_motion_lm_weight']
    elif isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.music_motion_lm.load_state_dict(sd)
    print(f'loaded LM weights from {path}')


@torch.no_grad()
def eval_set(mm, cache, batch_size, use_bf16):
    music_all, motion_all = cache['music_code'], cache['motion_code']
    N = music_all.shape[0]
    per_win_motion, per_win_music, per_q_motion = [], [], []
    for i in range(0, N, batch_size):
        mu = music_all[i:i + batch_size].to(mm.device)
        mo = motion_all[i:i + batch_size].to(mm.device)
        B = mu.shape[0]
        cond = mm.prepare_text_condition(['<separation>'] * B)        # null text: audio-only
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            mu_out, mo_out = mm.model.compute_predictions(mu, mo, 'music_motion', [], condition_tensors=cond)
        # per-window CE so results can be re-aggregated later
        for b in range(B):
            ce_mo, ce_mo_q = mm.compute_cross_entropy(mo_out.logits[b:b + 1].float(), mo[b:b + 1], mo_out.mask[b:b + 1])
            ce_mu, _ = mm.compute_cross_entropy(mu_out.logits[b:b + 1].float(), mu[b:b + 1], mu_out.mask[b:b + 1])
            per_win_motion.append(ce_mo.item())
            per_win_music.append(ce_mu.item())
            per_q_motion.append([q.item() for q in ce_mo_q])
    m = np.array(per_win_motion)
    return {
        'n_windows': int(N),
        'motion_ce': float(m.mean()),
        'motion_ce_sem': float(m.std(ddof=1) / np.sqrt(N)) if N > 1 else 0.0,
        'music_ce': float(np.mean(per_win_music)),
        'motion_ce_per_codebook': np.mean(per_q_motion, axis=0).tolist(),
    }, {'ids': cache['ids'], 'motion_ce': m, 'music_ce': np.array(per_win_music)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='pretrained/full.ckpt')
    ap.add_argument('--lm_state', default=None, help='fine-tuned music_motion_lm state_dict to evaluate')
    ap.add_argument('--manifest', action='append', default=[], help='name=path.jsonl (repeatable)')
    ap.add_argument('--aist_split', default=None)
    ap.add_argument('--aist_motions', default='data/motion/val/joint_vecs')
    ap.add_argument('--aist_wavs', default='../EDGE/edge_aistpp/wavs')
    ap.add_argument('--aist_name', default='aist_val')
    ap.add_argument('--aist_music_include', default=None, help="'heldout0' or comma list of track ids")
    ap.add_argument('--aist_music_exclude', default=None, help="'heldout0' or comma list of track ids")
    ap.add_argument('--window', type=float, default=5.0)
    ap.add_argument('--hop', type=float, default=5.0)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--cache_dir', default='outputs/token_cache')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--shuffle_music', action='store_true',
                    help='control: mismatched music; motion CE should rise if the model uses the audio')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sets = {}
    if args.aist_split:
        sets[args.aist_name] = lambda: aist_manifest(args.aist_split, args.aist_motions, args.aist_wavs,
                                                     args.aist_music_include, args.aist_music_exclude)
    for spec in args.manifest:
        name, path = spec.split('=', 1)
        sets[name] = (lambda p=path: read_manifest(p))
    assert sets, 'give --aist_split and/or --manifest'

    from unimumo.models import UniMuMo
    model = UniMuMo.from_checkpoint(args.ckpt)
    if args.lm_state:
        load_lm_state(model, args.lm_state)
    mm = model.music_motion_lm
    mm.eval()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    results = {}
    for name, get_items in sets.items():
        cache_path = pjoin(args.cache_dir, f'{name}_w{args.window:g}_h{args.hop:g}.pt')
        if os.path.exists(cache_path):
            cache = torch.load(cache_path)
            print(f'[{name}] token cache: {cache_path} ({len(cache["ids"])} windows)')
        else:
            print(f'[{name}] tokenizing...')
            cache = tokenize_set(model, get_items(), args.window, args.hop, args.batch_size)
            torch.save(cache, cache_path)
        if args.shuffle_music:
            cache = shuffle_music(cache)
            print(f'[{name}] CONTROL: music shuffled across windows')
        t0 = time.time()
        summary, per_window = eval_set(mm, cache, args.batch_size, args.bf16)
        summary['seconds'] = round(time.time() - t0, 1)
        results[name] = summary
        np.savez(args.out.replace('.json', f'_{name}_per_window.npz'), **per_window)
        print(f'[{name}] motion CE {summary["motion_ce"]:.4f} ± {summary["motion_ce_sem"]:.4f} | '
              f'music CE {summary["music_ce"]:.4f} | {summary["n_windows"]} windows | {summary["seconds"]}s')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    meta = {'ckpt': args.ckpt, 'lm_state': args.lm_state, 'window': args.window, 'hop': args.hop,
            'text': 'null', 'bf16': args.bf16, 'shuffle_music': args.shuffle_music}
    with open(args.out, 'w') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2)
    print(f'wrote {args.out}')
