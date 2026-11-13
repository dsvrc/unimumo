"""Score UniMuMo's dance on every FineDance session's test music with StyleRetention and the standard dance metrics,
following DiscoForcing's continual-learning protocol (metrics/cl_metrics.py, scripts/eval_cl_session.py).

  build : references and noise floors from the real test dance (run once; needs the style encoder)
          -> <ref_dir>/refs.npz : one frozen normalisation over all sessions' reference windows + each session's
                                  reference windows (kinetic, geometric) and style embeddings
          -> <ref_dir>/noise.json: per session: real-dance values, split-half noise (half the test songs vs the
                                  other half, 20 draws) and 'complete forgetting' (another session's real dance
                                  scored as this session's music = the model dancing the wrong style)
  eval  : one model on every session's test music = one column m[., t] of the forgetting matrix
          (default: the pretrained checkpoint; --lm_state <session lm.pt>; --gt real dance; --recon the motion
          tokenizer's reconstruction of it, i.e. the ceiling of any UniMuMo model)

Protocol (identical for references and generated dance, so the numbers are comparable):
  * every test song is cut into non-overlapping 10 s clips (UniMuMo's generation length);
  * the model dances each clip from its music with null text, as in fine-tuning (guidance only acts through the
    text prompt, so it is 1.0 = plain sampling with the checkpoint's top-k 250, temperature 1);
  * joints are recovered from the 263-d features per clip and taken at 30 fps (references the same way, so root
    placement matches; DiscoForcing's geometric features mix the global root with root-relative joints);
  * StyleRetention / StyleAcc / DanceRate / StyleFD from the style encoder (cl/train_style_encoder.py);
    FID_k / FID_g / Div_k / Div_g on 5 s windows (1 s stride) with ONE frozen normalisation (DiscoForcing);
    BAS and Foot Skating Ratio per clip (LODGE's FineDance metrics).

    python cl/eval_dance.py build --encoder pretrained/style_encoder_finedance.pt
    python cl/eval_dance.py eval --gt --out outputs/dance_eval/gt.json
    python cl/eval_dance.py eval --lm_state runs/fd/s1/lm.pt --out outputs/dance_eval/after_s1.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from dance_metrics import ClipSet, set_metrics
from eval_style import StyleEncoder

SESSIONS = {'s1': 's1_street', 's2': 's2_folk', 's3': 's3_classic'}
SR = 32000
HIGHER = {'StyleRetention': True, 'StyleAcc': True, 'DanceRate': True, 'StyleFD': False, 'FID_k': False,
          'FID_g': False, 'BAS': True, 'FSR': False}


# ---------------------------------------------------------------- clips
def load_clips(manifest_dir, sess, clip_seconds, max_clips):
    """Non-overlapping clip_seconds clips of every test song of a session: music (32 kHz) and 263-d motion (60 fps)."""
    import librosa
    clips = []
    for line in open(os.path.join(manifest_dir, f'{SESSIONS[sess]}_test.jsonl')):
        if not line.strip():
            continue
        it = json.loads(line)
        wav, _ = librosa.load(it['music'], sr=SR)
        feat = np.load(it['motion']).astype(np.float32)
        n = int(min(len(wav) / SR, len(feat) / 60) // clip_seconds)
        for k in range(min(n, max_clips) if max_clips else n):
            a = k * clip_seconds
            clips.append({'id': f"{it['id']}@{a:g}", 'song': it['id'], 'family': it['family'],
                          'wav': wav[int(a * SR):int((a + clip_seconds) * SR)],
                          'feat': feat[int(a * 60):int((a + clip_seconds) * 60)]})
    return clips


def gt_joints(feat):
    from unimumo.motion.motion_process import recover_from_ric
    return recover_from_ric(torch.from_numpy(feat).float()[None], 22)[0].numpy()[::2].astype(np.float32)


@torch.no_grad()
def generate(model, clips, samples, batch_size, cfg, seed):
    """Music -> dance for every clip (null text); 30 fps global joints, sample-major order."""
    mm = model.music_motion_lm
    out = []
    for s in range(samples):
        for b in range(0, len(clips), batch_size):
            torch.manual_seed(seed + 1000 * s + b)
            batch = clips[b:b + batch_size]
            codes = model.encode_music(np.stack([c['wav'] for c in batch])[:, None, :])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                motion = mm.generate_single_modality(music_code=codes, motion_code=None,
                                                     text_description=['<separation>'] * len(batch),
                                                     conditional_guidance_scale=cfg)
            feats = model.motion_vqvae.decode_from_code(codes, motion).float()
            out += [j[::2].astype(np.float32) for j in model.motion_vec_to_joint(feats)]
            print(f'  generated {min(b + batch_size, len(clips))}/{len(clips)} clips (sample {s + 1}/{samples})', flush=True)
    return out


@torch.no_grad()
def reconstruct(model, clips):
    """The motion tokenizer's reconstruction of the real clips, decoded with their own music like generated dance."""
    out = []
    for c in clips:
        codes = model.encode_music(c['wav'][None, None, :])
        mcodes = model.encode_motion(c['feat'][None])
        t = min(codes.shape[-1], mcodes.shape[-1])
        feats = model.motion_vqvae.decode_from_code(codes[..., :t], mcodes[..., :t]).float()
        out.append(model.motion_vec_to_joint(feats)[0][::2].astype(np.float32))
    return out


def load_refs(ref_dir):
    R = np.load(os.path.join(ref_dir, 'refs.npz'))
    norm = {k: R[k] for k in ('kin_mean', 'kin_std', 'geo_keep', 'geo_mean', 'geo_std')}
    refs = {s: {'kin': R[f'{s}_kin'], 'geo': R[f'{s}_geo'], 'emb': R[f'{s}_emb']}
            for s in SESSIONS if f'{s}_kin' in R.files}
    return norm, refs, json.load(open(os.path.join(ref_dir, 'noise.json')))


# ---------------------------------------------------------------- build: references + noise floors from real dance
def cmd_build(a):
    enc = StyleEncoder(a.encoder)
    sessions = a.sessions.split(',')
    data = {s: load_clips(a.manifest_dir, s, a.clip_seconds, a.max_clips_per_song) for s in sessions}
    fam = {s: data[s][0]['family'] for s in sessions}
    for s in sessions:
        assert fam[s] in enc.classes, f'{s}: family {fam[s]} is not an encoder class {enc.classes}'
    t0 = time.time()
    sets = {s: ClipSet([gt_joints(c['feat']) for c in data[s]], [c['wav'] for c in data[s]], SR, enc) for s in sessions}
    print(f'features of {sum(len(v) for v in data.values())} real clips ({time.time() - t0:.0f}s)')

    pool_k = np.concatenate([sets[s].windows()[0] for s in sessions])
    pool_g = np.concatenate([sets[s].windows()[1] for s in sessions])
    keep = pool_g.std(0) > 1e-6
    norm = {'kin_mean': pool_k.mean(0), 'kin_std': pool_k.std(0) + 1e-10, 'geo_keep': keep,
            'geo_mean': pool_g[:, keep].mean(0), 'geo_std': pool_g[:, keep].std(0)}
    refs = {s: {'kin': sets[s].windows()[0], 'geo': sets[s].windows()[1], 'emb': sets[s].emb} for s in sessions}

    rng = np.random.default_rng(a.seed)
    noise = {'meta': {'clip_seconds': a.clip_seconds, 'max_clips_per_song': a.max_clips_per_song,
                      'encoder': a.encoder, 'halves': a.halves, 'geo_features_kept': int(keep.sum()),
                      'clips': {s: [c['id'] for c in data[s]] for s in sessions}}}
    for s in sessions:
        cs, songs = sets[s], np.array([c['song'] for c in data[s]])
        uniq = sorted(set(songs))
        gt = set_metrics(cs, fam[s], norm, encoder=enc)
        halves = []
        for _ in range(a.halves):  # split by song, so the clips of one song stay together
            if len(uniq) >= 2:
                pick = set(rng.permutation(uniq)[:len(uniq) // 2])
                A = [i for i in range(cs.n) if songs[i] in pick]
                B = [i for i in range(cs.n) if songs[i] not in pick]
            else:  # a single test song: split its clips instead (noise is then underestimated)
                perm = rng.permutation(cs.n)
                A, B = sorted(perm[:cs.n // 2]), sorted(perm[cs.n // 2:])
            kB, gB = cs.windows(B)
            halves.append(set_metrics(cs, fam[s], norm, ref={'kin': kB, 'geo': gB, 'emb': cs.style(B)[1]},
                                      encoder=enc, clips=A))
        others = {o: set_metrics(sets[o], fam[s], norm, ref=refs[s], encoder=enc) for o in sessions if o != s}
        n = {'n_songs': len(uniq), 'n_clips': cs.n, 'n_windows': int(len(refs[s]['kin']))}
        for k in ('StyleRetention', 'StyleAcc', 'DanceRate', 'Div_k', 'Div_g', 'BAS', 'FSR'):
            if k in gt:
                n[f'{k}_gt'] = gt[k]
        for k in halves[0]:
            if k in HIGHER or k in ('Div_k', 'Div_g'):
                v = [h[k] for h in halves]
                n[f'{k}_halfset_mean'], n[f'{k}_halfset_std'] = float(np.nanmean(v)), float(np.nanstd(v))
        for k in ('StyleRetention', 'StyleAcc', 'DanceRate', 'StyleFD', 'FID_k', 'FID_g'):
            if others and k in others[next(iter(others))]:
                pick = (max if HIGHER[k] else min)(others, key=lambda o: others[o][k])
                n[f'{k}_complete_forgetting'], n[f'{k}_complete_forgetting_from'] = others[pick][k], pick
        n['class_mass_gt'] = gt.get('class_mass')
        noise[s] = n

    os.makedirs(a.ref_dir, exist_ok=True)
    flat = dict(norm)
    for s in sessions:
        flat.update({f'{s}_kin': refs[s]['kin'], f'{s}_geo': refs[s]['geo'], f'{s}_emb': refs[s]['emb']})
    np.savez(os.path.join(a.ref_dir, 'refs.npz'), **flat)
    json.dump(noise, open(os.path.join(a.ref_dir, 'noise.json'), 'w'), indent=2)

    print('\nWhich metric can see forgetting? Real test dance of each session: its value, the split-half noise, and the'
          '\nvalue when another session\'s real dance is scored instead (complete forgetting). SNR = gap / noise.')
    for s in sessions:
        n = noise[s]
        print(f"\n{s} {fam[s]}: {n['n_songs']} songs, {n['n_clips']} clips, {n['n_windows']} windows")
        if 'StyleRetention_complete_forgetting' not in n:
            continue
        sr = n['StyleRetention_gt'] - n['StyleRetention_complete_forgetting']
        print(f"  StyleRetention {n['StyleRetention_gt']:.3f} (noise {n['StyleRetention_halfset_std']:.3f}) -> complete "
              f"forgetting {n['StyleRetention_complete_forgetting']:.3f} ({n['StyleRetention_complete_forgetting_from']})"
              f"  SNR {sr / max(n['StyleRetention_halfset_std'], 1e-9):.1f}")
        for k in ('FID_k', 'FID_g', 'StyleFD'):
            gap = n[f'{k}_complete_forgetting'] - n[f'{k}_halfset_mean']
            print(f"  {k:8s} floor {n[f'{k}_halfset_mean']:8.3f} +- {n[f'{k}_halfset_std']:.3f} -> complete forgetting "
                  f"{n[f'{k}_complete_forgetting']:8.3f}  SNR {gap / max(n[f'{k}_halfset_std'], 1e-9):.1f}")
        print(f"  real dance: Div_k {n['Div_k_gt']:.2f}  Div_g {n['Div_g_gt']:.2f}  BAS {n['BAS_gt']:.3f}  "
              f"FSR {n['FSR_gt']:.3f}  DanceRate {n.get('DanceRate_gt', float('nan')):.3f}")
    print(f"\nwrote {a.ref_dir}/refs.npz and noise.json")


# ---------------------------------------------------------------- eval: one model on every session's test music
def cmd_eval(a):
    enc = StyleEncoder(a.encoder)
    norm, refs, noise = load_refs(a.ref_dir)
    clip_seconds, max_clips = noise['meta']['clip_seconds'], noise['meta']['max_clips_per_song']
    sessions = [s for s in a.sessions.split(',') if s in refs]
    data = {s: load_clips(a.manifest_dir, s, clip_seconds, max_clips) for s in sessions}
    for s in sessions:
        assert [c['id'] for c in data[s]] == noise['meta']['clips'][s], f'{s}: test clips differ from the build'
    mode = 'gt' if a.gt else 'recon' if a.recon else 'model'

    model, t0 = None, time.time()
    if a.joints_in:
        z = np.load(a.joints_in)
    elif mode != 'gt':
        from unimumo.models import UniMuMo
        from eval_nll import load_lm_state
        model = UniMuMo.from_checkpoint(a.ckpt)
        if a.lm_state:
            load_lm_state(model, a.lm_state)
        model.music_motion_lm.eval()

    results, saved = {}, {}
    for s in sessions:
        clips = data[s]
        fam = clips[0]['family']
        if a.joints_in:
            joints = [z[f'{s}_{i}'] for i in range(len([k for k in z.files if k.startswith(f'{s}_')]))]
        elif mode == 'gt':
            joints = [gt_joints(c['feat']) for c in clips]
        elif mode == 'recon':
            joints = reconstruct(model, clips)
        else:
            print(f'[{s}] dancing {len(clips)} clips x {a.samples} sample(s)')
            joints = generate(model, clips, a.samples, a.batch_size, a.cfg, a.seed)
        reps = len(joints) // len(clips)
        cs = ClipSet(joints, [c['wav'] for c in clips] * reps, SR, enc)
        r = set_metrics(cs, fam, norm, ref=refs[s], encoder=enc)
        r['family'] = fam
        results[f'{s}_test'] = r
        saved.update({f'{s}_{i}': j for i, j in enumerate(joints)})
        top = ', '.join(f'{c} {v:.2f}' for c, v in sorted(r['class_mass'].items(), key=lambda x: -x[1]))
        print(f"[{s}_test {fam}] StyleRetention {r['StyleRetention']:.3f} StyleAcc {r['StyleAcc']:.3f} DanceRate "
              f"{r.get('DanceRate', float('nan')):.3f} StyleFD {r['StyleFD']:.3f} | FID_k {r['FID_k']:.2f} FID_g "
              f"{r['FID_g']:.2f} Div_k {r['Div_k']:.2f} Div_g {r['Div_g']:.2f} BAS {r['BAS']:.3f} FSR {r['FSR']:.3f} "
              f"| {r['n_clips']} clips | mass: {top}", flush=True)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    meta = {'mode': mode, 'ckpt': a.ckpt, 'lm_state': a.lm_state, 'encoder': a.encoder, 'ref_dir': a.ref_dir,
            'samples': a.samples, 'cfg': a.cfg, 'seed': a.seed, 'clip_seconds': clip_seconds,
            'joints_in': a.joints_in, 'seconds': round(time.time() - t0, 1)}
    json.dump({'meta': meta, 'results': results}, open(a.out, 'w'), indent=2)
    if mode != 'gt' and not a.joints_in:  # keep generated joints, so they can be re-scored without regenerating
        np.savez_compressed(os.path.splitext(a.out)[0] + '.joints.npz', **saved)
    print(f'wrote {a.out} ({time.time() - t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'eval'])
    ap.add_argument('--encoder', default='pretrained/style_encoder_finedance.pt')
    ap.add_argument('--ref_dir', default='outputs/dance_eval')
    ap.add_argument('--manifest_dir', default='data/finedance/manifests')
    ap.add_argument('--sessions', default='s1,s2,s3')
    ap.add_argument('--seed', type=int, default=0)
    # build
    ap.add_argument('--clip_seconds', type=float, default=10.0)
    ap.add_argument('--max_clips_per_song', type=int, default=0, help='0 = every clip of every test song')
    ap.add_argument('--halves', type=int, default=20, help='split-half draws for the noise floors')
    # eval
    ap.add_argument('--ckpt', default='pretrained/full.ckpt')
    ap.add_argument('--lm_state', default=None, help='LM weights after a session (default: the pretrained model)')
    ap.add_argument('--gt', action='store_true', help='score the real test dance')
    ap.add_argument('--recon', action='store_true', help="score the motion tokenizer's reconstruction of it")
    ap.add_argument('--joints_in', default=None, help='re-score saved joints (<out>.joints.npz) instead of generating')
    ap.add_argument('--samples', type=int, default=1, help='generated samples per clip (DiscoForcing pools 3)')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--cfg', type=float, default=1.0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.cmd == 'build':
        cmd_build(a)
    else:
        assert a.out, '--out is required for eval'
        cmd_eval(a)


if __name__ == '__main__':
    main()
