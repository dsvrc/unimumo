"""StyleRetention of UniMuMo's generated dance (the DiscoForcing CL metric), per session test set.

For every test song: the first --clips_per_song music clips of --clip_seconds are danced by the model
(music -> motion, null text), the 60 fps joints are downsampled to 30 fps, cut into 5 s windows (1 s stride)
and classified by the frozen FineDance style encoder trained in DiscoForcing
(raw_data/cl_sessions/_style_encoder/finedance/encoder.pt; copy it to the cluster).

  StyleRetention = mean probability mass on the set's own family (forgetting = m_ii - m_it)
  StyleAcc       = share of windows whose arg-max family is the set's own
  DanceRate      = 1 - P(other)   (only if the encoder has the reject class)
  class_mass     = mean probability per family (shows WHICH style the model dances instead)

Reference rows: --gt scores the real test dance (encoder domain check on UniMuMo's skeleton),
--recon scores UniMuMo's motion-tokenizer reconstruction of it (ceiling for any UniMuMo model).

  python cl/eval_style.py --encoder pretrained/style_encoder_finedance.pt --gt --recon --no_gen \
      --manifest s1_test=data/finedance/manifests/s1_street_test.jsonl ... --out outputs/style/reference.json
  python cl/eval_style.py --encoder pretrained/style_encoder_finedance.pt --lm_state runs/s1/lm.pt \
      --manifest s1_test=... --manifest s2_test=... --out outputs/style/after_s1.json

The encoder code below is copied from DiscoForcing metrics/style_encoder.py (inference part only).
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
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- style encoder (from DiscoForcing)
FPS, WIN, STRIDE = 30, 150, 30
L_HIP, R_HIP, L_SHO, R_SHO = 1, 2, 16, 17
OTHER = 'other'
FAMILY = {'Street': 'Street', 'jiewu': 'Street', 'Classic': 'Classic', 'Folk': 'Folk', 'Mix': 'Mix'}


def canonical_features(w):
    """(B, T, 22, 3) global joints -> (B, 129, T): pelvis-relative pose in the window's facing frame (63),
    root height (1), root horizontal velocity (2) and pose velocity (63)."""
    root = w[:, :, 0]
    across = ((w[:, :, L_HIP] - w[:, :, R_HIP]) + (w[:, :, L_SHO] - w[:, :, R_SHO])).mean(1)
    theta = torch.atan2(-across[:, 2], across[:, 0])
    c, s = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
    rel = w[:, :, 1:] - root[:, :, None]
    rel = torch.stack([rel[..., 0] * c - rel[..., 2] * s, rel[..., 1], rel[..., 0] * s + rel[..., 2] * c], -1)
    c2, s2 = c[..., 0], s[..., 0]
    root_xz = torch.stack([root[..., 0] * c2 - root[..., 2] * s2, root[..., 0] * s2 + root[..., 2] * c2], -1)

    def vel(a):
        d = torch.diff(a, dim=1) * FPS
        return torch.cat([d[:, :1], d], 1)

    feats = torch.cat([rel.flatten(2), root[..., 1:2], vel(root_xz), vel(rel).flatten(2)], -1)
    return feats.transpose(1, 2)


class ResBlock(nn.Module):
    def __init__(self, ch, dilation, stride):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 5, stride=stride, padding=2 * dilation, dilation=dilation)
        self.n1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv1d(ch, ch, 5, padding=2, dilation=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.skip = nn.Conv1d(ch, ch, 1, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        return F.gelu(self.n2(self.c2(h)) + self.skip(x))


class StyleNet(nn.Module):
    def __init__(self, in_ch, n_classes, width=128, emb=64, p_drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(in_ch, width, 5, padding=2), nn.GroupNorm(8, width), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(width, d, s) for d, s in
                                      [(1, 1), (2, 2), (4, 1), (8, 2), (1, 1), (2, 2)]])
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Sequential(nn.Linear(2 * width, 256), nn.GELU(), nn.Linear(256, emb))
        self.head = nn.Linear(emb, n_classes)

    def forward(self, x):
        h = self.blocks(self.inp(x))
        e = self.proj(self.drop(torch.cat([h.mean(-1), h.amax(-1)], 1)))
        return F.normalize(e, dim=1), self.head(e)


def sequence_windows(joints_list):
    wins, owner = [], []
    for i, j in enumerate(joints_list):
        if len(j) < WIN:
            j = np.concatenate([j, np.repeat(j[-1:], WIN - len(j), 0)])
        for s in range(0, len(j) - WIN + 1, STRIDE):
            wins.append(j[s:s + WIN])
            owner.append(i)
    return np.stack(wins).astype(np.float32), np.array(owner)


class StyleEncoder:
    def __init__(self, ckpt, device=None):
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = list(ck['classes'])
        self.temperature = float(ck['temperature'])
        self.mean = ck['mean'].to(self.device)[None, :, None]
        self.std = ck['std'].to(self.device)[None, :, None]
        self.model = StyleNet(ck['in_ch'], len(self.classes)).to(self.device)
        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()

    @torch.no_grad()
    def encode(self, joints_list, batch=256):
        W, owner = sequence_windows(joints_list)
        embs, logits = [], []
        for k in range(0, len(W), batch):
            x = (canonical_features(torch.from_numpy(W[k:k + batch]).to(self.device)) - self.mean) / self.std
            z, lg = self.model(x)
            embs.append(z.float().cpu())
            logits.append(lg.float().cpu())
        prob = torch.softmax(torch.cat(logits) / self.temperature, 1).numpy()
        return {'emb': torch.cat(embs).numpy(), 'prob': prob, 'owner': owner}

    def scores(self, prob, family):
        own = np.array([c == family for c in self.classes])
        out = {'StyleRetention': float(prob[:, own].sum(1).mean()),
               'StyleAcc': float(own[prob.argmax(1)].mean()),
               'class_mass': {c: float(prob[:, i].mean()) for i, c in enumerate(self.classes)},
               'n_windows': int(len(prob))}
        if OTHER in self.classes:
            out['DanceRate'] = float(1.0 - prob[:, self.classes.index(OTHER)].mean())
        return out


# ---------------------------------------------------------------- clips, generation, references
def load_clips(manifest, clips_per_song, clip_seconds):
    import librosa
    clips = []
    for it in (json.loads(l) for l in open(manifest) if l.strip()):
        wav, _ = librosa.load(it['music'], sr=32000)
        feat = np.load(it['motion']).astype(np.float32)                      # 263-d @ 60 fps
        dur = min(len(wav) / 32000, len(feat) / 60)
        fam = it.get('family') or FAMILY[it['style']]
        for k in range(clips_per_song):
            a = k * clip_seconds
            if a + clip_seconds > dur:
                break
            clips.append({'id': f"{it['id']}@{a:g}", 'family': fam,
                          'wav': wav[int(a * 32000):int((a + clip_seconds) * 32000)],
                          'feat': feat[int(a * 60):int((a + clip_seconds) * 60)]})
    return clips


@torch.no_grad()
def dance(model, clips, samples, batch_size, cfg, seed):
    """Music -> dance with the model; returns 30 fps global joints, one array per (clip, sample)."""
    mm = model.music_motion_lm
    out = []
    for s in range(samples):
        for b in range(0, len(clips), batch_size):
            torch.manual_seed(seed + 1000 * s + b)
            batch = clips[b:b + batch_size]
            wav = np.stack([c['wav'] for c in batch])[:, None, :]
            codes = model.encode_music(wav)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                motion = mm.generate_single_modality(music_code=codes, motion_code=None,
                                                     text_description=['<separation>'] * len(batch),
                                                     conditional_guidance_scale=cfg)
            feats = model.motion_vqvae.decode_from_code(codes, motion).float()
            out += [j[::2] for j in model.motion_vec_to_joint(feats)]
    return out


@torch.no_grad()
def references(model, clips, kind):
    from unimumo.motion.motion_process import recover_from_ric
    out = []
    for c in clips:
        if kind == 'gt':
            j = recover_from_ric(torch.from_numpy(c['feat']).float()[None], 22)[0].numpy()
        else:  # tokenizer reconstruction, decoded with the clip's own music like generated motion
            codes = model.encode_music(c['wav'][None, None, :])
            mcodes = model.encode_motion(c['feat'][None])
            t = min(codes.shape[-1], mcodes.shape[-1])
            j = model.motion_vec_to_joint(model.motion_vqvae.decode_from_code(codes[..., :t], mcodes[..., :t]).float())[0]
        out.append(j[::2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True, help='DiscoForcing FineDance style encoder checkpoint')
    ap.add_argument('--ckpt', default='pretrained/full.ckpt')
    ap.add_argument('--lm_state', default=None)
    ap.add_argument('--manifest', action='append', required=True, help='name=path.jsonl (repeatable)')
    ap.add_argument('--clips_per_song', type=int, default=3)
    ap.add_argument('--clip_seconds', type=float, default=10.0)
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--cfg', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gt', action='store_true', help='also score the real test dance')
    ap.add_argument('--recon', action='store_true', help='also score the motion-tokenizer reconstruction')
    ap.add_argument('--no_gen', action='store_true', help='skip generation (reference rows only)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from unimumo.models import UniMuMo
    from eval_nll import load_lm_state
    enc = StyleEncoder(args.encoder)
    print(f'style encoder classes: {enc.classes} (temperature {enc.temperature:.2f})')
    model = UniMuMo.from_checkpoint(args.ckpt)
    if args.lm_state:
        load_lm_state(model, args.lm_state)
    model.music_motion_lm.eval()

    results = {}
    for spec in args.manifest:
        name, path = spec.split('=', 1)
        clips = load_clips(path, args.clips_per_song, args.clip_seconds)
        fams = sorted({c['family'] for c in clips})
        assert len(fams) == 1, f'{name}: one family per set expected, got {fams}'
        fam = fams[0]
        assert fam in enc.classes, f'{fam} is not an encoder class {enc.classes}'
        rows = []
        if not args.no_gen:
            rows.append(('', lambda: dance(model, clips, args.samples, args.batch_size, args.cfg, args.seed)))
        if args.gt:
            rows.append(('@gt', lambda: references(model, clips, 'gt')))
        if args.recon:
            rows.append(('@recon', lambda: references(model, clips, 'recon')))
        for suffix, fn in rows:
            t0 = time.time()
            r = enc.scores(enc.encode(fn())['prob'], fam)
            r['seconds'] = round(time.time() - t0, 1)
            results[name + suffix] = r
            top = ', '.join(f'{c} {v:.2f}' for c, v in sorted(r['class_mass'].items(), key=lambda x: -x[1]))
            print(f"[{name}{suffix}] family {fam}: StyleRetention {r['StyleRetention']:.3f} | StyleAcc "
                  f"{r['StyleAcc']:.3f}" + (f" | DanceRate {r['DanceRate']:.3f}" if 'DanceRate' in r else '')
                  + f" | {len(clips)} clips, {r['n_windows']} windows | mass: {top} | {r['seconds']}s")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    meta = {k: getattr(args, k) for k in ['ckpt', 'lm_state', 'encoder', 'clips_per_song', 'clip_seconds',
                                          'samples', 'cfg', 'seed']}
    with open(args.out, 'w') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
