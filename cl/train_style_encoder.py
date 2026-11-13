"""Train the FineDance style encoder for UniMuMo's sessions with DiscoForcing's recipe.

Port of DiscoForcing metrics/style_encoder.py train(..., negatives=True): StyleNet on 5 s windows of global
joints at 30 fps, style cross-entropy (label smoothing 0.1) + 0.5 x supervised contrastive loss, artefact
augmentations (smoothing, jitter, tempo, body scale, root drift), a reject class 'other', AdamW 1e-3 for
6000 steps with warm-up and cosine decay, then temperature scaling on the validation windows. The network,
features and windows are the ones in cl/eval_style.py (copied from DiscoForcing), so the checkpoint loads there.

What differs from DiscoForcing, and why:
  * Joints come from UniMuMo's own FineDance features (cl/motion_convert.py -> recover_from_ric), i.e. the
    HumanML3D skeleton UniMuMo generates on, so real and generated dance are scored in the same domain.
  * Classes = our sessions' families Street, Folk, Classic (Mix dropped, as DiscoForcing's plan recommends,
    since Mix overlaps with Street) + 'other'. Train = the sessions' train songs, val = their val songs;
    test songs are only used by the report at the end.
  * Negatives ('other'), DiscoForcing's NEG_MIX:
      babel      0.4   everyday motion = DiscoForcing's BABEL negatives (dance-annotated sequences excluded),
                       retargeted to the HumanML3D skeleton so body proportions cannot be a shortcut
      vae_random 0.3   generator-like garbage = random token sequences decoded by UniMuMo's motion tokenizer
                       (DiscoForcing decodes random latents with its frozen VAE; half of them held/smoothed)
      static     0.15  a held pose with small noise      } built on the fly from dance windows
      shuffled   0.15  time-shuffled dance                }

Run from the repo root on a GPU node (a few minutes on a GPU):
    python cl/train_style_encoder.py --babel_dir /path/to/DiscoForcing/raw_data/cl_sessions/_style_encoder
Writes pretrained/style_encoder_finedance.pt (read by cl/eval_style.py) and a validation report next to it
(.json): the swap matrix on the test songs (clean, smoothed sigma = 2, mirrored) and the mass non-dance probes
put on any session. DiscoForcing's criteria: right session >= 0.8, wrong <= 0.1, non-dance <= 0.2.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

from eval_style import FPS, WIN, STRIDE, OTHER, canonical_features, StyleNet, StyleEncoder, sequence_windows

SESSIONS = [('s1_street', 'Street'), ('s2_folk', 'Folk'), ('s3_classic', 'Classic')]
# ---------------------------------------------------------------- DiscoForcing training pieces (unchanged)
AUG = {"smooth": (0.5, 4.0), "jitter_cm": (0.2, 2.0), "tempo": (0.8, 1.25), "scale": (0.9, 1.1), "drift_mps": 0.1}
NEG_MIX = {"babel": 0.4, "vae_random": 0.3, "static": 0.15, "shuffled": 0.15}
LR_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]
MIRROR_PERM = list(range(22))
for _a, _b in LR_PAIRS:
    MIRROR_PERM[_a], MIRROR_PERM[_b] = _b, _a


def resample_time(j, speed, length=None):
    """Linear time resampling; speed > 1 plays the motion faster."""
    n_new = length or max(2, int(round((len(j) - 1) / speed)) + 1)
    t = np.linspace(0, len(j) - 1, n_new)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(j) - 1)
    w = (t - lo)[:, None, None]
    return j[lo] * (1 - w) + j[hi] * w


def sample_window(seq, rng, aug):
    """One random 150-frame training window, optionally with artefact augmentations."""
    speed = rng.uniform(*AUG["tempo"]) if aug and rng.random() < 0.5 else 1.0
    L = min(len(seq), int(round(WIN * speed)))
    s = rng.integers(0, len(seq) - L + 1)
    w = seq[s:s + L].astype(np.float64)
    if L != WIN:
        w = resample_time(w, 1.0, length=WIN)
    if aug:
        if rng.random() < 0.5:
            w = gaussian_filter1d(w, rng.uniform(*AUG["smooth"]), axis=0, mode="nearest")
        if rng.random() < 0.5:
            w = w + rng.normal(0, rng.uniform(*AUG["jitter_cm"]) / 100.0, w.shape)
        w = w * rng.uniform(*AUG["scale"])
        if rng.random() < 0.3:  # slow root drift (foot sliding)
            v = rng.normal(0, AUG["drift_mps"], 2) / FPS
            w[..., 0] += (np.arange(WIN) * v[0])[:, None]
            w[..., 2] += (np.arange(WIN) * v[1])[:, None]
    return w.astype(np.float32)


def negative_window(kind, rng, pool, dance):
    """One training window of the reject class."""
    if kind in pool:
        return sample_window(pool[kind][rng.integers(len(pool[kind]))], rng, aug=True)
    w = sample_window(dance[rng.integers(len(dance))], rng, aug=True)
    if kind == "static":
        w = np.repeat(w[rng.integers(WIN)][None], WIN, 0)
        return (w + rng.normal(0, rng.uniform(0, 0.01), w.shape)).astype(np.float32)
    return w[rng.permutation(WIN)]  # time-shuffled dance


def supcon(z, y, tau=0.1):
    """Supervised contrastive loss (Khosla et al., 2020) on L2-normalised embeddings."""
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = (z @ z.T / tau).masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -((log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


def corrupt(j, kind):
    """Validation corruptions: smoothing sigma = 2 frames (seen range), mirroring (never trained on)."""
    j = j.astype(np.float64).copy()
    if kind == "smooth2":
        j = gaussian_filter1d(j, 2, axis=0, mode="nearest")
    elif kind == "mirror":
        j[..., 0] *= -1
        j = j[:, MIRROR_PERM]
    return j.astype(np.float32)


# ---------------------------------------------------------------- data on UniMuMo's skeleton
def feats_to_joints(feat):
    """UniMuMo 263-d features (60 fps) -> global joints (T, 22, 3) at 30 fps, as cl/eval_style.py does."""
    from unimumo.motion.motion_process import recover_from_ric
    return recover_from_ric(torch.from_numpy(feat).float()[None], 22)[0].numpy()[::2].astype(np.float32)


def load_sessions(manifest_dir):
    """{split: [(id, family, joints)]} from the session manifests written by cl/finedance_sessions.py."""
    data = {'train': [], 'val': [], 'test': []}
    for sess, fam in SESSIONS:
        for split in data:
            for line in open(os.path.join(manifest_dir, f'{sess}_{split}.jsonl')):
                if line.strip():
                    it = json.loads(line)
                    assert it.get('family', fam) == fam, f"{it['id']}: family {it.get('family')} in {sess}"
                    j = feats_to_joints(np.load(it['motion']))
                    if not np.isfinite(j).all():  # one bad window would turn the whole training into NaN
                        print(f"WARNING: skipping {it['id']} ({split}): non-finite joints")
                        continue
                    data[split].append((it['id'], fam, j))
    return data


def babel_negatives(babel_dir, split, n, target_offset, max_frames, seed):
    """DiscoForcing's BABEL negatives (global joints, 30 fps), cropped to max_frames, retargeted to the
    HumanML3D skeleton (HumanML3D's uniform_skeleton) and put on the floor."""
    from motion_convert import uniform_skeleton
    path = os.path.join(babel_dir, f'neg_babel_{split}_{n}_s0.npz')
    if not os.path.exists(path):
        sys.exit(f'{path} not found: --babel_dir must be DiscoForcing\'s raw_data/cl_sessions/_style_encoder')
    z = np.load(path)
    rng = np.random.default_rng(seed)
    out = []
    for k in z.files:
        j = z[k].astype(np.float64)
        if len(j) < WIN:
            continue
        if len(j) > max_frames:
            s = rng.integers(0, len(j) - max_frames + 1)
            j = j[s:s + max_frames]
        j = uniform_skeleton(j.copy(), target_offset)
        j[..., 1] -= j[..., 1].min()
        if np.isfinite(j).all():
            out.append(j.astype(np.float32))
    print(f'BABEL {split}: {len(out)} of {len(z.files)} sequences kept (shorter than a window or non-finite dropped)')
    return out


@torch.no_grad()
def tokenizer_garbage(model, music_codes, n, kind, seed, batch=50):
    """Joints decoded by UniMuMo's motion tokenizer from synthetic token sequences (5 s each, real music codes):
      random   - i.i.d. uniform tokens                      } training negatives (half each)
      held     - random tokens, each held for 2-12 steps    }
      constant - one random token per codebook, held for the whole clip (never trained on)"""
    vq = model.motion_vqvae
    dev = next(vq.parameters()).device
    bins = int(getattr(vq.quantizer, 'bins', 2048))
    g = torch.Generator().manual_seed(seed)
    K, T = music_codes.shape[1], music_codes.shape[2]
    out = []
    for b in range(0, n, batch):
        m = min(batch, n - b)
        mu = music_codes[torch.randint(0, len(music_codes), (m,), generator=g)]
        if kind == 'random':
            tok = torch.randint(0, bins, (m, K, T), generator=g)
        elif kind == 'held':
            tok = torch.empty((m, K, T), dtype=torch.long)
            for i in range(m):
                h = int(torch.randint(2, 13, (1,), generator=g))
                tok[i] = torch.randint(0, bins, (K, -(-T // h)), generator=g).repeat_interleave(h, 1)[:, :T]
        elif kind == 'constant':
            tok = torch.randint(0, bins, (m, K, 1), generator=g).expand(m, K, T).contiguous()
        else:
            raise ValueError(kind)
        feats = vq.decode_from_code(mu.to(dev), tok.to(dev)).float()
        out += [j[::2].astype(np.float32) for j in model.motion_vec_to_joint(feats) if np.isfinite(j).all()]
    return out


# ---------------------------------------------------------------- training (DiscoForcing train(), negatives=True)
def train(data, pool, babel_val, steps, seed, device):
    genres = [fam for _, fam in SESSIONS]
    classes = genres + [OTHER]
    dance = [j for _, _, j in data['train']]
    by_class = {c: [j for _, f, j in data['train'] if f == c] for c in genres}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    neg_kinds, neg_p = list(NEG_MIX), np.array(list(NEG_MIX.values()))

    clean = np.stack([sample_window(dance[i], rng, aug=False) for i in rng.integers(0, len(dance), 4000)])
    with torch.no_grad():
        f = canonical_features(torch.from_numpy(clean).to(device))
        mean, std = f.mean((0, 2)), f.std((0, 2)) + 1e-6
    mu, sd = mean[None, :, None], std[None, :, None]

    model = StyleNet(f.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    per_class = max(4, 80 // len(genres))
    n_neg = 2 * per_class
    n_genre = per_class * len(genres)
    y = torch.cat([torch.arange(len(genres), device=device).repeat_interleave(per_class),
                   torch.full((n_neg,), len(genres), device=device, dtype=torch.long)])

    Wv, ov = sequence_windows([j for _, _, j in data['val']])
    yv = np.array([classes.index(data['val'][i][1]) for i in ov])
    Wb, _ = sequence_windows(babel_val)  # calibrate / monitor the reject decision on held-out everyday motion
    Wv, yv = np.concatenate([Wv, Wb]), np.concatenate([yv, np.full(len(Wb), len(genres))]).astype(np.int64)

    def val_logits():
        model.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(Wv), 256):
                x = (canonical_features(torch.from_numpy(Wv[k:k + 256]).to(device)) - mu) / sd
                out.append(model(x)[1].float().cpu())
        model.train()
        return torch.cat(out)

    t0 = time.time()
    model.train()
    for step in range(steps):
        windows = [sample_window(by_class[c][i], rng, aug=True)
                   for c in genres for i in rng.integers(0, len(by_class[c]), per_class)]
        windows += [negative_window(k, rng, pool, dance) for k in rng.choice(neg_kinds, n_neg, p=neg_p)]
        x = (canonical_features(torch.from_numpy(np.stack(windows).astype(np.float32)).to(device)) - mu) / sd
        z, logits = model(x)
        # the contrastive term shapes the style geometry only; the heterogeneous reject class gets CE only
        loss = F.cross_entropy(logits, y, label_smoothing=0.1) + 0.5 * supcon(z[:n_genre], y[:n_genre])
        if not math.isfinite(loss.item()):
            sys.exit(f'non-finite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0 or step + 1 == steps:
            pv = val_logits().argmax(1).numpy()
            g = yv < len(genres)
            print(f'step {step + 1}/{steps} loss {loss.item():.3f} | val style acc {float((pv[g] == yv[g]).mean()):.3f}, '
                  f'everyday motion rejected {float((pv[~g] == len(genres)).mean()):.3f} ({time.time() - t0:.0f}s)',
                  flush=True)

    lv, yt = val_logits(), torch.from_numpy(yv)  # temperature scaling (Guo et al., 2017)
    temps = np.linspace(0.3, 5.0, 95)
    temperature = float(temps[np.argmin([F.cross_entropy(lv / t, yt).item() for t in temps])])
    print(f'temperature {temperature:.2f}')
    return {'state_dict': model.state_dict(), 'mean': mean.cpu(), 'std': std.cpu(), 'in_ch': int(f.shape[1]),
            'classes': classes, 'session_of_class': list(range(len(genres))) + [-1], 'temperature': temperature,
            'train_sequences': len(dance), 'steps': steps, 'aug': AUG, 'negatives': NEG_MIX,
            'window': WIN, 'stride': STRIDE, 'seed': seed,
            'skeleton': 'HumanML3D (UniMuMo 263-d features -> recover_from_ric), 30 fps'}


# ---------------------------------------------------------------- validation report
def report(ckpt_path, data, probes):
    enc = StyleEncoder(ckpt_path)
    genres = [c for c in enc.classes if c != OTHER]
    rep = {}
    for kind in ('clean', 'smooth2', 'mirror'):
        rows = {}
        for c in genres:
            js = [corrupt(j, kind) for _, f, j in data['test'] if f == c]
            p = enc.encode(js)['prob']
            rows[c] = {k: float(p[:, i].mean()) for i, k in enumerate(enc.classes)}
            rows[c]['window_acc'] = float((p.argmax(1) == enc.classes.index(c)).mean())
            rows[c]['n_windows'] = int(len(p))
        rep[kind] = rows
        print(f'\ntest songs, {kind}: mean probability per class (rows = true family)')
        print('            ' + ''.join(f'{k:>9s}' for k in enc.classes) + '  window acc  windows')
        for c in genres:
            print(f'  {c:9s} ' + ''.join(f'{rows[c][k]:9.3f}' for k in enc.classes)
                  + f'  {rows[c]["window_acc"]:10.3f}  {rows[c]["n_windows"]:7d}')
    right = min(rep['clean'][c][c] for c in genres)
    wrong = max(rep['clean'][c][k] for c in genres for k in genres if k != c)
    drop = {kind: max(rep['clean'][c][c] - rep[kind][c][c] for c in genres) for kind in ('smooth2', 'mirror')}
    print(f'\nclean: right family >= {right:.3f} (criterion >= 0.8), wrong family <= {wrong:.3f} (<= 0.1); '
          f'worst retention drop smoothed {drop["smooth2"]:.3f} (<= 0.05), mirrored {drop["mirror"]:.3f} (<= 0.10)')
    rep['summary'] = {'right_min': right, 'wrong_max': wrong, 'drop_smooth2': drop['smooth2'], 'drop_mirror': drop['mirror']}
    print('\nnon-dance probes: mean mass on the most-hit session family (criterion <= 0.2) and on other')
    for name, js in probes.items():
        p = enc.encode(js)['prob']
        m = p[:, :len(genres)].mean(0)
        rep[f'probe_{name}'] = {'max_session_mass': float(m.max()), 'other': float(p[:, -1].mean()), 'n_windows': int(len(p))}
        print(f'  {name:34s} max session mass {m.max():.3f} ({genres[int(m.argmax())]}) | other {p[:, -1].mean():.3f}'
              f' | {len(p)} windows')
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--babel_dir', required=True,
                    help="DiscoForcing's raw_data/cl_sessions/_style_encoder (neg_babel_{train_3000,val_600}_s0.npz)")
    ap.add_argument('--manifest_dir', default='data/finedance/manifests')
    ap.add_argument('--music_cache', default='outputs/token_cache/s1_train_w5_h2.5.pt',
                    help='token cache whose music codes condition the decoded garbage')
    ap.add_argument('--ckpt', default='pretrained/full.ckpt')
    ap.add_argument('--out', default='pretrained/style_encoder_finedance.pt')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_garbage', type=int, default=600)
    ap.add_argument('--babel_max_frames', type=int, default=600, help='crop BABEL sequences (30 fps) before retargeting')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t0 = time.time()
    data = load_sessions(args.manifest_dir)
    for split, items in data.items():
        per = {fam: sum(f == fam for _, f, _ in items) for _, fam in SESSIONS}
        mins = sum(len(j) for _, _, j in items) / FPS / 60
        print(f'{split:5s}: {len(items)} songs {per}, {mins:.1f} min')

    from motion_convert import target_offsets_from_features
    first_train = json.loads(open(os.path.join(args.manifest_dir, f'{SESSIONS[0][0]}_train.jsonl')).readline())
    tgt, _ = target_offsets_from_features(first_train['motion'])     # HumanML3D's target skeleton
    babel_train = babel_negatives(args.babel_dir, 'train', 3000, tgt, args.babel_max_frames, args.seed)
    babel_val = babel_negatives(args.babel_dir, 'val', 600, tgt, args.babel_max_frames, args.seed)
    print(f'BABEL negatives retargeted to the HumanML3D skeleton: {len(babel_train)} train, {len(babel_val)} val')

    from unimumo.models import UniMuMo
    model = UniMuMo.from_checkpoint(args.ckpt)
    music_codes = torch.load(args.music_cache, weights_only=False)['music_code']
    half = args.n_garbage // 2
    garbage = (tokenizer_garbage(model, music_codes, half, 'random', args.seed)
               + tokenizer_garbage(model, music_codes, args.n_garbage - half, 'held', args.seed + 1))
    garbage_unseen = {'tokenizer garbage, new seed': tokenizer_garbage(model, music_codes, 100, 'random', 101),
                      'constant tokens (never trained on)': tokenizer_garbage(model, music_codes, 100, 'constant', 102)}
    del model
    torch.cuda.empty_cache()
    print(f'decoded {len(garbage)} garbage clips from random tokens ({time.time() - t0:.0f}s so far)')

    ckpt = train(data, {'babel': babel_train, 'vae_random': garbage}, babel_val[::2], args.steps, args.seed, device)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(ckpt, args.out)
    print(f'saved {args.out}')

    rng = np.random.default_rng(args.seed + 7)
    Wt, _ = sequence_windows([j for _, _, j in data['test']])
    Wt = Wt[rng.permutation(len(Wt))[:300]]
    probes = {'everyday motion (BABEL val, held out)': babel_val[1::2], **garbage_unseen,
              'static poses (test dance)': [np.repeat(w[rng.integers(WIN)][None], WIN, 0) for w in Wt],
              'time-shuffled test dance': [w[rng.permutation(WIN)] for w in Wt]}
    rep = report(args.out, data, probes)
    rep['meta'] = {k: v for k, v in ckpt.items() if k not in ('state_dict', 'mean', 'std')}
    rep_path = os.path.splitext(args.out)[0] + '.json'
    with open(rep_path, 'w') as f:
        json.dump(rep, f, indent=2)
    print(f'wrote {rep_path} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
