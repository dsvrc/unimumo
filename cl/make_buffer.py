"""Replay buffers for the UniMuMo continual sessions.

A buffer is a token-cache dict {'music_code', 'motion_code', 'ids', 'weights', ...} that
cl/train_session.py --replay reads (sampling windows with probability proportional to 'weights').

  # ER: random 10% of a session's training windows
  python cl/make_buffer.py random --train_cache runs/s1/train_tokens.pt --out runs/buffers/er_s1.pt

  # TMR: music-stream trajectories phi from the session checkpoint -> C_k -> rank -> whitening -> NN-OMP
  python cl/make_buffer.py tmr --lm_state runs/s1/lm.pt --train_cache runs/s1/train_tokens.pt \
      --out runs/buffers/tmr_s1.pt [--layer 12] [--solver omp]

  # before S3: one buffer for all past sessions
  python cl/make_buffer.py merge --inputs runs/buffers/er_s1.pt runs/buffers/er_s2.pt --out runs/buffers/er_s1s2.pt

NOTE: tmr_select() below is a PROVISIONAL reading of the paper (Eqs. 10-14). The rank criterion of
Eq. 12 is replaced by a 95%-explained-variance rule (or a PPCA-BIC, --rank_rule ppca_bic). For the
paper's numbers, replace tmr_select() with the implementation used for GestureLSM / TalkSHOW / MOSPA,
so all four backbones use identical selection.
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import torch


def subset(cache, idx, weights=None, extra=None):
    idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    out = {'music_code': cache['music_code'][idx].clone(), 'motion_code': cache['motion_code'][idx].clone(),
           'ids': [cache['ids'][i] for i in idx.tolist()], 'indices': idx,
           'weights': torch.ones(len(idx)) if weights is None else torch.as_tensor(weights, dtype=torch.float)}
    out.update(extra or {})
    return out


# ------------------------------------------------------------------ phi: audio trajectories from the LM
@torch.no_grad()
def audio_trajectories(mm, music_codes, layer=12, pool=2, batch_size=32, use_bf16=True):
    """Music-stream hidden state of transformer layer `layer` (1-based; 0 = summed token embeddings),
    motion stream masked with the special token, null text, training-time causal mask.
    Returns float32 array (N, T/pool, D)."""
    lm = mm.model
    lm.eval()
    K, T = music_codes.shape[1], music_codes.shape[2]
    out = []
    for i in range(0, len(music_codes), batch_size):
        mu = music_codes[i:i + batch_size].to(mm.device)
        B = mu.shape[0]
        if layer == 0:
            h = sum(lm.emb[k](mu[:, k]) for k in range(K))                         # (B, T, D)
        else:
            pat = lm.pattern_provider.get_pattern(T)
            mus, _, _ = pat.build_pattern_sequence(mu, lm.special_token_id, keep_only_valid_steps=True)
            mot = torch.full_like(mus, lm.special_token_id)
            seq = torch.cat([mus, mot], dim=-1)
            src_mask = lm.get_self_attn_mask(mus.shape[-1], mot.shape[-1], 'music_motion')
            cond = mm.prepare_text_condition(['<separation>'] * B)
            cross_mask = torch.where(cond['description'][-1], 0., float('-inf'))
            grab = {}
            hook = lm.transformer.layers[layer - 1].register_forward_hook(lambda m, inp, o: grab.update(h=o))
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                lm(seq, [], src_mask=src_mask, condition_tensors=cond, cross_attn_mask=cross_mask)
            hook.remove()
            h = grab['h'][:, :mus.shape[-1]]                                         # music half
            h = h[:, -T:]                                                            # drop delay-pattern padding
        h = h.float()
        if pool > 1:
            h = torch.nn.functional.avg_pool1d(h.transpose(1, 2), pool).transpose(1, 2)
        out.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(out)


# ------------------------------------------------------------------ PROVISIONAL TMR selection
def choose_rank(lam, N, rule='var95'):
    """Stand-in for the paper's Eq. 12.
    'var95'   : smallest M explaining >= 95% of tr(C)
    'ppca_bic': probabilistic-PCA BIC with N samples (penalises T*M parameters)."""
    lam = np.clip(lam, 1e-12, None)
    T = len(lam)
    if rule == 'var95':
        return int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)
    best, best_m = np.inf, 1
    for m in range(1, T):
        sigma2 = lam[m:].mean()
        nll = N * (np.log(lam[:m]).sum() + (T - m) * np.log(sigma2))
        k = T * m - m * (m - 1) / 2 + m + 1
        crit = nll + k * np.log(N)
        if crit < best:
            best, best_m = crit, m
    return best_m


def nn_omp(Z, y, k, ridge=1e-3):
    """Non-negative OMP: add the atom (row of Z) most positively correlated with the residual,
    refit non-negative ridge weights on the support, repeat. Returns support indices and weights."""
    from scipy.optimize import nnls
    S, w = [], np.zeros(0)
    r = y.copy()
    for _ in range(k):
        corr = Z @ r
        if S:
            corr[S] = -np.inf
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break
        S.append(j)
        ZS = Z[S]
        Q = ZS @ ZS.T + ridge * np.eye(len(S))
        L = np.linalg.cholesky(Q)
        w, _ = nnls(L.T, np.linalg.solve(L, ZS @ y))
        r = y - ZS.T @ w
    return np.array(S), w


def tmr_select(phi, budget, solver='omp', rank=None, rank_rule='var95', ridge=1e-3):
    N, T, D = phi.shape
    X = phi - phi.mean(0, keepdims=True)                                             # centred trajectories
    C = np.zeros((T, T), dtype=np.float64)
    for i in range(0, N, 256):
        xb = X[i:i + 256].astype(np.float64)
        C += np.einsum('ntd,nsd->ts', xb, xb)
    C /= N * D                                                                        # temporal feature operator
    lam, U = np.linalg.eigh(C)
    lam, U = lam[::-1], U[:, ::-1]
    M = rank or choose_rank(lam, N, rank_rule)
    Um = np.ascontiguousarray(U[:, :M]).astype(np.float32)
    gamma = np.einsum('tm,ntd->nmd', Um, X) / np.sqrt(np.clip(lam[:M], 1e-12, None)).astype(np.float32)[None, :, None]
    Z = gamma.reshape(N, -1)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12                              # unit-norm z_i
    y = Z.sum(0)
    assert solver == 'omp', 'provisional code implements OMP only; CoSaMP / SSP come with your implementation'
    S, w = nn_omp(Z, y, budget, ridge)
    keep = w > 0
    S, w = S[keep], w[keep]
    w = w * len(w) / w.sum()                                                          # mean-one weights
    return S, w, {'rank': int(M), 'eigvals': lam[:64].copy(), 'explained': float(lam[:M].sum() / lam.sum())}


# ------------------------------------------------------------------ commands
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ['random', 'tmr']:
        p = sub.add_parser(name)
        p.add_argument('--train_cache', required=True)
        p.add_argument('--frac', type=float, default=0.10)
        p.add_argument('--out', required=True)
        p.add_argument('--seed', type=int, default=0)
        if name == 'tmr':
            p.add_argument('--lm_state', required=True, help='checkpoint of the session just finished')
            p.add_argument('--ckpt', default='pretrained/full.ckpt')
            p.add_argument('--layer', type=int, default=12)
            p.add_argument('--pool', type=int, default=2)
            p.add_argument('--rank', type=int, default=None, help='fix the rank M (overrides --rank_rule)')
            p.add_argument('--rank_rule', choices=['var95', 'ppca_bic'], default='var95')
            p.add_argument('--solver', default='omp')
    m = sub.add_parser('merge')
    m.add_argument('--inputs', nargs='+', required=True)
    m.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    if args.cmd == 'merge':
        bufs = [torch.load(p) for p in args.inputs]
        out = {'music_code': torch.cat([b['music_code'] for b in bufs]),
               'motion_code': torch.cat([b['motion_code'] for b in bufs]),
               'ids': sum([list(b['ids']) for b in bufs], []),
               'weights': torch.cat([torch.as_tensor(b['weights'], dtype=torch.float) for b in bufs]),
               'sources': args.inputs}
        torch.save(out, args.out)
        print(f'merged {[len(b["ids"]) for b in bufs]} -> {len(out["ids"])} windows: {args.out}')
        return

    cache = torch.load(args.train_cache)
    N = cache['music_code'].shape[0]
    budget = max(1, int(round(args.frac * N)))
    t0 = time.time()
    if args.cmd == 'random':
        idx = np.random.default_rng(args.seed).choice(N, budget, replace=False)
        buf = subset(cache, np.sort(idx), extra={'method': 'random'})
    else:
        from unimumo.models import UniMuMo
        from eval_nll import load_lm_state
        model = UniMuMo.from_checkpoint(args.ckpt)
        load_lm_state(model, args.lm_state)
        phi = audio_trajectories(model.music_motion_lm, cache['music_code'], args.layer, args.pool)
        print(f'phi: {phi.shape} (windows, frames, dims) from layer {args.layer} in {time.time() - t0:.0f}s')
        del model
        torch.cuda.empty_cache()
        S, w, info = tmr_select(phi, budget, args.solver, args.rank, args.rank_rule)
        print(f"rank M = {info['rank']} (explains {100 * info['explained']:.1f}% of trace); "
              f"selected {len(S)} / budget {budget}; weights min {w.min():.2f} max {w.max():.2f}")
        buf = subset(cache, S, w, extra={'method': f'tmr_provisional_{args.solver}', 'layer': args.layer, **info})
    torch.save(buf, args.out)
    print(f'{args.cmd}: {len(buf["ids"])} of {N} windows ({100 * len(buf["ids"]) / N:.1f}%) '
          f'in {time.time() - t0:.0f}s -> {args.out}')


if __name__ == '__main__':
    main()
