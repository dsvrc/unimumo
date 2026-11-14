"""Replay buffer of one continual session of UniMuMo, ported from DiscoForcing (scripts/build_replay_buffer.py,
utils/cl_replay.py; the selection core scripts/evidence/tmr_select.py is cl/tmr_select.py, unchanged).

Run right after session j has been trained, with this method's model after session j. Buffers start at session 1
(the pretrained checkpoint gets none): S2 replays the buffer of S1, S3 the buffers of S1 and S2.
Candidates = the session's training clips: non-overlapping windows of the training window length tiling every
training song, plus one end-aligned window when the remainder is at least half a window (DiscoForcing tile_clips;
songs shorter than one window are skipped, since UniMuMo's token windows have a fixed length). The buffer keeps
--ratio of them: budget = max(1, round(ratio x #candidates)) = 10 % of the session's clips.

  random     ER: uniform sample; weights 1
  tmr        TMR (paper Algorithm 1, OMP variant): music trajectories phi_i -> temporal operator -> M modes by the
             PPCA BIC (n_eff = N D) -> whitened modal atoms -> simplex-constrained OMP towards the mean atom. It stops
             early when no clip lowers the modal error (the buffer is then smaller than the budget, as in DiscoForcing);
             its weights x |buffer| (mean 1) weight the replay loss
  tmr_joint  TMR on the joint trajectory [music, dance], each block scaled to unit mean squared deviation (equal
             weight), M by the BIC with N_eff = #clips x channels, simplex OMP, k-center fill to the budget if the
             pursuit stops early; OMP weights (k-center clips 1e-3) normalised to mean 1 weight the replay loss

UniMuMo's counterparts of DiscoForcing's trajectories:
  music  DiscoForcing: WanModel.music_embedding, the model's learned music input embedding, at the token rate.
         Here: the LM's music input embedding sum_k emb_k(code_k) at every 50 Hz token (--music_layer 0, default);
         --music_layer L > 0 uses layer L's music-stream hidden state instead (motion masked, null text).
  dance  DiscoForcing: the VAE latents of the dance (the generator's target). Here: the motion tokenizer's latents
         of the clip, motion_vqvae.quantizer.decode(motion codes): 128-d at 50 Hz.

Output: <out> (token cache + 'weights' + 'session', read by cl/train_session.py --replay) and <out>.json (summary).

  python cl/make_buffer.py --method tmr --session 1 --manifest data/finedance/manifests/s1_street_train.jsonl \
      --lm_state runs/fd/s1/lm.pt --out buffers/fd/tmr/s1.pt
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

import tmr_select
from eval_nll import MOTION_FPS, MUSIC_SR, TOKEN_RATE, load_lm_state, read_manifest


# ---------------------------------------------------------------- candidates: tiled training clips
def tile_starts(n_tok, clip_len):
    """DiscoForcing tile_clips in tokens. The end-aligned start is moved down to a multiple of 5 tokens (= 6 motion
    frames at 60 fps), so music and motion stay sample-aligned."""
    if n_tok < clip_len:
        return []
    starts = list(range(0, n_tok - clip_len + 1, clip_len))
    if n_tok - (starts[-1] + clip_len) >= clip_len // 2:
        starts.append((n_tok - clip_len) // 5 * 5)
    return starts


@torch.no_grad()
def candidate_cache(get_model, manifest, window, path, batch_size=16):
    """Token cache of a session's candidate clips (built once; the tokenizers are frozen, so it is shared by every
    method and model)."""
    if os.path.exists(path):
        c = torch.load(path, weights_only=False)
        assert abs(c['window'] - window) < 1e-6 and c['manifest'] == manifest, f'{path} was built for another setting'
        return c
    import librosa
    model = get_model()
    clip_len = int(round(window * TOKEN_RATE))
    n_mu, n_mo = clip_len * MUSIC_SR // TOKEN_RATE, clip_len * MOTION_FPS // TOKEN_RATE
    mus, mots, ids, songs = [], [], [], []
    items = read_manifest(manifest)
    for it in items:
        wav, _ = librosa.load(it['music'], sr=MUSIC_SR)
        mot = np.load(it['motion']).astype(np.float32)
        n_tok = int(min(len(wav) / MUSIC_SR, len(mot) / MOTION_FPS) * TOKEN_RATE)
        for s in tile_starts(n_tok, clip_len):
            a, b = s * MUSIC_SR // TOKEN_RATE, s * MOTION_FPS // TOKEN_RATE
            mus.append(wav[a:a + n_mu])
            mots.append(mot[b:b + n_mo])
            ids.append(f"{it['id']}@{s / TOKEN_RATE:.2f}")
            songs.append(it['id'])
    music_codes, motion_codes = [], []
    for i in range(0, len(ids), batch_size):
        c_mu = model.encode_music(np.stack(mus[i:i + batch_size])[:, None, :]).cpu()
        c_mo = model.encode_motion(np.stack(mots[i:i + batch_size])).cpu()
        t = min(c_mu.shape[-1], c_mo.shape[-1], clip_len)
        music_codes.append(c_mu[..., :t])
        motion_codes.append(c_mo[..., :t])
    c = {'ids': ids, 'songs': songs, 'music_code': torch.cat(music_codes), 'motion_code': torch.cat(motion_codes),
         'window': window, 'manifest': manifest}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(c, path)
    print(f'candidates: {len(items)} songs -> {len(ids)} clips of {window:g} s -> {path}')
    return c


# ---------------------------------------------------------------- trajectories
@torch.no_grad()
def music_trajectories(mm, music_codes, layer=0, pool=1, batch_size=32, use_bf16=True):
    """(N, T, D) music trajectories of the clips: layer 0 = the LM's music input embedding sum_k emb_k(code_k) at
    the token rate; layer L > 0 = the music-stream hidden state of transformer layer L (motion stream masked with
    the special token, null text, the training-time attention mask)."""
    lm = mm.model
    lm.eval()
    K, T = music_codes.shape[1], music_codes.shape[2]
    out = []
    for i in range(0, len(music_codes), batch_size):
        mu = music_codes[i:i + batch_size].to(mm.device)
        B = mu.shape[0]
        if layer == 0:
            h = sum(lm.emb[k](mu[:, k]) for k in range(K))
        else:
            pat = lm.pattern_provider.get_pattern(T)
            mus, _, _ = pat.build_pattern_sequence(mu, lm.special_token_id, keep_only_valid_steps=True)
            seq = torch.cat([mus, torch.full_like(mus, lm.special_token_id)], dim=-1)
            src_mask = lm.get_self_attn_mask(mus.shape[-1], mus.shape[-1], 'music_motion')
            cond = mm.prepare_text_condition(['<separation>'] * B)
            cross_mask = torch.where(cond['description'][-1], 0., float('-inf'))
            grab = {}
            hook = lm.transformer.layers[layer - 1].register_forward_hook(lambda m, inp, o: grab.update(h=o))
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                lm(seq, [], src_mask=src_mask, condition_tensors=cond, cross_attn_mask=cross_mask)
            hook.remove()
            h = grab['h'][:, :mus.shape[-1]][:, -T:]  # music half, delay-pattern padding dropped
        h = h.float()
        if pool > 1:
            h = torch.nn.functional.avg_pool1d(h.transpose(1, 2), pool).transpose(1, 2)
        out.append(h.cpu())
    return torch.cat(out)


@torch.no_grad()
def dance_latents(model, motion_codes, pool=1, batch_size=64):
    """(N, T, 128) the motion tokenizer's latents of the clips (quantizer.decode of their motion codes)."""
    q = model.motion_vqvae.quantizer
    dev = next(model.motion_vqvae.parameters()).device
    out = []
    for i in range(0, len(motion_codes), batch_size):
        z = q.decode(motion_codes[i:i + batch_size].to(dev)).float()  # (B, 128, T)
        if pool > 1:
            z = torch.nn.functional.avg_pool1d(z, pool)
        out.append(z.transpose(1, 2).cpu())
    return torch.cat(out)


# ---------------------------------------------------------------- selection (DiscoForcing build_replay_buffer.py)
def select_random(N, budget, rng, **_):
    return sorted(rng.choice(N, budget, replace=False).tolist()), None, None


def temporal_operator_masked(phi, mask, chunk=64):
    """C = 1/(N D) sum_i phi~_i phi~_i^T (T x T) for zero-padded clips: every entry (t, s) is averaged over the clips
    valid at both t and s. Same computation as DiscoForcing's, accumulated over chunks of clips to bound memory."""
    m = mask.double()
    D = phi.shape[2]
    bar = torch.zeros(phi.shape[1], D, dtype=torch.float64)
    for s in range(0, len(phi), chunk):
        bar += (phi[s:s + chunk].double() * m[s:s + chunk, :, None]).sum(0)
    bar /= m.sum(0).clamp(min=1)[:, None]
    C = torch.zeros(phi.shape[1], phi.shape[1], dtype=torch.float64)
    for s in range(0, len(phi), chunk):
        tilde = (phi[s:s + chunk].double() - bar) * m[s:s + chunk, :, None]
        C += torch.einsum("ntd,nsd->ts", tilde, tilde)
    return C / ((m.T @ m).clamp(min=1) * D)


def select_tmr(temporal, mask, budget, rng, stop="fw", **_):
    """TMR, paper Algorithm 1 (OMP variant): temporal operator of the uncentred trajectories phi_i, rank M by the PPCA
    BIC (n_eff = N D), whitened modal atoms gamma_i = Lambda^{-1/2} V^T phi_i, simplex-constrained OMP towards the
    mean atom; it stops early when no atom lowers the modal error (stop='fw')."""
    phi = (temporal * mask[..., None]).float()
    N, T, D = phi.shape
    lam_, U = tmr_select.eig_desc(temporal_operator_masked(phi, mask).numpy())
    Z, info = tmr_select.modal_dictionary(phi.numpy(), "temporal", ops={"lam": lam_, "U": U})
    idx, w, hist = tmr_select.omp_simplex(Z, budget, stop=stop, trace=True)
    M = info["M"]
    print(f"[tmr] T={T} D={D}: M={M} modes by PPCA BIC ({lam_[:M].sum() / lam_.sum():.1%} of temporal variance); "
          f"OMP picked {len(idx)}/{budget} clips, modal error {hist[0]:.3f} -> {hist[-1]:.3f}")
    if len(idx) < budget:
        print(f"[tmr] stopped early: no further clip lowers the modal error (Algorithm 1); buffer has {len(idx)} clips")
    return idx.tolist(), (np.asarray(w) * len(idx)).tolist(), None, {"M": M, "T": T, "D": D, "modal_error": hist}


def balanced_blocks(blocks, mask):
    """Concatenate trajectory blocks (N, T, D_b) along channels, each scaled to unit mean squared deviation per valid
    token so every block carries the same weight in the temporal operator."""
    m = mask.float()
    out = []
    for g in blocks:
        g = g.float()
        bar = (g * m[..., None]).sum(0) / m.sum(0).clamp(min=1)[:, None]
        dev = (((g - bar) ** 2).sum(-1) * m).sum() / m.sum().clamp(min=1)
        out.append(g / dev.sqrt().clamp(min=1e-12))
    return torch.cat(out, -1) * m[..., None]


def channel_covariance(phi, mask, chunk=64):
    """torch.cov of every valid (clip, time) row of phi (D x D), accumulated over chunks of clips."""
    D = phi.shape[2]
    s1, s2, n = torch.zeros(D, dtype=torch.float64), torch.zeros(D, D, dtype=torch.float64), 0
    for s in range(0, len(phi), chunk):
        x = phi[s:s + chunk][mask[s:s + chunk].bool()].double()
        s1 += x.sum(0)
        s2 += x.T @ x
        n += len(x)
    mu = s1 / n
    return (s2 - n * torch.outer(mu, mu)) / (n - 1)


def effective_samples(Cc, N, cond_groups=None, n_cond=0):
    """Satterthwaite effective number of independent temporal vectors among the N x D channel trajectories that the
    temporal operator averages: n_eff = (tr K)^2 / ||K||_F^2, K their correlation (DiscoForcing; only logged for
    tmr_joint, whose rank uses N x D). Returns (n_eff, #ordered same-condition pairs of distinct clips)."""
    tr2, f = float(torch.trace(Cc)) ** 2, float((Cc ** 2).sum())
    pairs = 0.0
    if cond_groups is not None and n_cond > 0:
        _, n_g = np.unique(np.asarray(cond_groups), return_counts=True)
        pairs = float((n_g * (n_g - 1)).sum())
    f_c = float((Cc[:n_cond, :n_cond] ** 2).sum()) if pairs else 0.0
    return N * tr2 / (f + pairs / N * f_c), pairs


def condition_groups(temporal, mask, eps, device="cpu"):
    """Clips whose music trajectories coincide within eps (relative to the mean trajectory norm), with the largest
    distance within and the smallest between groups (DiscoForcing condition_groups + condition_gap, one distance
    matrix for both)."""
    flat = (temporal * mask[..., None]).reshape(len(temporal), -1).to(device, torch.float32)
    d = (torch.cdist(flat, flat) / flat.norm(dim=1).mean().clamp(min=1e-12)).double().cpu()
    del flat
    parent = list(range(len(d)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = torch.nonzero(torch.triu(d < eps, 1), as_tuple=True)
    for i, j in zip(ii.tolist(), jj.tolist()):
        parent[find(i)] = find(j)
    gid = [find(i) for i in range(len(d))]
    g = torch.tensor(gid)
    same = (g[:, None] == g[None]) & ~torch.eye(len(g), dtype=torch.bool)
    within = float(d[same].max()) if same.any() else 0.0
    between = float(d[g[:, None] != g[None]].min()) if (g[:, None] != g[None]).any() else float("inf")
    return gid, within, between


def select_tmr_dance(blocks, mask, groups, budget, rng, per_dance=1, rank="energy", energy=0.9, weights="uniform",
                     stop="fw", label="tmr_dance", cond_groups=None, fill="kcenter", **_):
    """DiscoForcing select_tmr_dance (the function behind tmr_joint) with moments=1: TMR core on balanced trajectory
    blocks, at most `per_dance` clips per song (0 = no limit), k-center fill to the budget if OMP stops early.
    rank: energy | bic_clips | bic_dims (N_eff = #clips x channels, tmr_joint) | bic_ess | bic_neff."""
    phi = balanced_blocks(blocks, mask)
    N, T, D = phi.shape
    lam_, U = tmr_select.eig_desc(temporal_operator_masked(phi, mask).numpy())
    Cc = channel_covariance(phi, mask)
    d_eff = float(torch.trace(Cc) ** 2 / (Cc ** 2).sum())
    n_eff, pairs = effective_samples(Cc, N, cond_groups, blocks[0].shape[-1])
    M_clips, M_dims = tmr_select.bic_rank(lam_, N)[0], tmr_select.bic_rank(lam_, N * D)[0]
    M_ess, M_neff = tmr_select.bic_rank(lam_, N * d_eff)[0], tmr_select.bic_rank(lam_, n_eff)[0]
    M_energy = int(np.searchsorted(np.cumsum(lam_) / lam_.sum(), energy) + 1)
    M = {"energy": M_energy, "bic_clips": M_clips, "bic_dims": M_dims, "bic_ess": M_ess, "bic_neff": M_neff}[rank]
    Z, _ = tmr_select.modal_dictionary(phi.numpy(), "temporal", ops={"lam": lam_, "U": U}, M=M)
    del phi
    groups = np.asarray(groups)
    quotas = {int(g): per_dance for g in np.unique(groups)} if per_dance > 0 else None
    idx, w, hist = tmr_select.omp_simplex(Z, budget, groups if quotas else None, quotas, stop=stop, trace=True)
    n_omp = len(idx)
    if n_omp < budget and fill == "pursuit":  # the same objective at a fixed budget: the first n_omp picks are unchanged
        idx, w, hist = tmr_select.omp_simplex(Z, budget, groups if quotas else None, quotas, stop="none", trace=True)
    idx = [int(i) for i in idx]
    modal_w = dict(zip(idx, (np.asarray(w) * max(len(idx), 1)).tolist()))
    # modal and operator errors of the buffer (DiscoForcing computes them on [Z / sqrt(D), vech(S_i) / sqrt(2)])
    Gm = Z.reshape(N, M, D)
    bar = Gm.mean(0, dtype=np.float64).astype(np.float32)
    iu = np.triu_indices(M)
    scale = (np.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)) / math.sqrt(2.0)).astype(np.float32)
    second = np.empty((N, len(iu[0])), np.float32)
    for i in range(N):
        g = Gm[i] - bar
        second[i] = (g @ g.T / D)[iu] * scale
    wv = np.asarray(w, np.float64)
    eps_mod = np.linalg.norm(Z.mean(0, dtype=np.float64) - wv @ Z[idx].astype(np.float64))
    eps_C = np.linalg.norm(second.mean(0, dtype=np.float64) - wv @ second[idx].astype(np.float64)) * math.sqrt(2.0)
    del second, Gm
    if len(idx) < budget:  # k-center fill: the clip farthest (in modal space) from the buffer, from an unused song
        def dist(s, chunk=256):  # DiscoForcing's ||Zd - Zd[s]|| in float64, over chunks of rows to bound memory
            zs = Z[s].astype(np.float64)
            return np.concatenate([np.linalg.norm(Z[c:c + chunk].astype(np.float64) - zs, axis=1)
                                   for c in range(0, N, chunk)])
        dmin = np.full(N, np.inf)
        for s in idx:
            dmin = np.minimum(dmin, dist(s))
        count = {}
        for i in idx:
            count[groups[i]] = count.get(groups[i], 0) + 1
        while len(idx) < budget:
            ok = np.array([i not in modal_w and (per_dance <= 0 or count.get(groups[i], 0) < per_dance) for i in range(N)])
            if not ok.any():
                break
            j = int(np.argmax(np.where(ok, dmin if idx else 0.0, -np.inf)))
            idx.append(j)
            modal_w[j] = 0.0
            count[groups[j]] = count.get(groups[j], 0) + 1
            dmin = np.minimum(dmin, dist(j))
    wts = [1.0] * len(idx) if weights == "uniform" else [max(modal_w[i], 1e-3) for i in idx]
    s = sum(wts)
    wts = [v / s * len(wts) for v in wts]
    n_fill = len(idx) - n_omp
    print(f"[{label}] T={T} channels={D} (blocks {[b.shape[-1] for b in blocks]}): M={M} modes by {rank} "
          f"({energy:.0%} energy: {M_energy}, BIC N_eff=N: {M_clips}, BIC N_eff=N*D: {M_dims}, BIC N_eff=N*D_eff "
          f"(D_eff={d_eff:.1f}): {M_ess}, BIC n_eff={n_eff:.0f} ({pairs:.0f} same-music pairs): {M_neff}; "
          f"{lam_[:M].sum() / lam_.sum():.1%} of temporal variance); mean matched; OMP picked {n_omp}/{budget}"
          + (f", {'k-center' if fill == 'kcenter' else 'pursuit'} fill {n_fill}" if n_fill else "")
          + f"; objective {hist[0]:.3f} -> {hist[-1]:.3f}; modal error {eps_mod:.3f}, operator error {eps_C:.3f}; "
          f"{len(set(groups[idx].tolist()))} songs; weights {weights} "
          f"(effective size {sum(wts) ** 2 / sum(v * v for v in wts):.1f} of {len(wts)})")
    score = torch.full((N,), float("nan"))
    for i in idx:
        score[i] = modal_w[i]  # each clip's "score": its OMP (simplex) weight, 0 for k-center fill
    return idx, wts, score, {"M": M, "T": T, "D": D, "n_omp": n_omp, "n_fill": n_fill, "modal_error": eps_mod,
                             "operator_error": eps_C, "objective": hist}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['random', 'tmr', 'tmr_joint'])
    ap.add_argument('--session', type=int, required=True, help='session whose buffer is built (1, 2, ...)')
    ap.add_argument('--manifest', required=True, help="that session's training manifest (cl/finedance_sessions.py)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--lm_state', default=None, help='LM after this session (needed for tmr / tmr_joint)')
    ap.add_argument('--ckpt', default='pretrained/full.ckpt')
    ap.add_argument('--window', type=float, default=5.0, help='clip length = the training window (seconds)')
    ap.add_argument('--cand_cache', default=None, help='default outputs/token_cache/<s>_cand_w<window>.pt')
    ap.add_argument('--ratio', type=float, default=0.10, help="buffer size as a fraction of the session's clips")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tmr_stop', choices=['fw', 'positive'], default='fw', help='OMP stopping rule (tmr_select)')
    ap.add_argument('--mix_eps', type=float, default=0.02, help='tmr_joint log: music windows closer than this are one')
    ap.add_argument('--music_layer', type=int, default=0, help='0 = the LM music input embedding (DiscoForcing analogue)')
    ap.add_argument('--pool', type=int, default=1, help='average-pool trajectories in time (1 = the 50 Hz token rate)')
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed * 1000 + a.session)
    torch.manual_seed(a.seed)

    model = None

    def get_model():
        nonlocal model
        if model is None:
            from unimumo.models import UniMuMo
            model = UniMuMo.from_checkpoint(a.ckpt)
            if a.lm_state:
                load_lm_state(model, a.lm_state)
            model.music_motion_lm.eval()
        return model

    cand_path = a.cand_cache or os.path.join('outputs/token_cache', f"{Path(a.manifest).stem.split('_')[0]}_cand_w{a.window:g}.pt")
    cands = candidate_cache(get_model, a.manifest, a.window, cand_path)
    N = len(cands['ids'])
    budget = max(1, round(a.ratio * N))
    print(f'session {a.session}: {N} candidate clips from {a.manifest}, budget {budget} ({a.ratio:.0%})')

    info = {}
    if a.method == 'random':
        idx, wts, score = select_random(N, budget, rng)
    else:
        if not a.lm_state:
            ap.error(f'--lm_state (the model after session {a.session}) is required for {a.method}')
        m = get_model()
        music = music_trajectories(m.music_motion_lm, cands['music_code'], a.music_layer, a.pool)
        mask = torch.ones(music.shape[:2], dtype=torch.bool)
        print(f'music trajectories {tuple(music.shape)} (layer {a.music_layer}, pool {a.pool}) ({time.time() - t0:.0f}s)')
        if a.method == 'tmr':
            idx, wts, score, info = select_tmr(music, mask, budget, rng, stop=a.tmr_stop)
        else:
            dance = dance_latents(m, cands['motion_code'], a.pool)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cond, within, between = condition_groups(music, mask, a.mix_eps, device)
            print(f'[tmr_joint] music windows: {N} clips -> {len(set(cond))} distinct (same window: relative distance '
                  f'<= {within:.2e}; different windows: >= {between:.3f}; eps {a.mix_eps})')
            T = min(music.shape[1], dance.shape[1])
            song_id = {n: i for i, n in enumerate(sorted(set(cands['songs'])))}
            idx, wts, score, info = select_tmr_dance(
                [music[:, :T], dance[:, :T]], mask[:, :T], [song_id[n] for n in cands['songs']], budget, rng,
                per_dance=0, rank='bic_dims', weights='modal', stop=a.tmr_stop, label='tmr_joint', cond_groups=cond,
                fill='kcenter')

    wts = wts or [1.0] * len(idx)
    it = torch.as_tensor(idx, dtype=torch.long)
    buf = {'music_code': cands['music_code'][it].clone(), 'motion_code': cands['motion_code'][it].clone(),
           'ids': [cands['ids'][i] for i in idx], 'songs': [cands['songs'][i] for i in idx],
           'weights': torch.tensor(wts, dtype=torch.float), 'indices': it, 'session': a.session, 'method': a.method,
           'window': a.window, 'ratio': a.ratio, 'n_candidates': N, 'budget': budget}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    torch.save(buf, a.out)
    clips = [dict({'id': cands['ids'][i], 'weight': float(w)},
                  **({'score': float(score[i])} if score is not None and not torch.isnan(score[i]) else {}))
             for i, w in zip(idx, wts)]
    meta = {'method': a.method, 'session': a.session, 'manifest': a.manifest, 'lm_state': a.lm_state,
            'ratio': a.ratio, 'n_candidates': N, 'budget': budget, 'n_clips': len(idx),
            'n_songs': len(set(buf['songs'])), 'music_layer': a.music_layer, 'pool': a.pool, 'seed': a.seed,
            'tmr_stop': a.tmr_stop, 'info': {k: v for k, v in info.items() if k != 'objective'},
            'seconds': round(time.time() - t0, 1), 'clips': clips}
    json.dump(meta, open(os.path.splitext(a.out)[0] + '.json', 'w'), indent=1, default=float)
    w = np.array(wts)
    print(f'saved {a.out}: {len(idx)} clips from {len(set(buf["songs"]))} songs ({len(idx) / N:.1%} of the session); '
          f'weights mean {w.mean():.3f} min {w.min():.3f} max {w.max():.3f}; {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
