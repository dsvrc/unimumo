"""Fine-tune UniMuMo's music-motion LM for one continual session (optionally with replay).

Objective = UniMuMo's own loss: (1 - w) * CE_music + w * CE_motion  (w = 0.15 by default),
joint 'music_motion' mode, null text (audio-only conditioning).
Trainable: the LM (embeddings, transformer, motion FFNs, heads, T5 output projection).
Frozen: T5 encoder, caption-stage self-attention, captioner, Encodec, motion VQ-VAE.

Every --eval_every steps it reports, for each eval cache: motion CE (matched music), motion CE
with music shuffled across windows, and audio reliance d_audio = shuffled - matched.

G0 pilot (real AIST++ pairs -> does the model start using the music?):
    python cl/train_session.py --run_dir runs/g0_aist \
        --train_aist_split data/motion/aist_train.txt --train_motions data/motion/train/joint_vecs \
        --wavs /path/to/EDGE/edge_aistpp/wavs \
        --eval aist_val=outputs/token_cache/aist_val_w5_h5.pt --steps 2000 --eval_every 250

Later sessions:
    --init_lm_state runs/<method>/s1/lm.pt     start from the previous session
    --replay buffer.pt                         token cache of replayed windows (optional 'weights')
"""
import argparse
import json
import math
import os
import sys
import time
from os.path import join as pjoin
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from eval_nll import aist_manifest, read_manifest, tokenize_set, eval_set, shuffle_music, load_lm_state


def setup_trainable(mm):
    for p in mm.parameters():
        p.requires_grad = False
    trainable = []
    for n, p in mm.model.named_parameters():
        if 'captioning_self_attn' in n:
            continue
        if n.startswith('condition_provider') and 'output_proj' not in n:
            continue  # frozen T5 encoder
        p.requires_grad = True
        trainable.append(p)
    print(f'trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M')
    return trainable


def lm_loss(mm, music, motion, motion_weight, use_bf16):
    cond = mm.prepare_text_condition(['<separation>'] * music.shape[0])
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        mu_out, mo_out = mm.model.compute_predictions(music, motion, 'music_motion', [], condition_tensors=cond)
    ce_mu, _ = mm.compute_cross_entropy(mu_out.logits.float(), music, mu_out.mask)
    ce_mo, _ = mm.compute_cross_entropy(mo_out.logits.float(), motion, mo_out.mask)
    return (1 - motion_weight) * ce_mu + motion_weight * ce_mo, ce_mu.detach(), ce_mo.detach()


def sample(cache, n, gen, weights=None):
    N = cache['music_code'].shape[0]
    if weights is None:
        idx = torch.randint(0, N, (n,), generator=gen)
    else:
        idx = torch.multinomial(weights, n, replacement=True, generator=gen)
    return cache['music_code'][idx], cache['motion_code'][idx]


@torch.no_grad()
def evaluate(mm, eval_caches, batch_size, use_bf16):
    mm.eval()
    out = {}
    for name, cache in eval_caches.items():
        matched, pw_m = eval_set(mm, cache, batch_size, use_bf16)
        shuffled, pw_s = eval_set(mm, shuffle_music(cache), batch_size, use_bf16)
        d = pw_s['motion_ce'] - pw_m['motion_ce']
        out[name] = {
            'motion_ce': matched['motion_ce'], 'music_ce': matched['music_ce'],
            'motion_ce_shuffled': shuffled['motion_ce'],
            'd_audio': float(d.mean()), 'd_audio_sem': float(d.std(ddof=1) / np.sqrt(len(d))),
            'd_audio_frac_pos': float((d > 0).mean()),
            'motion_ce_per_codebook': matched['motion_ce_per_codebook'],
        }
    mm.train()
    mm.text_model.eval()
    return out


def load_or_build_cache(model, path, items_fn, window, hop, batch_size):
    if path and os.path.exists(path):
        print(f'using token cache {path}')
        return torch.load(path)
    cache = tokenize_set(model, items_fn(), window, hop, batch_size)
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(cache, path)
        print(f'saved token cache {path}')
    return cache


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--ckpt', default='pretrained/full.ckpt')
    ap.add_argument('--init_lm_state', default=None, help='LM weights of the previous session')
    # training data: an AIST++ split or a JSONL manifest; tokenized once into --train_cache
    ap.add_argument('--train_aist_split', default=None)
    ap.add_argument('--train_motions', default='data/motion/train/joint_vecs')
    ap.add_argument('--wavs', default='../EDGE/edge_aistpp/wavs')
    ap.add_argument('--train_music_exclude', default=None,
                    help="AIST++ only: 'heldout0' (one track per genre) or comma list of track ids to hold out")
    ap.add_argument('--train_manifest', default=None)
    ap.add_argument('--train_cache', default=None, help='default: <run_dir>/train_tokens.pt')
    ap.add_argument('--window', type=float, default=5.0)
    ap.add_argument('--hop', type=float, default=2.5)
    # replay
    ap.add_argument('--replay', action='append', default=[], help='token cache of replay windows (repeatable)')
    ap.add_argument('--replay_lambda', type=float, default=1.0)
    # eval
    ap.add_argument('--eval', action='append', default=[], help='name=token_cache.pt (repeatable)')
    ap.add_argument('--eval_every', type=int, default=500)
    # optimisation
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--motion_weight', type=float, default=0.15)
    ap.add_argument('--max_norm', type=float, default=1.0)
    ap.add_argument('--no_bf16', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_save', action='store_true')
    ap.add_argument('--save_best', default=None,
                    help='eval set name: save lm.pt whenever its motion CE improves (instead of the last step)')
    args = ap.parse_args()
    use_bf16 = not args.no_bf16

    os.makedirs(args.run_dir, exist_ok=True)
    with open(pjoin(args.run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    from unimumo.models import UniMuMo
    model = UniMuMo.from_checkpoint(args.ckpt)
    if args.init_lm_state:
        load_lm_state(model, args.init_lm_state)
    mm = model.music_motion_lm

    # data
    if args.train_aist_split:
        items_fn = lambda: aist_manifest(args.train_aist_split, args.train_motions, args.wavs,
                                         exclude_music=args.train_music_exclude)
    else:
        assert args.train_manifest, 'give --train_aist_split or --train_manifest'
        items_fn = lambda: read_manifest(args.train_manifest)
    train_cache = load_or_build_cache(model, args.train_cache or pjoin(args.run_dir, 'train_tokens.pt'),
                                      items_fn, args.window, args.hop, args.batch_size)
    print(f'train windows: {train_cache["music_code"].shape[0]}')
    replay = []
    for p in args.replay:
        c = torch.load(p)
        w = c.get('weights')
        replay.append((c, None if w is None else torch.as_tensor(w, dtype=torch.float)))
        print(f'replay buffer {p}: {c["music_code"].shape[0]} windows, weighted={w is not None}')
    eval_caches = {}
    for spec in args.eval:
        name, p = spec.split('=', 1)
        eval_caches[name] = torch.load(p)

    trainable = setup_trainable(mm)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

    log_path = pjoin(args.run_dir, 'log.jsonl')
    log = open(log_path, 'a')

    def write(rec):
        log.write(json.dumps(rec) + '\n')
        log.flush()

    best = {'step': 0, 'motion_ce': float('inf')}
    if args.save_best:
        assert args.save_best in eval_caches, f'--save_best {args.save_best} is not one of the --eval names'

    def maybe_save_best(step, res):
        if not args.save_best:
            return
        v = res[args.save_best]['motion_ce']
        if v < best['motion_ce']:
            best.update(step=step, motion_ce=v)
            if step > 0:
                torch.save(mm.state_dict(), pjoin(args.run_dir, 'lm.pt'))
            with open(pjoin(args.run_dir, 'best.json'), 'w') as f:
                json.dump({'eval': args.save_best, **best}, f)
            print(f'  new best {args.save_best} motion CE {v:.4f} at step {step}' + (' (saved lm.pt)' if step else ''))

    if eval_caches:
        res = evaluate(mm, eval_caches, args.batch_size, use_bf16)
        write({'step': 0, 'eval': res})
        for k, v in res.items():
            print(f'[step 0] {k}: motion CE {v["motion_ce"]:.4f} | d_audio {v["d_audio"]:+.4f} ± {v["d_audio_sem"]:.4f}')
        maybe_save_best(0, res)

    mm.train()
    mm.text_model.eval()
    dev = mm.device
    t0 = time.time()
    for step in range(1, args.steps + 1):
        mu, mo = sample(train_cache, args.batch_size, gen)
        loss, ce_mu, ce_mo = lm_loss(mm, mu.to(dev), mo.to(dev), args.motion_weight, use_bf16)
        rec = {'step': step, 'loss': loss.item(), 'ce_music': ce_mu.item(), 'ce_motion': ce_mo.item()}
        for i, (rc, rw) in enumerate(replay):
            rmu, rmo = sample(rc, args.batch_size, gen, rw)
            rloss, _, rce_mo = lm_loss(mm, rmu.to(dev), rmo.to(dev), args.motion_weight, use_bf16)
            loss = loss + args.replay_lambda * rloss
            rec[f'replay{i}_ce_motion'] = rce_mo.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not math.isfinite(loss.item()):
            sys.exit(f'non-finite loss at step {step}')
        rec['grad_norm'] = torch.nn.utils.clip_grad_norm_(trainable, args.max_norm).item()
        opt.step()
        sched.step()
        rec['lr'] = sched.get_last_lr()[0]
        if step % 50 == 0:
            write(rec)
            print(f'step {step} loss {rec["loss"]:.4f} ce_mu {rec["ce_music"]:.4f} ce_mo {rec["ce_motion"]:.4f} '
                  f'gn {rec["grad_norm"]:.2f} | {(time.time() - t0) / step:.2f}s/step')
        if eval_caches and (step % args.eval_every == 0 or step == args.steps):
            res = evaluate(mm, eval_caches, args.batch_size, use_bf16)
            write({'step': step, 'eval': res})
            for k, v in res.items():
                print(f'[step {step}] {k}: motion CE {v["motion_ce"]:.4f} | '
                      f'd_audio {v["d_audio"]:+.4f} ± {v["d_audio_sem"]:.4f} (frac>0 {v["d_audio_frac_pos"]:.2f})')
            maybe_save_best(step, res)

    if args.save_best:
        if best['step'] == 0:
            print(f'WARNING: {args.save_best} never improved on the pretrained model; no lm.pt saved')
        else:
            print(f'best {args.save_best} motion CE {best["motion_ce"]:.4f} at step {best["step"]} '
                  f'-> {pjoin(args.run_dir, "lm.pt")}')
    elif not args.no_save:
        torch.save(mm.state_dict(), pjoin(args.run_dir, 'lm.pt'))
        print(f'saved {pjoin(args.run_dir, "lm.pt")}')
    log.close()
