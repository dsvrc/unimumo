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
    --schedule_from runs/fd/s1                 same session length and warm-up as S1, counted in passes
                                               (epochs) over this session's own windows, so a smaller
                                               session is not trained for more epochs than S1 was
    --replay buffers/fd/<m>/s1.pt [--replay buffers/fd/<m>/s2.pt]   the buffers of the earlier sessions

Replay (DiscoForcing utils/cl_replay.py + train_ldf.py): every step adds, per replay batch, replay_weight x
mean_i(w_i l_i) to the current loss, with l_i the clip's own loss and w_i its buffer weight (TMR / TMR-joint OMP
weights with mean 1, 1 for ER). Buffer clips are drawn without replacement and reshuffled once used up.
  --replay_balance equal   (default) every earlier session gets its own replay batch of --batch_size clips and its
                           own replay_weight-weighted term: the current session and each earlier session weigh the
                           same, in the batch and in the loss (S3: current : S1 : S2 = 1 : 1 : 1)
  --replay_balance session DiscoForcing balance=session: one replay batch of --batch_size with equal clips per
                           earlier session (S3: current : S1 : S2 = 2 : 1 : 1)
  --replay_balance uniform DiscoForcing's default: one replay batch drawn from the union of the buffers (sessions in
                           proportion to their buffer sizes)
With a single earlier session (S2) the three are identical: current : S1 = 1 : 1.
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


def per_sample_ce(logits, targets, mask):
    """Per-clip MusicMotionTransformer.compute_cross_entropy: mean over each codebook's valid tokens, then over
    codebooks. Its batch mean equals the batch value, since every clip has the same valid tokens."""
    B, K, T = targets.shape
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                           reduction='none').view(B, K, T)
    m = mask.float()
    return ((ce * m).sum(-1) / m.sum(-1).clamp(min=1)).mean(1)


def lm_loss_per_sample(mm, music, motion, motion_weight, use_bf16):
    cond = mm.prepare_text_condition(['<separation>'] * music.shape[0])
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        mu_out, mo_out = mm.model.compute_predictions(music, motion, 'music_motion', [], condition_tensors=cond)
    ce_mu = per_sample_ce(mu_out.logits.float(), music, mu_out.mask)
    ce_mo = per_sample_ce(mo_out.logits.float(), motion, mo_out.mask)
    return (1 - motion_weight) * ce_mu + motion_weight * ce_mo, ce_mu.detach(), ce_mo.detach()


class ReplayStream:
    """Replay batches over the buffers of the earlier sessions (DiscoForcing ReplayStream / SessionBalancedBatches).
    equal: one batch of `batch_size` per buffer; session: one batch with equal clips per buffer (remainder slots
    rotate over the buffers); uniform: one shuffled batch from the union (an incomplete last batch of a pass is
    dropped, as DataLoader(shuffle=True, drop_last=True)). Each buffer's clips are drawn without replacement and
    reshuffled once used up."""

    def __init__(self, buffers, batch_size, balance, seed=0):
        self.buffers, self.balance = buffers, balance
        self.rng = np.random.default_rng(seed)
        self.n = [b['music_code'].shape[0] for b in buffers]
        self.pools = [[] for _ in buffers]
        self.offset = 0
        if balance == 'uniform':
            self.union = [(j, i) for j, n in enumerate(self.n) for i in range(n)]
            self.batch_size = min(batch_size, len(self.union))
            self.upool = []
        elif balance == 'session':
            self.batch_size = min(batch_size, sum(self.n))
        else:
            self.batch_size = [min(batch_size, n) for n in self.n]

    def _pop(self, j):
        if not self.pools[j]:
            self.pools[j] = self.rng.permutation(self.n[j]).tolist()
        return self.pools[j].pop()

    def next(self):
        """[(name, [(buffer, clip), ...]), ...]: one entry per replay term of this step."""
        if self.balance == 'equal':
            return [(f's{self.buffers[j]["session"]}', [(j, self._pop(j)) for _ in range(self.batch_size[j])])
                    for j in range(len(self.buffers))]
        if self.balance == 'session':
            k = len(self.buffers)
            batch = [((self.offset + s) % k, None) for s in range(self.batch_size)]
            self.offset = (self.offset + self.batch_size) % k
            return [('replay', [(j, self._pop(j)) for j, _ in batch])]
        if len(self.upool) < self.batch_size:
            self.upool = [self.union[i] for i in self.rng.permutation(len(self.union))]
        batch, self.upool = self.upool[:self.batch_size], self.upool[self.batch_size:]
        return [('replay', batch)]

    def tensors(self, picks, use_weights):
        mu = torch.stack([self.buffers[j]['music_code'][i] for j, i in picks])
        mo = torch.stack([self.buffers[j]['motion_code'][i] for j, i in picks])
        w = torch.tensor([float(self.buffers[j]['weights'][i]) if use_weights else 1.0 for j, i in picks])
        return mu, mo, w


def sample(cache, n, gen):
    N = cache['music_code'].shape[0]
    idx = torch.randint(0, N, (n,), generator=gen)
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


def schedule_from(run_dir):
    """Best length and warm-up of a finished --save_best run, in epochs over that run's training windows."""
    a = json.load(open(pjoin(run_dir, 'args.json')))
    b = json.load(open(pjoin(run_dir, 'best.json')))
    assert b['step'] > 0, f'{run_dir}: the best step is 0 (the run never improved)'
    n = b.get('n_train')
    if n is None:  # runs from before n_train was recorded: count the windows in its token cache
        n = torch.load(a['train_cache'] or pjoin(run_dir, 'train_tokens.pt'), weights_only=False)['music_code'].shape[0]
    bs = a['batch_size']
    return {'epochs': b['step'] * bs / n, 'warmup_epochs': a['warmup'] * bs / n, 'n_train': n, 'step': b['step']}


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
    ap.add_argument('--replay', action='append', default=[],
                    help='replay buffer of an earlier session (cl/make_buffer.py); one per earlier session')
    ap.add_argument('--replay_weight', type=float, default=1.0, help='lambda of every replay term (DiscoForcing 1.0)')
    ap.add_argument('--replay_balance', choices=['equal', 'session', 'uniform'], default='equal')
    ap.add_argument('--no_sample_weights', action='store_true', help='ignore the buffer weights (all 1)')
    # eval
    ap.add_argument('--eval', action='append', default=[], help='name=token_cache.pt (repeatable)')
    ap.add_argument('--eval_every', type=int, default=500)
    # optimisation
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--epochs', type=float, default=None,
                    help='session length in passes over the training windows (overrides --steps)')
    ap.add_argument('--schedule_from', default=None,
                    help='run dir of a finished --save_best session (e.g. runs/fd/s1): take its best length and '
                         'its warm-up in epochs (overrides --steps/--warmup unless --epochs/--warmup_epochs given)')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--warmup_epochs', type=float, default=None, help='warm-up in epochs (overrides --warmup)')
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
    args.train_cache = args.train_cache or pjoin(args.run_dir, 'train_tokens.pt')
    train_cache = load_or_build_cache(model, args.train_cache, items_fn, args.window, args.hop, args.batch_size)
    n_train = train_cache['music_code'].shape[0]
    if args.schedule_from:
        ref = schedule_from(args.schedule_from)
        print(f"schedule from {args.schedule_from}: best step {ref['step']} = {ref['epochs']:.2f} epochs over "
              f"{ref['n_train']} windows, warm-up {ref['warmup_epochs']:.2f} epochs")
        args.epochs = args.epochs if args.epochs is not None else ref['epochs']
        args.warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else ref['warmup_epochs']
    if args.epochs is not None:
        args.steps = max(1, round(args.epochs * n_train / args.batch_size))
    if args.warmup_epochs is not None:
        args.warmup = max(1, round(args.warmup_epochs * n_train / args.batch_size))
    args.n_train = n_train
    with open(pjoin(args.run_dir, 'args.json'), 'w') as f:  # again, with the resolved steps / warm-up
        json.dump(vars(args), f, indent=2)
    ep = lambda step: step * args.batch_size / n_train
    print(f'train windows: {n_train} | session: {args.steps} steps = {ep(args.steps):.2f} epochs, '
          f'warm-up {args.warmup} steps = {ep(args.warmup):.2f} epochs')
    replay = None
    if args.replay:
        buffers = []
        for p in args.replay:
            c = torch.load(p, weights_only=False)
            c['weights'] = torch.as_tensor(c.get('weights', torch.ones(c['music_code'].shape[0])), dtype=torch.float)
            c.setdefault('session', len(buffers) + 1)
            w = c['weights'].double()
            print(f"replay buffer S{c['session']} {p}: {len(w)} clips ({c.get('method', '?')}), weights mean "
                  f"{w.mean():.3f} min {w.min():.3f} max {w.max():.3f}, effective size {w.sum() ** 2 / (w ** 2).sum():.1f}")
            assert c['music_code'].shape[1:] == train_cache['music_code'].shape[1:], f'{p}: clip length differs from training'
            buffers.append(c)
        replay = ReplayStream(buffers, args.batch_size, args.replay_balance, seed=args.seed)
        if args.replay_balance == 'equal':
            mix = ' + '.join(f"{b} x S{c['session']}" for b, c in zip(replay.batch_size, buffers))
            names = [f"s{c['session']}" for c in buffers]
        else:
            mix = f"{replay.batch_size} from {'+'.join('S%d' % c['session'] for c in buffers)} ({args.replay_balance})"
            names = ['replay']
        terms = ' + '.join(f"{args.replay_weight} x mean(w l)_{n}" for n in names)
        print(f'each step: {args.batch_size} current clips + {mix} replay clips; loss = L_current + {terms}'
              f"{' (buffer weights ignored)' if args.no_sample_weights else ''}")
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
            best.update(step=step, motion_ce=v, epochs=ep(step))
            if step > 0:
                torch.save(mm.state_dict(), pjoin(args.run_dir, 'lm.pt'))
            with open(pjoin(args.run_dir, 'best.json'), 'w') as f:
                json.dump({'eval': args.save_best, 'n_train': n_train, 'batch_size': args.batch_size, **best}, f)
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
        opt.zero_grad(set_to_none=True)
        loss, ce_mu, ce_mo = lm_loss(mm, mu.to(dev), mo.to(dev), args.motion_weight, use_bf16)
        rec = {'step': step, 'loss': loss.item(), 'ce_music': ce_mu.item(), 'ce_motion': ce_mo.item()}
        loss.backward()  # each term is backpropagated on its own: same gradient as their sum, one graph in memory
        total = loss.item()
        if replay is not None:
            for name, picks in replay.next():
                rmu, rmo, w = replay.tensors(picks, not args.no_sample_weights)
                per, _, rce_mo = lm_loss_per_sample(mm, rmu.to(dev), rmo.to(dev), args.motion_weight, use_bf16)
                term = args.replay_weight * (w.to(dev) * per).mean()
                term.backward()
                total += term.item()
                rec[f'replay_{name}'] = term.item()
                rec[f'replay_{name}_ce_motion'] = rce_mo.mean().item()
            rec['loss_total'] = total
        if not math.isfinite(total):
            sys.exit(f'non-finite loss at step {step}')
        rec['grad_norm'] = torch.nn.utils.clip_grad_norm_(trainable, args.max_norm).item()
        opt.step()
        sched.step()
        rec['lr'] = sched.get_last_lr()[0]
        if step % 50 == 0:
            write(rec)
            rep = ' '.join(f'{k[7:]} {v:.4f}' for k, v in rec.items()
                           if k.startswith('replay_') and not k.endswith('_ce_motion'))
            print(f'step {step} loss {rec["loss"]:.4f} ce_mu {rec["ce_music"]:.4f} ce_mo {rec["ce_motion"]:.4f} '
                  + (f'| replay {rep} | total {rec["loss_total"]:.4f} ' if rep else '')
                  + f'gn {rec["grad_norm"]:.2f} | {(time.time() - t0) / step:.2f}s/step')
        if eval_caches and (step % args.eval_every == 0 or step == args.steps):
            res = evaluate(mm, eval_caches, args.batch_size, use_bf16)
            write({'step': step, 'epochs': ep(step), 'eval': res})
            for k, v in res.items():
                print(f'[step {step}] ({ep(step):.2f} ep) {k}: motion CE {v["motion_ce"]:.4f} | '
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
