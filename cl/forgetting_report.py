"""Decide forgetting from cl/eval_dance.py results, with held-out loss from cl/eval_nll.py as an extra row.

    python cl/forgetting_report.py --noise outputs/dance_eval/noise.json \
        --s0 outputs/dance_eval/s0.json --gt outputs/dance_eval/gt.json --recon outputs/dance_eval/recon.json \
        --method vanilla=outputs/dance_eval/after_s1.json,outputs/dance_eval/vanilla_after_s2.json \
        --nll_s0 outputs/nll/fd_s0.json --nll vanilla=outputs/nll/fd_after_s1.json,outputs/nll/fd_vanilla_after_s2.json
(one comma-separated list per method: the result after S1, after S2[, after S3]; --nll lists are matched by name)

For every method, metric and old session i (each session before the last trained one, T):
  change      = m[i][T] - m[i][i], signed so that > 0 means worse
  noise       = the metric's split-half std on session i's real test dance (noise.json from `eval_dance.py build`):
                how much the number moves just from which test songs are used
  SNR         = change / noise  ->  FORGETTING if >= 3, likely if >= 2, otherwise within noise
  of complete = change as a share of the gap between m[i][i] and 'complete forgetting' (another session's real
                dance scored as session i's music: the model dancing the wrong style)
  held-out loss: change with a paired 95% CI over the same test windows (eval_nll's *_per_window.npz files)
Quality metrics (Div_k, Div_g, BAS, Foot Skating Ratio) are shown with the real-dance values, without a verdict.
Also prints F_avg / A_avg per metric (TMR paper, Sec. 4: F_i = mean_{t>i} change).
"""
import argparse
import json
import os

import numpy as np

SESS = ['s1', 's2', 's3']
FORGET = [('StyleRetention', True), ('StyleAcc', True), ('DanceRate', True), ('StyleFD', False),
          ('FID_k', False), ('FID_g', False)]
QUALITY = [('Div_k', None), ('Div_g', None), ('BAS', True), ('FSR', False)]


def results(path):
    return json.load(open(path))['results']


def val(res, s, k):
    v = res.get(f'{s}_test', {}).get(k)
    return float('nan') if v is None else float(v)


FLOOR = ('FID_k', 'FID_g', 'StyleFD')  # distances to real dance: their real-dance reference is the split-half floor


def num(v, w=9, d=3):
    if np.isnan(v):
        return f'{"-":>{w}s}'
    return f'{v:{w}.{d}f}' if abs(v) < 10 ** (w - d - 3) else f'{v:{w}.2e}'


def fmt(v, untrained=False, w=12):
    if np.isnan(v):
        return f'{"-":>{w}s}'
    s = f'{v:.4f}' if abs(v) < 1e5 else f'{v:.3e}'
    return f'{f"({s})":>{w}s}' if untrained else f'{s:>{w}s}'


def print_matrix(name, higher, s0, stages, extra, noise=None):
    T = len(stages)
    arrow = {True: 'higher is better', False: 'lower is better', None: 'closest to real dance is best'}[higher]
    print(f'\n  {name} ({arrow}); rows: test music of session i, columns: model after session t, ( ) = not trained on yet')
    labels = [('real floor' if name in FLOOR and lab == 'real dance' else lab) for lab, _ in extra]
    cols = (['S0'] if s0 else []) + [f'after {SESS[t]}' for t in range(T)] + labels
    print('        ' + ''.join(f'{c:>12s}' for c in cols))
    for i, s in enumerate(SESS):
        cells = ([fmt(val(s0, s, name), True)] if s0 else []) + [fmt(val(st, s, name), i > t) for t, st in enumerate(stages)]
        for lab, r in extra:
            if name in FLOOR and lab == 'real dance':  # FID of real dance against itself is 0 by construction
                cells.append(fmt((noise or {}).get(s, {}).get(f'{name}_halfset_mean', float('nan'))))
            else:
                cells.append(fmt(val(r, s, name)))
        if any(c.strip() != '-' for c in cells):
            print(f'    {s:4s}' + ''.join(cells))


def verdict(change, snr, metric):
    if change <= 0:
        return 'no forgetting' if metric != 'DanceRate' else 'no change'
    if np.isnan(snr):
        return '?'
    if metric == 'DanceRate':  # dancing another style is still dance: this flags 'stopped dancing', not forgetting
        return 'LESS DANCE-LIKE' if snr >= 3 else 'within noise'
    return 'FORGETTING' if snr >= 3 else 'likely' if snr >= 2 else 'within noise'


def paired_nll(path_a, path_b, s):
    """Mean and SE of the per-window change in held-out motion loss on session s's test windows (b - a)."""
    pa = path_a.replace('.json', f'_{s}_test_per_window.npz')
    pb = path_b.replace('.json', f'_{s}_test_per_window.npz')
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return float('nan'), float('nan')
    a, b = np.load(pa), np.load(pb)
    assert (a['ids'] == b['ids']).all(), 'per-window files cover different windows'
    d = b['motion_ce'] - a['motion_ce']
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise', required=True, help='noise.json from cl/eval_dance.py build')
    ap.add_argument('--method', action='append', required=True, help='name=after_s1.json,after_s2.json[,...]')
    ap.add_argument('--s0', default=None, help='eval_dance result of the pretrained model')
    ap.add_argument('--gt', default=None, help='eval_dance --gt result (real dance)')
    ap.add_argument('--recon', default=None, help='eval_dance --recon result (tokenizer ceiling)')
    ap.add_argument('--nll_s0', default=None, help='eval_nll result of the pretrained model')
    ap.add_argument('--nll', action='append', default=[], help='name=eval_nll after_s1.json,after_s2.json[,...]')
    a = ap.parse_args()

    noise = json.load(open(a.noise))
    s0 = results(a.s0) if a.s0 else {}
    extra = [(lab, results(p)) for lab, p in (('real dance', a.gt), ('tokenizer', a.recon)) if p]
    nll = {spec.split('=', 1)[0]: spec.split('=', 1)[1].split(',') for spec in a.nll}
    nll_s0 = {k: {'motion_ce': v['motion_ce']} for k, v in results(a.nll_s0).items()} if a.nll_s0 else {}

    for spec in a.method:
        name, paths = spec.split('=', 1)
        stages = [results(p) for p in paths.split(',')]
        T = len(stages)
        print(f'\n{"=" * 100}\n{name}: {T} sessions trained ({" -> ".join(SESS[:T])})\n{"=" * 100}')
        for metric, higher in [('StyleRetention', True), ('FID_k', False), ('FID_g', False)]:
            print_matrix(metric, higher, s0, stages, extra, noise)
        npaths = nll.get(name)
        if npaths:
            nstages = [{k: {'motion_ce': v['motion_ce']} for k, v in results(p).items()} for p in npaths]
            print_matrix('motion_ce', False, nll_s0, nstages, [])

        print(f'\n  Forgetting of each old session after training up to {SESS[T - 1]} (change > 0 = worse; noise = '
              'split-half std on real dance)')
        print(f'    {"session":8s}{"metric":16s}{"S0":>9s}{"own":>9s}{"last":>9s}{"change":>9s}{"noise":>8s}'
              f'{"SNR":>7s}{"of complete":>13s}   verdict')
        for i in range(T - 1):
            s = SESS[i]
            for metric, higher in FORGET:
                mii, miT = val(stages[i], s, metric), val(stages[T - 1], s, metric)
                if np.isnan(mii) or np.isnan(miT):
                    continue
                change = (mii - miT) if higher else (miT - mii)
                nz = noise.get(s, {}).get(f'{metric}_halfset_std', float('nan'))
                snr = (change / nz if nz > 0 else np.inf * np.sign(change)) if not np.isnan(nz) else float('nan')
                comp = noise.get(s, {}).get(f'{metric}_complete_forgetting', float('nan'))
                gap = (mii - comp) if higher else (comp - mii)
                share = change / gap if gap > 1e-9 and metric != 'DanceRate' else float('nan')
                sh = f'{100 * share:12.0f}%' if not np.isnan(share) else f'{"-":>13s}'
                print(f'    {s:8s}{metric:16s}{num(val(s0, s, metric))}{num(mii)}{num(miT)}{change:+9.3f}{nz:8.3f}'
                      f'{snr:7.1f}{sh}   {verdict(change, snr, metric)}')
            if npaths and len(npaths) == T:
                d, se = paired_nll(npaths[i], npaths[T - 1], s)
                mii = val({k: v for k, v in results(npaths[i]).items()}, s, 'motion_ce')
                miT = val({k: v for k, v in results(npaths[T - 1]).items()}, s, 'motion_ce')
                ci = f'95% CI {d - 1.96 * se:+.3f}..{d + 1.96 * se:+.3f}' if not np.isnan(se) else 'no per-window files'
                v = '?' if np.isnan(se) else ('FORGETTING' if d - 1.96 * se > 0 else 'within noise')
                print(f'    {s:8s}{"held-out loss":16s}{num(val(nll_s0, s, "motion_ce"))}{num(mii)}{num(miT)}{miT - mii:+9.3f}'
                      f'{se:8.3f}{(d / se if se else float("nan")):7.1f}{"-":>13s}   {v} ({ci})')
        print('    (held-out loss: noise = standard error of the paired change over the same test windows; '
              'DanceRate: a drop means less dance-like output, not a style change)')

        print('\n  Quality of the dance (real dance and tokenizer ceiling for reference; no forgetting verdict)')
        for metric, higher in QUALITY:
            print_matrix(metric, higher, s0, stages, extra)

        print('\n  Summary over the trained sessions (TMR paper: F_i = mean over later sessions of the change; '
              'A_avg = mean of m[i][i])')
        for metric, higher in FORGET:
            M = np.array([[val(stages[t], SESS[i], metric) for t in range(T)] for i in range(T)])
            if np.isnan(np.diag(M)).any():
                continue
            sgn = 1.0 if higher else -1.0
            F = [np.mean([sgn * (M[i, i] - M[i, t]) for t in range(i + 1, T)]) for i in range(T - 1)]
            print(f'    {metric:16s} F_avg {np.mean(F):+.4f}   A_avg {np.mean(np.diag(M)):.4f}   '
                  f'F_i {" ".join(f"{f:+.4f}" for f in F)}')


if __name__ == '__main__':
    main()
