"""Forgetting matrix and F_avg / A_avg (TMR paper, Sec. 4) from cl/eval_nll.py result files.

m[i][t] = held-out motion CE on session i's test set after training session t (lower is better).
F_i = mean_{t>i} (m[i][t] - m[i][i]);  F_avg = mean over old sessions;  A_avg = mean_i m[i][i].

    python cl/forgetting_table.py --s0 outputs/nll/fd_s0.json \
        --method vanilla=outputs/nll/after_s1.json,outputs/nll/vanilla_after_s2.json \
        --method er=outputs/nll/after_s1.json,outputs/nll/er_after_s2.json
(one comma-separated list per method: the result file after S1, after S2, [after S3])
"""
import argparse
import json

import numpy as np

SESSIONS = ['s1', 's2', 's3']


def load(path, key='motion_ce'):
    res = json.load(open(path))['results']
    return {s: res[f'{s}_test'][key] for s in SESSIONS if f'{s}_test' in res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s0', default=None, help='pretrained-model results (reference row)')
    ap.add_argument('--method', action='append', required=True, help='name=after_s1.json,after_s2.json[,...]')
    ap.add_argument('--key', default='motion_ce', help="motion_ce (eval_nll) or StyleRetention / StyleAcc (eval_style)")
    ap.add_argument('--higher_is_better', action='store_true', help='set for StyleRetention / StyleAcc')
    args = ap.parse_args()
    sgn = -1.0 if args.higher_is_better else 1.0      # F > 0 always means forgetting

    if args.s0:
        s0 = load(args.s0, args.key)
        print('S0 (pretrained) test ' + args.key + ': ' + '  '.join(f'{s}={v:.4f}' for s, v in s0.items()))
    summary = []
    for spec in args.method:
        name, paths = spec.split('=', 1)
        stages = [load(p, args.key) for p in paths.split(',')]
        T = len(stages)
        m = np.full((T, T), np.nan)
        for t, st in enumerate(stages):
            for i in range(t + 1):
                m[i, t] = st[SESSIONS[i]]
        print(f'\n[{name}] {args.key} on test sets (rows: session i, columns: after training session t)')
        print('        ' + ''.join(f'after {SESSIONS[t]:>4s} ' for t in range(T)))
        for i in range(T):
            print(f'  {SESSIONS[i]:4s}  ' + ''.join(f'{m[i, t]:10.4f} ' if not np.isnan(m[i, t]) else f'{"":10s} '
                                                     for t in range(T)))
        F = [np.mean([sgn * (m[i, t] - m[i, i]) for t in range(i + 1, T)]) for i in range(T - 1)]
        F_avg = float(np.mean(F)) if F else float('nan')
        A_avg = float(np.mean([m[i, i] for i in range(T)]))
        summary.append((name, F_avg, A_avg, F))
    better = 'higher' if args.higher_is_better else 'lower'
    print(f'\n{"method":10s} {"F_avg (lower=less forgetting)":>30s} {f"A_avg ({better}=better learning)":>30s}  per-session F')
    for name, F_avg, A_avg, F in summary:
        print(f'{name:10s} {F_avg:30.4f} {A_avg:30.4f}  ' + ' '.join(f'{f:+.4f}' for f in F))


if __name__ == '__main__':
    main()
