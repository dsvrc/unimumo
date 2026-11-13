"""Forgetting matrix and F_avg / A_avg (TMR paper, Sec. 4) from cl/eval_nll.py or cl/eval_style.py results.

m[i][t] = metric on session i's test set after training session t.
F_i = mean_{t>i} (m[i][t] - m[i][i]), sign flipped for higher-is-better metrics, so F > 0 always means forgetting.
F_avg = mean over old sessions;  A_avg = mean_i m[i][i].
With --s0 the table also shows the pretrained model, and for every old session the share of its own gain
(S0 -> m[i][i]) that is lost again by the last session (100% = back to the pretrained level).

    python cl/forgetting_table.py --s0 outputs/nll/fd_s0.json \
        --method vanilla=outputs/nll/fd_after_s1.json,outputs/nll/fd_vanilla_after_s2.json
    python cl/forgetting_table.py --key StyleRetention --higher_is_better --s0 outputs/style/fd_s0.json \
        --method vanilla=outputs/style/fd_after_s1.json,outputs/style/fd_vanilla_after_s2.json
(one comma-separated list per method: the result file after S1, after S2, [after S3])
Cells in ( ) are sessions the model has not been trained on yet.
"""
import argparse
import json

import numpy as np

SESSIONS = ['s1', 's2', 's3']


def load(path, key='motion_ce'):
    res = json.load(open(path))['results']
    return {s: res[f'{s}_test'][key] for s in SESSIONS if f'{s}_test' in res}


def cell(v, untrained):
    if np.isnan(v):
        return f'{"":>12s}'
    return f'{f"({v:.4f})":>12s}' if untrained else f'{v:11.4f} '


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s0', default=None, help='pretrained-model results (reference column)')
    ap.add_argument('--method', action='append', required=True, help='name=after_s1.json,after_s2.json[,...]')
    ap.add_argument('--key', default='motion_ce', help="motion_ce (eval_nll) or StyleRetention / StyleAcc (eval_style)")
    ap.add_argument('--higher_is_better', action='store_true', help='set for StyleRetention / StyleAcc')
    args = ap.parse_args()
    sgn = -1.0 if args.higher_is_better else 1.0

    s0 = load(args.s0, args.key) if args.s0 else {}
    summary = []
    for spec in args.method:
        name, paths = spec.split('=', 1)
        stages = [load(p, args.key) for p in paths.split(',')]
        T = len(stages)
        m = np.array([[st.get(s, np.nan) for st in stages] for s in SESSIONS])          # (sessions, T)
        z = np.array([s0.get(s, np.nan) for s in SESSIONS])
        print(f'\n[{name}] {args.key} on the test sets (rows: test set of session i, columns: model after '
              f'session t, ( ) = not trained on yet)')
        print('      ' + (f'{"S0":>12s}' if s0 else '') + ''.join(f'{"after " + SESSIONS[t]:>12s}' for t in range(T)))
        for i, s in enumerate(SESSIONS):
            if np.isnan(z[i]) and np.isnan(m[i]).all():
                continue
            print(f'  {s:4s}' + (cell(z[i], True) if s0 else '') + ''.join(cell(m[i, t], i > t) for t in range(T)))
        F = [float(np.mean([sgn * (m[i, t] - m[i, i]) for t in range(i + 1, T)])) for i in range(T - 1)]
        F_avg = float(np.mean(F)) if F else float('nan')
        A_avg = float(np.mean([m[i, i] for i in range(T)]))
        lost = []
        for i in range(T - 1):
            gain = m[i, i] - z[i]
            lost.append((m[i, i] - m[i, T - 1]) / gain if abs(gain) > 1e-9 else float('nan'))
        summary.append((name, F_avg, A_avg, F, lost))
    better = 'higher' if args.higher_is_better else 'lower'
    print(f'\n{"method":18s} {"F_avg (lower = less forgetting)":>32s} {f"A_avg ({better} = better)":>24s}'
          f'  per-session F' + ('   share of the session gain lost' if s0 else ''))
    for name, F_avg, A_avg, F, lost in summary:
        print(f'{name:18s} {F_avg:32.4f} {A_avg:24.4f}  ' + ' '.join(f'{f:+.4f}' for f in F)
              + ('   ' + ' '.join(f'{SESSIONS[i]} {100 * l:.0f}%' for i, l in enumerate(lost)) if s0 else ''))


if __name__ == '__main__':
    main()
