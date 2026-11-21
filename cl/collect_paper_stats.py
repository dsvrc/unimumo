"""Collect, on the server, the UniMuMo facts and numbers that the TMR supplementary still has in red, from the files
the runs have already written. It trains, generates and selects nothing, and needs no GPU (numpy only; torch is not
imported).

Run from the repo root on the server, after the runs:
    python cl/collect_paper_stats.py
    python cl/collect_paper_stats.py --methods "vanilla mir_kd csrel_kd gcr_kd tmr_joint_neff_kd"   # fewer rows
Writes outputs/paper_stats/ with
    summary.md   one section per paper item (below), as tables ready to read
    stats.json   everything summary.md shows, plus the raw values behind it
    raw/         copies of the small files it read (json, logs, split lists; no .pt / .npz)
and outputs/paper_stats_unimumo.tar.gz (the same folder) to copy back.

Code sessions s1 Street, s2 Folk, s3 Classic are the paper's S0, S1, S2 (UniMuMo on FineDance, Sec. app:unimumo).
Paper rows of Tab. tab:unimumo: Fine-tuning = vanilla, MIR = mir_kd, CSReL = csrel_kd, GCR = gcr_kd,
TMR = tmr_joint_neff_kd.

What it collects, and the place in sec/X_suppl.tex it is for:
  A. Tab. instantiation, UniMuMo row (phi, T x D, when selected), and the N_eff, start / stopping rule and selection
     time that the main paper says this table lists        <- buffers/fd/<m>/s*.json, logs/buffer_<m>_s*.log
  B. Sec. app:unimumo, sessions and splits: songs, sub-genres and hours per split, training windows, replay
     candidates and budget B, test clips and windows        <- data/finedance/manifests, runs/fd, buffers, outputs
  C. Sec. app:unimumo, training: optimizer settings, session length, warm-up, batch and replay mix, distillation
     weights, time per step and per session, GPU           <- runs/fd/**/args.json, best.json, log.jsonl, logs/
  D. TMR buffers per session: rank M with each choice of N_eff, clips picked by OMP before the fill, objective,
     modal error, effective buffer size (the questions asked of Audio2Photoreal, and Sec. app:algorithm 'state
     whether the runs used Algorithm 1 as written')         <- buffer json + log
  E. FAQ (v): every method starts from the same Session-0 model, with the same seed and current-session data order;
     restarted runs                                          <- args.json, log.jsonl
  F. Tab. baseline_hp, 'state where other benchmarks differ': the baselines' settings on UniMuMo
  G. Style classifier on FineDance (validation report) and which metric can see forgetting on FineDance (SNR of
     StyleRetention, StyleFD, FID_k, FID_g with UniMuMo's frozen normalisation; Sec. app:sr text)
  H. Cross-check of Tab. tab:unimumo from the raw result files, per-genre SR and final dance quality of every
     finished method, held-out loss                          <- outputs/dance_eval, outputs/nll
Seeds / variance and the user study are not collected here.
"""
import argparse
import json
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except ImportError:  # hours per split are then left out
    np = None

SESS = [('s1', 's1_street', 'Street'), ('s2', 's2_folk', 'Folk'), ('s3', 's3_classic', 'Classic')]
PAPER = {'s1': 'S0', 's2': 'S1', 's3': 'S2'}
PAPER_ROWS = [('Fine-tuning', 'vanilla'), ('MIR', 'mir_kd'), ('CSReL', 'csrel_kd'), ('GCR', 'gcr_kd'),
              ('TMR', 'tmr_joint_neff_kd')]
ALL_METHODS = ('vanilla random random_star ocs gcr mir csrel tmr tmr_joint tmr_joint_neff random_kd ocs_kd gcr_kd '
               'mir_kd csrel_kd tmr_kd tmr_joint_kd tmr_joint_neff_kd').split()
DANCE_METRICS = ['StyleRetention', 'StyleAcc', 'DanceRate', 'StyleFD', 'FID_k', 'FID_g', 'Div_k', 'Div_g', 'BAS',
                 'FSR']
TRAIN_KEYS = ['init_lm_state', 'train_manifest', 'train_cache', 'n_train', 'window', 'hop', 'steps', 'epochs',
              'warmup', 'warmup_epochs', 'batch_size', 'lr', 'weight_decay', 'motion_weight', 'max_norm', 'no_bf16',
              'seed', 'save_best', 'schedule_from', 'eval_every', 'replay', 'replay_weight', 'replay_balance',
              'no_sample_weights', 'distill', 'kd_weight', 'gt_weight', 'star', 'star_lam', 'star_gamma', 'star_eps',
              'star_steps']

ROOT = Path('.')
OUT = Path('.')
MISSING = []   # expected files that were not found
ERRORS = []    # collectors that failed, with the reason
COPIED = set()


# ---------------------------------------------------------------- small helpers
def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def exists(p, note=True):
    p = Path(p)
    if p.exists():
        return True
    if note:
        MISSING.append(rel(p))
    return False


def copy_raw(p):
    p = Path(p)
    if not p.is_file() or p.suffix in ('.pt', '.npz', '.npy', '.ckpt') or rel(p) in COPIED:
        return
    r = rel(p)
    dst = OUT / 'raw' / (r if not Path(r).is_absolute() else Path('outside_repo') / p.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    COPIED.add(rel(p))


def load_json(p, note=True):
    if not exists(p, note):
        return None
    copy_raw(p)
    with open(p) as f:
        return json.load(f)


def read_text(p, note=True):
    if not exists(p, note):
        return ''
    copy_raw(p)
    with open(p, errors='replace') as f:
        return f.read()


def grab(pattern, text, cast=str, group=1, last=False):
    m = list(re.finditer(pattern, text, re.M))
    if not m:
        return None
    m = m[-1] if last else m[0]
    try:
        return cast(m.group(group))
    except (TypeError, ValueError):
        return None


def sh(cmd):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001 - a missing tool is only reported
        return f'unavailable ({e.__class__.__name__})'


def fmt(v, nd=3):
    if v is None:
        return '-'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, float):
        if v != v:
            return 'nan'
        if v != 0 and (abs(v) >= 1e5 or abs(v) < 10 ** -nd):
            return f'{v:.3e}'
        return f'{v:.{nd}f}'
    if isinstance(v, (list, tuple)):
        return ', '.join(fmt(x, nd) for x in v)
    return str(v)


def table(header, rows, nd=3):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(fmt(c, nd) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def guarded(name):
    def deco(fn):
        def run(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:  # noqa: BLE001 - one failing collector must not stop the others
                ERRORS.append(f'{name}: {e.__class__.__name__}: {e}')
                return {}
        return run
    return deco


# ---------------------------------------------------------------- environment
@guarded('environment')
def environment():
    out = {'host': socket.gethostname(), 'date': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'root': str(ROOT.resolve()),
           'python': sys.version.split()[0],
           'git_head': sh(['git', 'rev-parse', 'HEAD']),
           'git_last_cl_commit': sh(['git', 'log', '-1', '--format=%h %cd %s', '--', 'cl']),
           'git_uncommitted_in_cl': sh(['git', 'status', '--porcelain', '--', 'cl']),
           'gpus_now': sh(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])}
    gpus = Counter()
    for p in sorted((ROOT / 'logs').glob('replay_*.log')):
        for m in re.finditer(r' GPU \d+, ([^,]+), (\d+) MiB, (\d+) MiB', read_text(p, note=False)):
            gpus[f'{m.group(1).strip()} ({int(m.group(3)) // 1024} GiB)'] += 1
    out['gpus_in_replay_logs'] = dict(gpus)
    return out


# ---------------------------------------------------------------- B. sessions and splits
@guarded('splits')
def splits(runs, buffers, noise, nll_s0):
    man = ROOT / 'data/finedance/manifests'
    out = {}
    for s, name, fam in SESS:
        d = {'family': fam, 'paper_session': PAPER[s]}
        for split in ('train', 'val', 'test'):
            p = man / f'{name}_{split}.jsonl'
            if not exists(p):
                continue
            copy_raw(p)
            items = [json.loads(line) for line in open(p) if line.strip()]
            secs = []
            if np is not None:
                for it in items:
                    mp = Path(it['motion'])
                    mp = mp if mp.is_absolute() else ROOT / mp
                    try:
                        secs.append(np.load(mp, mmap_mode='r').shape[0] / 60.0)  # joint_vecs are 60 fps
                    except Exception:  # noqa: BLE001
                        MISSING.append(rel(mp))
            d[split] = {'songs': len(items), 'sub_genres': dict(sorted(Counter(it.get('genre') for it in items).items())),
                        'hours': round(sum(secs) / 3600, 3) if secs else None,
                        'song_seconds_min_median_max': ([round(min(secs), 1), round(statistics.median(secs), 1),
                                                         round(max(secs), 1)] if secs else None),
                        'ids': [it['id'] for it in items]}
            copy_raw(ROOT / 'data/finedance/splits' / f'{name}_{split}.txt')
        # training windows (5 s, hop 2.5 s) of the run that trained on this session
        tr = runs.get('s1') if s == 's1' else next((runs[k] for k in (f'vanilla/{s}', f'tmr_joint_neff_kd/{s}')
                                                    if k in runs), None)
        d['train_windows'] = (tr or {}).get('args', {}).get('n_train')
        d['train_window_s'] = (tr or {}).get('args', {}).get('window')
        d['train_hop_s'] = (tr or {}).get('args', {}).get('hop')
        # replay candidates and budget (any method's buffer of this session; they all tile the same clips)
        cand = sorted({(b['n_candidates'], b['budget']) for b in buffers.values()
                       if b.get('session_code') == s and b.get('n_candidates')})
        d['replay_candidates_budget'] = cand
        # test set of the dance metrics (10 s clips; 5 s windows, 1 s stride) and of the held-out loss
        n = (noise or {}).get(s, {})
        d['test_dance'] = {k: n.get(k) for k in ('n_songs', 'n_clips', 'n_windows')}
        res = (nll_s0 or {}).get('results', {})
        d['nll_windows'] = {k: res.get(f'{s}_{k}', {}).get('n_windows') for k in ('val', 'test')}
        out[s] = d
    return out


# ---------------------------------------------------------------- C / E. training runs
def train_log_path(run):
    parts = run.relative_to(ROOT / 'runs/fd').parts
    if parts == ('s1',):
        return ROOT / 'logs/fd_s1.log'
    return ROOT / f'logs/fd_{parts[0]}_{parts[1]}.log'


@guarded('runs')
def training_runs():
    R = ROOT / 'runs/fd'
    dirs = ([R / 's1'] if (R / 's1').exists() else []) + sorted(p.parent for p in R.glob('*/s[0-9]/args.json'))
    if not dirs:
        MISSING.append(rel(R))
    out = {}
    for d in dirs:
        key = str(d.relative_to(R)).replace('\\', '/')
        a = load_json(d / 'args.json') or {}
        rec = {'args': {k: a.get(k) for k in TRAIN_KEYS if k in a}, 'best': load_json(d / 'best.json', note=False)}
        # log.jsonl: every start of a run appends to it from step 0 (train_session.py has no resume: a restarted
        # run trains again from the start with the same seed)
        steps, starts, last_eval = [], 0, None
        lj = d / 'log.jsonl'
        if exists(lj):
            for line in open(lj):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get('step') == 0:
                    starts += 1
                if 'eval' in r:
                    last_eval = r
                steps.append(r.get('step', 0))
        rec['starts_in_log'] = starts
        rec['last_logged_step'] = max(steps) if steps else None
        rec['final_eval_motion_ce'] = ({k: round(v['motion_ce'], 4) for k, v in last_eval['eval'].items()}
                                       if last_eval else None)
        rec['final_eval_d_audio'] = ({k: round(v['d_audio'], 5) for k, v in last_eval['eval'].items()}
                                     if last_eval else None)
        # wall clock: from the second write of args.json (after the token cache) to the last log line
        if (d / 'args.json').exists() and lj.exists():
            rec['wall_clock_min_approx'] = round((lj.stat().st_mtime - (d / 'args.json').stat().st_mtime) / 60, 1)
        txt = read_text(train_log_path(d), note=True)
        rec['log'] = {
            'trainable_params_M': grab(r'^trainable params: ([\d.]+)M', txt, float),
            'train_line': grab(r'^(train windows: .*)$', txt),
            'schedule_line': grab(r'^(schedule from .*)$', txt),
            'replay_lines': re.findall(r'^(replay buffer S\d .*)$', txt, re.M),
            'each_step_line': grab(r'^(each step: .*)$', txt),
            'distill_line': grab(r'^(distillation replay: .*)$', txt),
            'star_line': grab(r'^(STAR: .*)$', txt),
            'best_line': grab(r'^(best \S+ motion CE .*)$', txt, last=True),
            'sec_per_step': grab(r'\| ([\d.]+)s/step', txt, float, last=True),
        }
        st = rec['args'].get('steps')
        if st and rec['log']['sec_per_step']:
            rec['train_minutes_from_sec_per_step'] = round(st * rec['log']['sec_per_step'] / 60, 1)
        out[key] = rec
    return out


# ---------------------------------------------------------------- A / D / F. buffers
def parse_buffer_log(txt):
    d = {}
    line = grab(r'^(\[tmr_joint(?:_neff)?\] T=.*)$', txt)
    if line:
        d['tmr_line'] = line
        d['T'] = grab(r'T=(\d+)', line, int)
        d['channels'] = grab(r'channels=(\d+)', line, int)
        d['blocks'] = grab(r'\(blocks \[([\d, ]+)\]\)', line)
        d['M'] = grab(r'M=(\d+) modes', line, int)
        d['rank_rule'] = grab(r'modes by (\w+)', line)
        d['M_energy90'] = grab(r'% energy: (\d+)', line, int)
        d['M_bic_N'] = grab(r'BIC N_eff=N: (\d+)', line, int)
        d['M_bic_ND'] = grab(r'BIC N_eff=N\*D: (\d+)', line, int)
        d['D_eff'] = grab(r'D_eff=([\d.]+)\)', line, float)
        d['M_bic_ND_eff'] = grab(r'D_eff=[\d.]+\): (\d+)', line, int)
        d['n_eff'] = grab(r'BIC n_eff=(\d+)', line, int)
        d['same_music_pairs'] = grab(r'\((\d+) same-music pairs\)', line, int)
        d['M_bic_neff'] = grab(r'same-music pairs\): (\d+)', line, int)
        d['variance_kept_pct'] = grab(r'; ([\d.]+)% of temporal variance', line, float)
        d['omp_picked'] = grab(r'OMP picked (\d+)/\d+', line, int)
        d['fill'] = grab(r'(k-center|pursuit) fill (\d+)', line)
        d['n_fill'] = grab(r'(?:k-center|pursuit) fill (\d+)', line, int) or 0
        d['objective_start'] = grab(r'objective ([\d.]+) ->', line, float)
        d['objective_end'] = grab(r'objective [\d.]+ -> ([\d.]+)', line, float)
        d['modal_error'] = grab(r'modal error ([\d.]+)', line, float)
        d['operator_error'] = grab(r'operator error ([\d.]+)', line, float)
        d['effective_size'] = grab(r'effective size ([\d.]+) of', line, float)
    line = grab(r'^(\[tmr\] T=.*)$', txt)
    if line:
        d['tmr_line'] = line
        d['T'] = grab(r'T=(\d+)', line, int)
        d['channels'] = grab(r' D=(\d+)', line, int)
        d['M'] = grab(r'M=(\d+) modes', line, int)
        d['variance_kept_pct'] = grab(r'\(([\d.]+)% of temporal variance', line, float)
        d['omp_picked'] = grab(r'OMP picked (\d+)/\d+', line, int)
        d['objective_start'] = grab(r'modal error ([\d.]+) ->', line, float)
        d['objective_end'] = grab(r'modal error [\d.]+ -> ([\d.]+)', line, float)
        d['stopped_early'] = 'stopped early' in txt
    d['music_windows_line'] = grab(r'^(\[tmr_joint\S*\] music windows: .*)$', txt)
    d['distinct_music_windows'] = grab(r'music windows: \d+ clips -> (\d+) distinct', txt, int)
    d['trajectory_seconds'] = grab(r'^music trajectories .*\((\d+)s\)$', txt, int)
    d['trajectory_shape'] = grab(r'^music trajectories \(([\d, ]+)\)', txt)
    d['gcr_line'] = grab(r'^(\[gcr\] .*)$', txt)
    d['mir_lines'] = re.findall(r'^(\[mir\] .*)$', txt, re.M)
    d['csrel_lines'] = re.findall(r'^(\[csrel\] .*)$', txt, re.M)
    d['saved_line'] = grab(r'^(saved .*)$', txt, last=True)
    return {k: v for k, v in d.items() if v not in (None, [], '')}


@guarded('buffers')
def buffer_files():
    B = ROOT / 'buffers/fd'
    if not B.exists():
        MISSING.append(rel(B))
        return {}
    out = {}
    for j in sorted(B.glob('*/s[0-9].json')):
        m, s = j.parent.name, j.stem
        b = load_json(j)
        w = [c['weight'] for c in b.get('clips', [])]
        ids = [c['id'] for c in b.get('clips', [])]
        rec = {k: b.get(k) for k in ('method', 'session', 'lm_state', 'ratio', 'n_candidates', 'budget', 'n_clips',
                                     'n_songs', 'music_layer', 'pool', 'seed', 'tmr_stop', 'seconds', 'info', 'params')}
        rec['dir'], rec['session_code'] = m, s
        if w:
            rec['weights'] = {'mean': sum(w) / len(w), 'min': min(w), 'max': max(w),
                              'effective_size': sum(w) ** 2 / sum(x * x for x in w),
                              'n_below_1pct_of_mean': sum(1 for x in w if x < 0.01 * sum(w) / len(w))}
        rec['_ids'] = ids
        # its log; an S1 buffer copied from another method with the same selection has the log under that method
        log = ROOT / f'logs/buffer_{m}_{s}.log'
        src = m
        if not log.exists():
            for other in sorted(ROOT.glob(f'logs/buffer_*_{s}.log')):
                o = other.name[len('buffer_'):-len(f'_{s}.log')]
                oj = B / o / f'{s}.json'
                if oj.exists() and [c['id'] for c in json.load(open(oj)).get('clips', [])] == ids:
                    log, src = other, o
                    break
        rec['log_from'] = src if log.exists() else None
        if log.exists():
            rec['log'] = parse_buffer_log(read_text(log))
        else:
            MISSING.append(rel(ROOT / f'logs/buffer_{m}_{s}.log'))
        out[f'{m}/{s}'] = rec
    return out


def overlaps(buffers, dirs, s='s1'):
    """Share of the row buffer's clips that the column buffer also holds (S1 buffers share the S1 model)."""
    ids = {d: set(buffers[f'{d}/{s}']['_ids']) for d in dirs if f'{d}/{s}' in buffers}
    names = list(ids)
    return names, [[len(ids[a] & ids[b]) / max(len(ids[a]), 1) for b in names] for a in names]


# ---------------------------------------------------------------- G. style encoder, metric SNR
@guarded('style_encoder')
def style_encoder():
    rep = load_json(ROOT / 'pretrained/style_encoder_finedance.json')
    txt = read_text(ROOT / 'logs/style_encoder.log', note=False)
    out = {'report': rep}
    out['log_test_tables'] = re.findall(r'(test songs, [^\n]*\n(?:[^\n]+\n)+)', txt)
    out['log_criteria'] = grab(r'^(clean: .*)$', txt)
    out['log_probes'] = re.findall(r'^(\s+\S.*max session mass.*)$', txt, re.M)
    out['log_split_lines'] = re.findall(r'^((?:train|val|test)\s*: \d+ songs.*)$', txt, re.M)
    out['log_temperature'] = grab(r'^temperature ([\d.]+)', txt, float)
    out['log_babel'] = grab(r'^(BABEL negatives .*)$', txt)
    return out


@guarded('metric_snr')
def metric_snr(noise):
    if not noise:
        return {}
    per, agg = {}, {}
    for s, _, fam in SESS:
        n = noise.get(s)
        if not n:
            continue
        r = {'family': fam, 'n_songs': n.get('n_songs'), 'n_clips': n.get('n_clips'), 'n_windows': n.get('n_windows')}
        if 'StyleRetention_complete_forgetting' in n:
            r['StyleRetention_gt'] = n['StyleRetention_gt']
            r['StyleRetention_noise'] = n['StyleRetention_halfset_std']
            r['StyleRetention_complete'] = n['StyleRetention_complete_forgetting']
            r['SNR_StyleRetention'] = ((n['StyleRetention_gt'] - n['StyleRetention_complete_forgetting'])
                                       / max(n['StyleRetention_halfset_std'], 1e-9))
        for k in ('StyleFD', 'FID_k', 'FID_g'):
            if f'{k}_complete_forgetting' in n:
                r[f'{k}_floor'] = n[f'{k}_halfset_mean']
                r[f'{k}_noise'] = n[f'{k}_halfset_std']
                r[f'{k}_complete'] = n[f'{k}_complete_forgetting']
                r[f'SNR_{k}'] = (n[f'{k}_complete_forgetting'] - n[f'{k}_halfset_mean']) / max(n[f'{k}_halfset_std'], 1e-9)
        for k in ('Div_k', 'Div_g', 'BAS', 'FSR', 'DanceRate'):
            r[f'{k}_real'] = n.get(f'{k}_gt')
        per[s] = r
    for k in ('StyleRetention', 'StyleFD', 'FID_k', 'FID_g'):
        v = [r[f'SNR_{k}'] for r in per.values() if f'SNR_{k}' in r]
        if v:
            agg[k] = {'min': min(v), 'median': statistics.median(v), 'per_session': v}
    meta = noise.get('meta', {})
    return {'per_session': per, 'min_median_over_sessions': agg,
            'meta': {k: meta.get(k) for k in ('clip_seconds', 'max_clips_per_song', 'halves', 'geo_features_kept')}}


# ---------------------------------------------------------------- H. dance results and held-out loss
def sr(res, s):
    v = (res or {}).get('results', {}).get(f'{s}_test', {}).get('StyleRetention')
    return None if v is None else float(v)


def mean_of(*v):
    return None if any(x is None for x in v) else sum(v) / len(v)


@guarded('dance_results')
def dance_results(methods):
    E = ROOT / 'outputs/dance_eval'
    ref = {k: load_json(E / f'{k}.json', note=(k != 'recon')) for k in ('gt', 'recon', 's0', 'after_s1')}
    for t in ('report_replay.txt', 'report.txt'):
        copy_raw(E / t)
    found = sorted({p.name[:-len('_after_s3.json')] for p in E.glob('*_after_s3.json')})
    order = [m for m in methods if m in found] + [m for m in found if m not in methods]
    out = {'references': {}, 'methods': {}, 'meta': {}}
    for k, r in ref.items():
        if r:
            out['references'][k] = {s: {m: r['results'].get(f'{s}_test', {}).get(m) for m in DANCE_METRICS}
                                    for s, _, _ in SESS}
            out['meta'][k] = r.get('meta')
    a1 = ref.get('after_s1')
    for m in order:
        r2, r3 = load_json(E / f'{m}_after_s2.json'), load_json(E / f'{m}_after_s3.json')
        if not (r2 and r3 and a1):
            continue
        earlier = [sr(r2, 's1'), sr(r3, 's1'), sr(r3, 's2')]
        cells = {'after_S0_street': sr(a1, 's1'),
                 'S1_earlier': earlier[0],                          # Street after Folk
                 'S2_earlier': mean_of(earlier[1], earlier[2]),     # Street and Folk after Classic
                 'mean_earlier': mean_of(*earlier),
                 'new': mean_of(sr(r2, 's2'), sr(r3, 's3')),       # Folk after Folk, Classic after Classic
                 'street_final': earlier[1], 'folk_final': earlier[2]}
        out['methods'][m] = {
            'tab_unimumo_cells': cells,
            'sr_matrix': {'after_s1': {'s1': sr(a1, 's1')}, 'after_s2': {s: sr(r2, s) for s in ('s1', 's2')},
                          'after_s3': {s: sr(r3, s) for s in ('s1', 's2', 's3')}},
            'final_quality': {s: {k: r3['results'].get(f'{s}_test', {}).get(k) for k in DANCE_METRICS}
                              for s, _, _ in SESS},
            'class_mass_final': {s: r3['results'].get(f'{s}_test', {}).get('class_mass') for s, _, _ in SESS},
            'meta': {'after_s2': r2.get('meta'), 'after_s3': r3.get('meta')}}
    return out


@guarded('nll')
def nll_results():
    out = {}
    for p in sorted((ROOT / 'outputs/nll').glob('fd_*.json')):
        r = load_json(p)
        out[p.stem] = {k: {'motion_ce': v.get('motion_ce'), 'sem': v.get('motion_ce_sem'), 'n_windows': v.get('n_windows')}
                       for k, v in r.get('results', {}).items()}
    return out


@guarded('check_setup')
def check_setup():
    out = {}
    for p in sorted(set(ROOT.glob('logs/*check*setup*.log')) | set(ROOT.glob('logs/check*.log'))):
        txt = read_text(p)
        out[rel(p)] = re.findall(r'^\s*(LM params total.*|caption-stage.*|text conditioner.*|motion-specific.*|'
                                 r'trainable in a session.*|captioner.*|transformer layers.*|LM stage.*)$', txt, re.M)
    return out


# ---------------------------------------------------------------- summary
def write_summary(S, methods):
    L = []
    add = L.append
    env = S.get('environment', {})
    add('# UniMuMo numbers for the TMR supplementary\n')
    add(f"Collected {env.get('date')} on {env.get('host')} from `{env.get('root')}`; git {env.get('git_head')} "
        f"(last change to cl/: {env.get('git_last_cl_commit')}).")
    if env.get('git_uncommitted_in_cl'):
        add(f"\n**Uncommitted changes in cl/ on the server:**\n```\n{env['git_uncommitted_in_cl']}\n```")
    add(f"GPUs in the replay logs: {env.get('gpus_in_replay_logs') or '-'}; GPUs now: {env.get('gpus_now')}.\n")
    add('Code sessions s1 Street, s2 Folk, s3 Classic = paper S0, S1, S2. Paper rows: '
        + ', '.join(f'{p} = `{m}`' for p, m in PAPER_ROWS) + '.\n')

    # A
    B = S.get('buffers', {})
    add('## A. Tab. instantiation, UniMuMo row (and N_eff, start/stop rule, selection time)\n')
    add('From the code (cl/make_buffer.py, cl/tmr_select.py, cl/train_session.py): phi = the LM music input embedding, '
        'the sum of the four codebook embeddings at every 50 Hz token (--music_layer 0), next to the motion '
        "tokenizer's latents of the clip (quantizer.decode, 128 channels), each block scaled to unit mean squared "
        'deviation; rank by the BIC with N_eff = Satterthwaite (tr K)^2/||K||_F^2 (bic_neff); OMP starts at the clip '
        'of largest correlation with the target, stops on the Frank-Wolfe gap (tmr_stop fw), and if it stops before '
        'the budget the same pursuit continues to the budget (fill pursuit); replay weights = simplex weights x |B| '
        '(mean 1, floor 1e-3). Conditioning passed with the motion: the empty text prompt `<separation>`. Replay loss: '
        'distillation (kd rows). Selected once, right after its session, with the model just trained on it.\n')
    rows = []
    for k, b in sorted(B.items()):
        if b.get('method') not in ('tmr', 'tmr_joint', 'tmr_joint_neff'):
            continue
        lg, info = b.get('log', {}), b.get('info') or {}
        tr, tot = lg.get('trajectory_seconds'), b.get('seconds')
        rows.append([k, PAPER.get(b['session_code']), b.get('lm_state'), info.get('T', lg.get('T')),
                     info.get('D', lg.get('channels')), lg.get('blocks'), lg.get('rank_rule'), b.get('tmr_stop'),
                     lg.get('fill'), tr, None if tr is None or tot is None else round(tot - tr, 1), tot,
                     b.get('log_from')])
    add(table(['buffer', 'paper session', 'selected with', 'T', 'D', 'blocks', 'rank rule', 'stop', 'fill',
               'start to trajectories done (s)', 'after trajectories (s)', 'whole build (s)', 'log from'], rows))
    add('\n`whole build` is make_buffer.py end to end (model load, candidate cache, trajectories, selection, save); '
        '`after trajectories` is the rest once the music trajectories exist (dance latents, music-window groups, '
        'operator, BIC, OMP, save). Peak GPU memory of the build is not logged.\n')

    # B
    SP = S.get('splits', {})
    add('## B. Sessions and splits (Sec. app:unimumo; layout of Tabs. sam_split / a2p_split)\n')
    rows = []
    for s, name, fam in SESS:
        d = SP.get(s, {})
        g = lambda sp, k: d.get(sp, {}).get(k)
        rows.append([PAPER[s], fam, g('train', 'songs'), g('val', 'songs'), g('test', 'songs'), g('train', 'hours'),
                     g('val', 'hours'), g('test', 'hours'), d.get('train_windows'),
                     '; '.join(f'{n} / {b}' for n, b in d.get('replay_candidates_budget', [])) or '-',
                     d.get('test_dance', {}).get('n_clips'), d.get('test_dance', {}).get('n_windows'),
                     d.get('nll_windows', {}).get('val'), d.get('nll_windows', {}).get('test')])
    add(table(['Session', 'Family', 'Train songs', 'Val songs', 'Test songs', 'Train h', 'Val h', 'Test h',
               'Train windows (5 s, hop 2.5 s)', 'Candidates / B', 'Test clips (10 s)', 'Test windows (5 s, 1 s)',
               'Val windows (CE)', 'Test windows (CE)'], rows))
    add('\nSub-genres per split:\n')
    for s, name, fam in SESS:
        for sp in ('train', 'val', 'test'):
            sg = SP.get(s, {}).get(sp, {}).get('sub_genres')
            if sg:
                add(f'- {PAPER[s]} {fam} {sp}: {sg}')
    add('')

    # C
    R = S.get('runs', {})
    add('## C. Training (Sec. app:unimumo)\n')
    keys = ['n_train', 'steps', 'epochs', 'warmup', 'warmup_epochs', 'batch_size', 'lr', 'weight_decay',
            'motion_weight', 'max_norm', 'seed', 'replay_balance', 'replay_weight', 'distill', 'gt_weight',
            'kd_weight', 'star']
    rows = []
    for k, r in R.items():
        a, lg = r.get('args', {}), r.get('log', {})
        if k != 's1' and k.split('/')[0] not in methods:
            continue
        rows.append([k] + [a.get(x) for x in keys] + [lg.get('trainable_params_M'), lg.get('sec_per_step'),
                                                      r.get('train_minutes_from_sec_per_step'),
                                                      r.get('wall_clock_min_approx'), r.get('starts_in_log')])
    add(table(['run'] + keys + ['trainable M', 's/step', 'train min', 'wall min', 'starts'], rows, nd=4))
    s1 = R.get('s1', {})
    add(f"\nS0 (Street) length: {(s1.get('best') or {})} ; {s1.get('log', {}).get('best_line')}")
    for k, r in R.items():
        if k.split('/')[0] in methods and r.get('log', {}).get('each_step_line'):
            add(f"- {k}: {r['log']['each_step_line']}")
            for line in r['log'].get('replay_lines', []):
                add(f'  - {line}')
    add('\nOptimizer from the code: AdamW, betas (0.9, 0.95), eps 1e-8; linear warm-up then constant rate; '
        'gradient clipping at max_norm; bf16 autocast unless no_bf16. `train min` = steps x the last logged s/step '
        '(training only); `wall min` = args.json to the last log line (includes evaluations).\n')

    # D
    add('## D. TMR buffers per session (rank, early stop, objective, weights)\n')
    rows = []
    for k, b in sorted(B.items()):
        if b.get('method') not in ('tmr', 'tmr_joint', 'tmr_joint_neff'):
            continue
        lg, w = b.get('log', {}), b.get('weights', {})
        rows.append([k, PAPER.get(b['session_code']), b.get('n_candidates'), b.get('budget'), b.get('n_clips'),
                     lg.get('omp_picked'), lg.get('n_fill'), lg.get('M'), lg.get('M_bic_neff'), lg.get('M_bic_ND'),
                     lg.get('M_bic_N'), lg.get('M_bic_ND_eff'), lg.get('n_eff'), lg.get('D_eff'),
                     lg.get('same_music_pairs'), lg.get('variance_kept_pct'), lg.get('objective_start'),
                     lg.get('objective_end'), lg.get('modal_error'), w.get('effective_size'),
                     w.get('n_below_1pct_of_mean'), b.get('n_songs')])
    add(table(['buffer', 'paper S', 'N', 'B', '|buffer|', 'OMP picks', 'fill', 'M used', 'M n_eff', 'M N*D',
               'M N', 'M N*D_eff', 'n_eff', 'D_eff', 'same-music pairs', '% var kept', 'objective start',
               'objective end', 'modal error', 'effective size', 'weights < 1% of mean', 'songs'], rows))
    add('\n`OMP picks` < B means the Frank-Wolfe test stopped OMP before the budget and `fill` clips were added by '
        'the same pursuit without the test.\n')

    # E
    add('## E. Shared start, seed and data order (FAQ (v))\n')
    rows = []
    for k, r in R.items():
        if k == 's1' or k.split('/')[0] not in methods:
            continue
        a = r.get('args', {})
        rows.append([k, a.get('init_lm_state'), a.get('seed'), a.get('n_train'), a.get('steps'), a.get('batch_size'),
                     a.get('train_cache'), r.get('starts_in_log')])
    add(table(['run', 'init', 'seed', 'n_train', 'steps', 'batch', 'train cache', 'starts in log.jsonl'], rows))
    for s in ('s2', 's3'):
        sig = {tuple(str(R[k]['args'].get(x)) for x in ('seed', 'n_train', 'steps', 'batch_size', 'lr', 'train_cache'))
               for k in R if k.endswith('/' + s) and k.split('/')[0] in methods}
        add(f"\n{PAPER[s]}: {'identical' if len(sig) == 1 else 'DIFFERENT'} seed, windows, steps, batch, lr and cache "
            f'across the listed methods ({len(sig)} distinct).')
    add('\nFrom the code: current-session batches are drawn by `sample()` from a torch.Generator seeded with --seed '
        '(with replacement), separate from the replay stream (numpy RNG, same seed) and from the distillation draws, so '
        'methods with the same seed see the same current-session batches. train_session.py has no resume: a run that '
        'was started again (starts > 1) retrained from step 0 with the same seed.\n')

    # F
    add('## F. Baseline settings on UniMuMo (Tab. baseline_hp)\n')
    rows = []
    for k, b in sorted(B.items()):
        if b.get('method') in ('tmr', 'tmr_joint', 'tmr_joint_neff', 'random'):
            continue
        rows.append([k, b.get('method'), json.dumps(b.get('params')), json.dumps(b.get('info')), b.get('seconds')])
    add(table(['buffer', 'selection', 'params', 'info', 'build (s)'], rows))
    for k, b in sorted(B.items()):
        lg = b.get('log', {})
        for line in [lg.get('gcr_line')] + lg.get('mir_lines', []) + lg.get('csrel_lines', []):
            if line:
                add(f'- {k}: `{line}`')
    star = [(k, r['log']['star_line']) for k, r in R.items() if r.get('log', {}).get('star_line')]
    for k, line in star:
        add(f'- {k}: `{line}`')
    add('\nThe files record the values used, not whether any range was searched.\n')

    # G
    G = S.get('style_encoder', {})
    add('## G. Style classifier on FineDance, and which metric can see forgetting\n')
    rep = G.get('report') or {}
    if rep.get('summary'):
        add('Validation summary: ' + json.dumps(rep['summary']))
    probes = {k: v for k, v in rep.items() if k.startswith('probe_')}
    if probes:
        add(table(['probe', 'max session mass', 'other', 'windows'],
                  [[k[6:], v.get('max_session_mass'), v.get('other'), v.get('n_windows')] for k, v in probes.items()]))
    for t in G.get('log_test_tables', []):
        add('```\n' + t.rstrip() + '\n```')
    for k in ('log_criteria', 'log_temperature', 'log_babel'):
        if G.get(k) is not None:
            add(f'- {k[4:]}: {G[k]}')
    for line in G.get('log_split_lines', []):
        add(f'- {line}')
    M = S.get('metric_snr', {})
    rows = []
    for s, r in M.get('per_session', {}).items():
        rows.append([PAPER[s], r.get('family'), r.get('n_songs'), r.get('n_clips'), r.get('n_windows'),
                     r.get('StyleRetention_gt'), r.get('StyleRetention_noise'), r.get('SNR_StyleRetention'),
                     r.get('SNR_StyleFD'), r.get('SNR_FID_k'), r.get('SNR_FID_g'), r.get('BAS_real'), r.get('FSR_real')])
    add('\n' + table(['Session', 'Family', 'songs', '10 s clips', '5 s windows', 'SR real', 'SR noise', 'SNR SR',
                      'SNR StyleFD', 'SNR FID_k', 'SNR FID_g', 'BAS real', 'FSR real'], rows, nd=2))
    agg = M.get('min_median_over_sessions', {})
    add('\nMin / median SNR over the three sessions: '
        + '; '.join(f"{k} {fmt(v['min'], 1)} / {fmt(v['median'], 1)}" for k, v in agg.items()))
    add('\nThese use one frozen normalisation over all sessions and 5 s windows (1 s stride), not the per-call '
        'normalisation of the AIST++ code.\n')

    # H
    D = S.get('dance_results', {})
    add('## H. Tab. tab:unimumo cross-check and per-genre results\n')
    rows = []
    for p, m in PAPER_ROWS + [(m, m) for m in D.get('methods', {}) if m not in dict(PAPER_ROWS).values()]:
        c = D.get('methods', {}).get(m, {}).get('tab_unimumo_cells')
        if c:
            rows.append([p, m, c['S1_earlier'], c['S2_earlier'], c['mean_earlier'], c['new'], c['street_final'],
                         c['folk_final'], c['after_S0_street']])
    add(table(['row', 'method', 'S1 earlier', 'S2 earlier', 'Mean', 'New', 'Street final', 'Folk final',
               'Street after S0'], rows))
    add('\nS1 = Street after Folk; S2 = mean of Street and Folk after Classic; Mean = mean of those three values; '
        'New = mean of Folk after Folk and Classic after Classic.\n')
    rows = []
    for m, d in D.get('methods', {}).items():
        q = d['final_quality']
        rows.append([m] + [q[s].get(k) for s, _, _ in SESS for k in ('StyleFD', 'FID_k', 'BAS', 'FSR', 'DanceRate')])
    hdr = ['method'] + [f'{PAPER[s]} {k}' for s, _, _ in SESS for k in ('StyleFD', 'FID_k', 'BAS', 'FSR', 'DanceRate')]
    add('Final dance quality after the last session:\n')
    add(table(hdr, rows, nd=3))
    refs = D.get('references', {})
    if refs:
        add('\nReference rows (real dance, tokenizer reconstruction, pretrained, after S0):\n')
        add(table(['row'] + [f'{PAPER[s]} {k}' for s, _, _ in SESS for k in ('StyleRetention', 'FID_k', 'BAS')],
                  [[k] + [v[s].get(x) for s, _, _ in SESS for x in ('StyleRetention', 'FID_k', 'BAS')]
                   for k, v in refs.items()]))
    samples = {(m, t): (d['meta'].get(t) or {}).get('samples') for m, d in D.get('methods', {}).items()
               for t in ('after_s2', 'after_s3')}
    add(f"\nGenerated samples per test clip: {sorted(set(v for v in samples.values() if v is not None))} "
        f"(cfg {sorted({(d['meta'].get('after_s3') or {}).get('cfg') for d in D.get('methods', {}).values()}, key=str)}).")
    N = S.get('nll', {})
    if N:
        add('\nHeld-out motion CE (outputs/nll):\n')
        sets = sorted({k for v in N.values() for k in v})
        add(table(['file'] + sets, [[f] + [v.get(k, {}).get('motion_ce') for k in sets] for f, v in N.items()], nd=4))

    if S.get('check_setup'):
        add('\n## Parameter counts (check_setup logs)\n')
        for p, lines in S['check_setup'].items():
            add(f'- {p}: ' + ' | '.join(lines))
    if S.get('overlap_s1', {}).get('names'):
        o = S['overlap_s1']
        add('\n## S0 buffers: share of the row buffer also in the column buffer\n')
        add(table(['buffer'] + o['names'], [[n] + r for n, r in zip(o['names'], o['matrix'])], nd=2))
    add('\n## Missing files\n')
    add('\n'.join(f'- {m}' for m in sorted(set(MISSING))) or 'none')
    add('\n## Collector errors\n')
    add('\n'.join(f'- {e}' for e in ERRORS) or 'none')
    (OUT / 'summary.md').write_text('\n'.join(L) + '\n')


# ---------------------------------------------------------------- main
def main():
    global ROOT, OUT
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=str(Path(__file__).resolve().parent.parent), help='repo root on the server')
    ap.add_argument('--out', default='outputs/paper_stats')
    ap.add_argument('--methods', default=' '.join(ALL_METHODS),
                    help='methods whose runs and results to list (paper rows first in the summary)')
    ap.add_argument('--no_tar', action='store_true')
    a = ap.parse_args()
    ROOT = Path(a.root)
    OUT = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'raw').mkdir(parents=True)
    methods = a.methods.split()

    S = {'environment': environment()}
    E = ROOT / 'outputs/dance_eval'
    noise = load_json(E / 'noise.json')
    nll_s0 = load_json(ROOT / 'outputs/nll/fd_s0.json')
    S['runs'] = training_runs()
    S['buffers'] = buffer_files()
    S['splits'] = splits(S['runs'], S['buffers'], noise, nll_s0)
    S['style_encoder'] = style_encoder()
    S['metric_snr'] = metric_snr(noise)
    S['dance_results'] = dance_results(methods)
    S['nll'] = nll_results()
    S['check_setup'] = check_setup()
    dirs, seen = [], set()  # one buffer per selection: the S1 buffers of x and x_kd are the same copy
    for m in methods:
        b = S['buffers'].get(f'{m}/s1')
        if b and b.get('method') not in seen:
            seen.add(b.get('method'))
            dirs.append(m)
    names, mat = overlaps(S['buffers'], dirs)
    S['overlap_s1'] = {'names': names, 'matrix': mat}
    for p in sorted((ROOT / 'logs').glob('replay_*.log')) + sorted((ROOT / 'logs').glob('run_*.log')):
        copy_raw(p)
    for p in ('logs/dance_build.log', 'logs/style_encoder.log', 'TMR_UniMuMo_plan.md'):
        copy_raw(ROOT / p)
    for b in S['buffers'].values():
        b.pop('_ids', None)
    S['missing'] = sorted(set(MISSING))
    S['errors'] = ERRORS
    write_summary(S, methods)
    with open(OUT / 'stats.json', 'w') as f:
        json.dump(S, f, indent=1, default=str)
    print(f'wrote {OUT}/summary.md and stats.json; {len(COPIED)} raw files copied; '
          f'{len(set(MISSING))} expected files missing; {len(ERRORS)} collector errors')
    if not a.no_tar:
        tar = OUT.parent / 'paper_stats_unimumo.tar.gz'
        with tarfile.open(tar, 'w:gz') as t:
            t.add(OUT, arcname=OUT.name)
        print(f'archive: {tar} ({tar.stat().st_size / 1e6:.1f} MB) -> copy it back, e.g. '
              f'scp <server>:{tar.resolve()} .')


if __name__ == '__main__':
    main()
