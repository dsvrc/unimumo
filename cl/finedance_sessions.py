"""Build the three continual sessions from FineDance (song-level train / val / test).

Sessions (by FineDance coarse style, label_json 'style1'):
    s1_street  : Street (+ 'jiewu', pinyin for street dance) - Popping, Jazz, Hiphop, Breaking, Urban, Locking
    s2_classic : Classic - ShenYun, HanTang, DunHuang, Kun
    s3_folk    : Folk    - Dai, Miao, Wei
    ('Mix' = Korean / Choreography / Chinese is left out of the sequence)

Splits: FineDance@Genre ('cross_genre') official test songs stay in test; each session's test set is
topped up to max(3, 15%) of its songs and a 15% (>= 2 songs) validation set is drawn, both spread
across fine genres (style2). The official ignore list is dropped. Seeded, so splits are reproducible.

Run from the repo root after `python cl/motion_convert.py finedance`:
    python cl/finedance_sessions.py
Outputs: data/finedance/splits/<session>_<split>.txt and data/finedance/manifests/<session>_<split>.jsonl
"""
import argparse
import glob
import json
import math
import os
import random
from collections import Counter, defaultdict
from os.path import join as pjoin

import numpy as np

# FineDance dataset/FineDance_dataset.py, get_train_test_list('cross_genre')
CROSS_GENRE_TEST = ['063', '132', '143', '036', '098', '198', '130', '012', '211', '193',
                    '179', '065', '137', '161', '092', '120', '037', '109', '204', '144']
CROSS_GENRE_IGNORE = ['116', '117', '118', '119', '120', '121', '122', '123', '202', '130']

STYLE_TO_SESSION = {'Street': 's1_street', 'jiewu': 's1_street', 'Classic': 's2_classic', 'Folk': 's3_folk'}
SESSIONS = ['s1_street', 's2_classic', 's3_folk']


def spread_by_genre(ids, genre_of, rng):
    """Order ids round-robin over genres (largest genre first) so picks are spread across genres."""
    groups = defaultdict(list)
    for i in ids:
        groups[genre_of[i]].append(i)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups, key=lambda g: (-len(groups[g]), g))
    out = []
    while any(groups[g] for g in order):
        for g in order:
            if groups[g]:
                out.append(groups[g].pop())
    return out


def pick(ordered, k, genre_of, left, min_left):
    """Take up to k ids in order, never letting a genre's count in `left` drop below min_left."""
    chosen = []
    for i in ordered:
        if len(chosen) >= k:
            break
        if left[genre_of[i]] - 1 < min_left:
            continue
        chosen.append(i)
        left[genre_of[i]] -= 1
    return chosen


def main(args):
    rng = random.Random(args.seed)
    root = args.root
    labels = {}
    for f in sorted(glob.glob(pjoin(root, 'label_json', '*.json'))):
        labels[os.path.splitext(os.path.basename(f))[0]] = json.load(open(f))

    songs = defaultdict(list)
    genre_of, skipped = {}, Counter()
    for sid, lab in labels.items():
        sess = STYLE_TO_SESSION.get(lab['style1'])
        if sess is None:
            skipped[f"style {lab['style1']}"] += 1
            continue
        if sid in CROSS_GENRE_IGNORE:
            skipped['official ignore list'] += 1
            continue
        if not os.path.exists(pjoin(root, 'joint_vecs', sid + '.npy')) or \
                not os.path.exists(pjoin(root, 'music_wav', sid + '.wav')):
            skipped['missing motion features or wav'] += 1
            continue
        songs[sess].append(sid)
        genre_of[sid] = {'jazz': 'Jazz'}.get(lab['style2'], lab['style2'])   # one label is lower-case
    print('skipped:', dict(skipped))

    os.makedirs(pjoin(root, 'splits'), exist_ok=True)
    os.makedirs(pjoin(root, 'manifests'), exist_ok=True)
    seconds = lambda ids: sum(np.load(pjoin(root, 'joint_vecs', i + '.npy'), mmap_mode='r').shape[0] for i in ids) / 60
    print(f"\n{'session':12s} {'split':6s} {'songs':>5s} {'hours':>6s}  genres")
    for sess in SESSIONS:
        ids = sorted(songs[sess])
        n = len(ids)
        test = [i for i in ids if i in CROSS_GENRE_TEST]
        n_official = len(test)
        n_test = max(n_official, 3, math.ceil(args.test_frac * n))
        # songs not yet held out, per genre; every genre keeps >= min_train songs for training
        cand = [i for i in ids if i not in test]
        left = Counter(genre_of[i] for i in cand)
        test += pick(spread_by_genre(cand, genre_of, rng), n_test - n_official, genre_of, left, args.min_train)
        rest = [i for i in ids if i not in test]
        n_val = max(2, round(args.val_frac * n))
        val = pick(spread_by_genre(rest, genre_of, rng), n_val, genre_of, left, args.min_train)
        train = [i for i in rest if i not in val]
        missing = sorted(set(genre_of[i] for i in test + val) - set(genre_of[i] for i in train))
        if missing:
            print(f'  note: {sess} held-out genres with no training song (official test only): {missing}')
        assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
        for split, sids in [('train', train), ('val', val), ('test', test)]:
            sids = sorted(sids)
            with open(pjoin(root, 'splits', f'{sess}_{split}.txt'), 'w') as f:
                f.write('\n'.join(sids) + '\n')
            with open(pjoin(root, 'manifests', f'{sess}_{split}.jsonl'), 'w') as f:
                for i in sids:
                    f.write(json.dumps({'id': i, 'music': pjoin(root, 'music_wav', i + '.wav'),
                                        'motion': pjoin(root, 'joint_vecs', i + '.npy'),
                                        'style': labels[i]['style1'], 'genre': genre_of[i]}) + '\n')
            g = Counter(genre_of[i] for i in sids)
            print(f'{sess:12s} {split:6s} {len(sids):5d} {seconds(sids) / 3600:6.2f}  {dict(g)}')
        print(f'{"":12s} (official cross_genre test songs in this session: {n_official})')
    print(f"\nwrote {pjoin(root, 'splits')} and {pjoin(root, 'manifests')}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/finedance')
    ap.add_argument('--test_frac', type=float, default=0.15)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--min_train', type=int, default=1, help='songs per fine genre that must stay in train')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
