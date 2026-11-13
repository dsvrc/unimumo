"""Build the three continual sessions from FineDance, aligned with the DiscoForcing `finedance_style4`
benchmark (scripts/make_cl_sessions.py there) with the Mix family dropped, as that plan recommends:

    s1_street  : Street (+ 'jiewu', pinyin for street dance) - Popping, Jazz, Hiphop, Breaking, Urban, Locking
    s2_folk    : Folk    - Dai, Miao, Wei
    s3_classic : Classic - ShenYun, HanTang, DunHuang, Kun
    (order = DiscoForcing's escalating-drift order: distance from S1 rises 1.10 -> 1.64)

Splits (same algorithm as DiscoForcing):
    test  = official FineDance cross_genre test songs of the family (ignore list removed)
    train = all other labelled songs (ignore list removed)
    val   = round(10%) of each family's train songs, drawn only from sub-genres with >= 3 train songs,
            random.Random(seed) over the families in sorted order (Mix included, so the draws match)
Or import DiscoForcing's split files directly (guarantees identical splits):
    python cl/finedance_sessions.py --disco_dir /path/to/raw_data/cl_sessions/finedance_style4

Run from the repo root after `python cl/motion_convert.py finedance`:
    python cl/finedance_sessions.py
Outputs: data/finedance/splits/<session>_<split>.txt and data/finedance/manifests/<session>_<split>.jsonl
"""
import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict
from os.path import join as pjoin

import numpy as np

# FineDance dataset/FineDance_dataset.py, get_train_test_list('cross_genre')
CROSS_GENRE_TEST = ['063', '132', '143', '036', '098', '198', '130', '012', '211', '193',
                    '179', '065', '137', '161', '092', '120', '037', '109', '204', '144']
CROSS_GENRE_IGNORE = ['116', '117', '118', '119', '120', '121', '122', '123', '202', '130']

FAMILY = {'Street': 'Street', 'jiewu': 'Street', 'Classic': 'Classic', 'Folk': 'Folk', 'Mix': 'Mix'}
ORDER = [('s1_street', 'Street'), ('s2_folk', 'Folk'), ('s3_classic', 'Classic')]   # Mix dropped


def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]


def own_splits(labels, val_ratio, seed):
    """DiscoForcing's make_cl_sessions.py FineDance logic, reproduced."""
    ids = sorted(labels)
    f_test = [i for i in ids if i in CROSS_GENRE_TEST and i not in CROSS_GENRE_IGNORE]
    f_train = [i for i in ids if i not in CROSS_GENRE_TEST and i not in CROSS_GENRE_IGNORE]
    fam = {n: FAMILY[labels[n]['style1']] for n in f_train + f_test}
    by_fam = defaultdict(lambda: {'train': [], 'val': [], 'test': []})
    vrng = random.Random(seed)
    for family in sorted(set(fam.values())):
        members = sorted(n for n in f_train if fam[n] == family)
        sub_count = defaultdict(int)
        for n in members:
            sub_count[labels[n]['style2']] += 1
        pool = [n for n in members if sub_count[labels[n]['style2']] >= 3]
        n_val = max(1, round(val_ratio * len(members)))
        held = set(vrng.sample(pool, n_val))
        by_fam[family]['train'] = [n for n in members if n not in held]
        by_fam[family]['val'] = sorted(held)
        by_fam[family]['test'] = sorted(n for n in f_test if fam[n] == family)
    return by_fam


def disco_splits(disco_dir):
    man = json.load(open(pjoin(disco_dir, 'manifest.json')))
    by_fam = {}
    for k, s in enumerate(man['sessions']):
        by_fam[s['groups'][0]] = {sp: read_list(pjoin(disco_dir, f's{k}', f'{sp}.txt')) for sp in ('train', 'val', 'test')}
    print('DiscoForcing order:', ' -> '.join(s['groups'][0] for s in man['sessions']))
    return by_fam


def main(args):
    root = args.root
    labels = {os.path.splitext(os.path.basename(f))[0]: json.load(open(f))
              for f in sorted(glob.glob(pjoin(root, 'label_json', '*.json')))}
    by_fam = disco_splits(args.disco_dir) if args.disco_dir else own_splits(labels, args.val_ratio, args.seed)

    os.makedirs(pjoin(root, 'splits'), exist_ok=True)
    os.makedirs(pjoin(root, 'manifests'), exist_ok=True)
    have = lambda i: os.path.exists(pjoin(root, 'joint_vecs', i + '.npy')) and os.path.exists(pjoin(root, 'music_wav', i + '.wav'))
    hours = lambda ids: sum(np.load(pjoin(root, 'joint_vecs', i + '.npy'), mmap_mode='r').shape[0] for i in ids) / 60 / 3600
    print(f"\n{'session':12s} {'split':6s} {'songs':>5s} {'hours':>6s}  sub-genres")
    for sess, family in ORDER:
        for split in ('train', 'val', 'test'):
            ids = sorted(by_fam[family][split])
            missing = [i for i in ids if not have(i)]
            ids = [i for i in ids if have(i)]
            with open(pjoin(root, 'splits', f'{sess}_{split}.txt'), 'w') as f:
                f.write('\n'.join(ids) + '\n')
            with open(pjoin(root, 'manifests', f'{sess}_{split}.jsonl'), 'w') as f:
                for i in ids:
                    f.write(json.dumps({'id': i, 'music': pjoin(root, 'music_wav', i + '.wav'),
                                        'motion': pjoin(root, 'joint_vecs', i + '.npy'),
                                        'family': family, 'genre': labels[i]['style2']}) + '\n')
            g = Counter(labels[i]['style2'] for i in ids)
            print(f'{sess:12s} {split:6s} {len(ids):5d} {hours(ids):6.2f}  {dict(g)}'
                  + (f'  (missing features/wav: {missing})' if missing else ''))
    print(f"\nwrote {pjoin(root, 'splits')} and {pjoin(root, 'manifests')}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/finedance')
    ap.add_argument('--disco_dir', default=None, help='DiscoForcing raw_data/cl_sessions/finedance_style4 folder')
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
