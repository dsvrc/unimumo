"""SMPL-family motion -> UniMuMo's 263-d HumanML3D features at 60 fps.

Port of HumanML3D's motion_representation (uniform_skeleton + process_file) on top of the
Skeleton / quaternion utilities that ship with UniMuMo, plus SMPL forward kinematics,
up-axis normalisation and fps resampling (needed for FineDance, which is 30 fps).

Self-checks, run from the repo root before converting any FineDance file:
  # 1. port check: UniMuMo features -> joints -> our features must give back the originals
  python cl/motion_convert.py selftest
  # 2. full pipeline: AIST++ SMPL (EDGE pkl) -> our features, compared with UniMuMo's own features
  python cl/motion_convert.py aist --edge_root /path/to/EDGE --split data/motion/aist_val.txt \
      --motion_dir data/motion/val/joint_vecs --n 10 --resample_check
  # 3. convert AIST++ sequences that UniMuMo's download lacks (its test split), with the best variant
  python cl/motion_convert.py aist --edge_root /path/to/EDGE --split data/motion/aist_test.txt \
      --save_dir data/motion/test/joint_vecs --no_compare --mirror <best> --feet_thre <best>

263-d layout (per frame): root rot vel (1), root lin vel xz (2), root y (1), ric positions (21*3),
6D rotations (21*6), local velocities (22*3), foot contacts (4).
"""
import argparse
import ast
import os
import pickle
import sys
from os.path import join as pjoin
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

# skeleton.py imports the deprecated scipy.ndimage.filters; alias it on SciPy versions without it
try:
    import scipy.ndimage.filters  # noqa: F401
except ImportError:
    import scipy.ndimage as _ndi
    sys.modules['scipy.ndimage.filters'] = _ndi

from unimumo.motion.common.skeleton import Skeleton
from unimumo.motion.common.quaternion import qbetween_np, qrot_np, qmul_np, qinv_np, quaternion_to_cont6d_np
from unimumo.motion.motion_process import recover_from_ric

# ---- HumanML3D constants (paramUtil.t2m_raw_offsets / t2m_kinematic_chain) ----
T2M_RAW_OFFSETS = np.array([
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0],
    [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1],
    [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]], dtype=np.float64)
T2M_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
N_RAW_OFFSETS = torch.from_numpy(T2M_RAW_OFFSETS)
L_IDX1, L_IDX2 = 5, 8                 # lower legs (scale ratio)
FID_R, FID_L = [8, 11], [7, 10]       # right / left foot joints
FACE_JOINT_IDX = [2, 1, 17, 16]       # r_hip, l_hip, sdr_r, sdr_l
JOINTS_NUM = 22

GROUPS = {'root': slice(0, 4), 'ric': slice(4, 67), 'rot': slice(67, 193), 'vel': slice(193, 259),
          'contact': slice(259, 263)}


# ------------------------------------------------------------------ HumanML3D port
def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(N_RAW_OFFSETS, T2M_CHAIN, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0])).numpy()
    tgt_offset = target_offset.numpy()
    src_leg_len = np.abs(src_offset[L_IDX1]).max() + np.abs(src_offset[L_IDX2]).max()
    tgt_leg_len = np.abs(tgt_offset[L_IDX1]).max() + np.abs(tgt_offset[L_IDX2]).max()
    scale_rt = tgt_leg_len / src_leg_len
    tgt_root_pos = positions[:, 0] * scale_rt
    quat_params = src_skel.inverse_kinematics_np(positions, FACE_JOINT_IDX)
    src_skel.set_offset(target_offset)
    return src_skel.forward_kinematics_np(quat_params, tgt_root_pos)


def positions_to_hml263(positions, target_offset, feet_thre=0.002):
    """positions: (T, 22, 3), y-up. Returns (T-1, 263) raw (unnormalised) features."""
    positions = uniform_skeleton(np.asarray(positions, dtype=np.float64).copy(), target_offset)

    # put on floor, XZ at origin, first frame faces Z+
    positions[:, :, 1] -= positions.min(axis=0).min(axis=0)[1]
    root_pos_init = positions[0]
    positions = positions - root_pos_init[0] * np.array([1, 0, 1])
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_IDX
    across = (root_pos_init[r_hip] - root_pos_init[l_hip]) + (root_pos_init[sdr_r] - root_pos_init[sdr_l])
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    root_quat_init = qbetween_np(forward_init, np.array([[0, 0, 1]]))
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = qrot_np(root_quat_init, positions)
    global_positions = positions.copy()

    # foot contacts
    def foot_detect(pos, thres):
        velfactor = np.array([thres, thres])
        feet_l = ((pos[1:, FID_L] - pos[:-1, FID_L]) ** 2).sum(-1)
        feet_r = ((pos[1:, FID_R] - pos[:-1, FID_R]) ** 2).sum(-1)
        return (feet_l < velfactor).astype(np.float32), (feet_r < velfactor).astype(np.float32)

    feet_l, feet_r = foot_detect(positions, feet_thre)

    # rotations (smoothed forward), root velocities
    skel = Skeleton(N_RAW_OFFSETS, T2M_CHAIN, 'cpu')
    quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_IDX, smooth_forward=True)
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    r_rot = quat_params[:, 0].copy()
    velocity = qrot_np(r_rot[1:], (positions[1:, 0] - positions[:-1, 0]).copy())
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

    # rotation-invariant (root-local) positions
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)

    root_y = positions[:, 0, 1:2]
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    ric_data = positions[:, 1:].reshape(len(positions), -1)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1]).reshape(len(global_positions) - 1, -1)
    data = np.concatenate([root_data, ric_data[:-1], rot_data[:-1], local_vel, feet_l, feet_r], axis=-1)
    return data.astype(np.float32)


def target_offsets_from_features(path):
    """HumanML3D's target skeleton = raw directions x bone lengths of the uniform skeleton.
    Every UniMuMo feature file was made with it, so the lengths can be read back from any of them."""
    joints = recover_from_ric(torch.from_numpy(np.load(path)).float()[None], JOINTS_NUM)[0].numpy()
    parents = Skeleton(N_RAW_OFFSETS, T2M_CHAIN, 'cpu').parents()
    lengths = np.stack([np.linalg.norm(joints[:, j] - joints[:, parents[j]], axis=-1)
                        for j in range(1, JOINTS_NUM)], axis=1)              # (T, 21)
    offs = T2M_RAW_OFFSETS.copy()
    offs[1:] *= lengths.mean(0)[:, None]
    return torch.from_numpy(offs).float(), float(lengths.std(0).max())


# ------------------------------------------------------------------ SMPL helpers
def axis_angle_to_quat(aa):
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = angle < 1e-8
    k = np.where(small, 0.5 - angle ** 2 / 48.0, np.sin(angle / 2) / np.where(small, 1.0, angle))
    return np.concatenate([np.cos(angle / 2), aa * k], axis=-1)


def smpl_fk(quats, root_pos, offsets, parents):
    """quats (T, J, 4) local, root_pos (T, 3), offsets (J, 3) rest offsets to parent -> (T, J, 3)."""
    T, J, _ = quats.shape
    pos = np.zeros((T, J, 3))
    world = [None] * J
    pos[:, 0] = root_pos
    world[0] = quats[:, 0]
    for j in range(1, J):
        p = parents[j]
        pos[:, j] = qrot_np(np.ascontiguousarray(world[p]), np.repeat(offsets[j][None], T, 0)) + pos[:, p]
        world[j] = qmul_np(np.ascontiguousarray(world[p]), np.ascontiguousarray(quats[:, j]))
    return pos


def resample(quats, root_pos, src_fps, dst_fps):
    """nlerp rotations / lerp root positions from src_fps to dst_fps (both (T, ...))."""
    T = len(quats)
    n = int(np.floor((T - 1) * dst_fps / src_fps + 1e-9)) + 1
    t = np.arange(n) * src_fps / dst_fps
    i0 = np.floor(t + 1e-9).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (t - i0)[:, None]
    q0, q1 = quats[i0], quats[i1]
    sign = np.sign((q0 * q1).sum(-1, keepdims=True))
    sign[sign == 0] = 1
    q = (1 - w[..., None]) * q0 + w[..., None] * q1 * sign
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q, (1 - w) * root_pos[i0] + w * root_pos[i1]


_UP_TO_Y = {  # rotation R with R @ axis = +y, det(R) = +1
    (1, 1): np.eye(3),
    (1, -1): np.diag([1.0, -1.0, -1.0]),
    (2, 1): np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float),
    (2, -1): np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
    (0, 1): np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
    (0, -1): np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float),
}


def to_y_up(joints):
    """Rotate so the pelvis->head direction is +y (proper rotation, no mirroring)."""
    up = (joints[:, 15] - joints[:, 0]).mean(0)
    ax = int(np.argmax(np.abs(up)))
    key = (ax, int(np.sign(up[ax])))
    return joints @ _UP_TO_Y[key].T, 'xyz'[ax] if key[1] > 0 else '-' + 'xyz'[ax]


def load_edge_skeleton(edge_root):
    """Read smpl_offsets / smpl_parents literally from EDGE's vis.py (it imports pytorch3d, so not importable)."""
    tree = ast.parse(open(pjoin(edge_root, 'vis.py')).read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('smpl_offsets', 'smpl_parents'):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    assert {'smpl_offsets', 'smpl_parents'} <= found.keys(), f'could not find SMPL skeleton in {edge_root}/vis.py'
    return np.array(found['smpl_offsets'], dtype=np.float64), list(found['smpl_parents'])


def edge_pkl_to_joints(pkl_path, offsets, parents, src_fps=60, dst_fps=60):
    m = pickle.load(open(pkl_path, 'rb'))
    q = np.asarray(m['q'], dtype=np.float64).reshape(len(m['q']), -1, 3)       # (T, 24, 3) axis-angle
    pos = np.asarray(m['pos'], dtype=np.float64) / np.asarray(m['scale']).reshape(-1)[0]
    quats = axis_angle_to_quat(q)
    if src_fps != dst_fps:
        quats, pos = resample(quats, pos, src_fps, dst_fps)
    return smpl_fk(quats, pos, offsets, parents)[:, :JOINTS_NUM]


# ------------------------------------------------------------------ comparison helpers
def best_shift(a, b, max_shift=30):
    """Shift s such that a[t] ~ b[t + s]; compared on root height + ric positions."""
    best = (np.inf, 0)
    cols = list(range(3, 67))
    for s in range(-max_shift, max_shift + 1):
        a0, b0 = max(0, -s), max(0, s)
        n = min(len(a) - a0, len(b) - b0)
        if n < 30:
            continue
        err = np.mean((a[a0:a0 + n, cols] - b[b0:b0 + n, cols]) ** 2)
        best = min(best, (err, s))
    return best[1]


def compare(ours, ref, shift=0):
    a0, b0 = max(0, -shift), max(0, shift)
    n = min(len(ours) - a0, len(ref) - b0)
    a, b = ours[a0:a0 + n], ref[b0:b0 + n]
    out = {}
    for g, sl in GROUPS.items():
        if g == 'contact':
            out[g] = float((a[:, sl] == b[:, sl]).mean())            # agreement
        else:
            out[g] = float(np.sqrt(np.mean((a[:, sl] - b[:, sl]) ** 2)) / (b[:, sl].std() + 1e-8))  # rel. RMSE
    out['frames'] = int(n)
    return out


def fmt(res):
    return ' '.join(f'{k}={v:.3f}' if k != 'frames' else f'{k}={v}' for k, v in res.items())


def default_ref(motion_root='data/motion'):
    for split in ['train', 'val']:
        p = pjoin(motion_root, split, 'joint_vecs', '000021.npy')
        if os.path.exists(p):
            return p
    with open(pjoin(motion_root, 'humanml3d_train.txt')) as f:
        for l in f:
            p = pjoin(motion_root, 'train', 'joint_vecs', l.strip() + '.npy')
            if os.path.exists(p):
                return p
    raise FileNotFoundError('no HumanML3D feature file found for the target skeleton')


# ------------------------------------------------------------------ commands
def cmd_selftest(args):
    tgt, len_std = target_offsets_from_features(args.ref)
    print(f'target skeleton from {args.ref}: max bone-length std over frames {len_std:.2e} (should be ~0)')
    files = args.files
    if not files:
        files = []
        for name, n in [('aist', 3), ('humanml3d', 2)]:
            with open(f'data/motion/{name}_val.txt') as f:
                ids = [l.strip() for l in f if l.strip()]
            files += [p for p in (f'data/motion/val/joint_vecs/{i}.npy' for i in ids) if os.path.exists(p)][:n]
    for path in files:
        ref = np.load(path)
        joints = recover_from_ric(torch.from_numpy(ref).float()[None], JOINTS_NUM)[0].numpy()
        print(f'\n{os.path.basename(path)}: {ref.shape}')
        for thre in args.feet_thre:
            ours = positions_to_hml263(joints, tgt, thre)
            print(f'  feet_thre={thre:<9g} {fmt(compare(ours, ref))}')
    print('\nExpect ric/vel/root rel. RMSE ~0 and contact agreement ~1 for the right threshold. '
          'A small rot error is expected only from frame-0 / smoothing edge effects.')


def cmd_aist(args):
    offsets, parents = load_edge_skeleton(args.edge_root)
    tgt, _ = target_offsets_from_features(args.ref)
    with open(args.split) as f:
        seqs = [l.strip() for l in f if l.strip()]
    pkl_dir = pjoin(args.edge_root, 'edge_aistpp', 'motions')
    done = 0
    for seq in seqs:
        pkl = pjoin(pkl_dir, seq + '.pkl')
        ref_path = pjoin(args.motion_dir, seq + '.npy') if args.motion_dir else None
        if not os.path.exists(pkl) or (not args.no_compare and not (ref_path and os.path.exists(ref_path))):
            continue
        joints = edge_pkl_to_joints(pkl, offsets, parents)
        joints, up = to_y_up(joints)
        mirrors = [False, True] if args.mirror == 'both' else [args.mirror == 'yes']
        if args.no_compare:
            j = joints.copy()
            if mirrors[0]:
                j[..., 0] *= -1
            feats = positions_to_hml263(j, tgt, args.feet_thre[0])
            os.makedirs(args.save_dir, exist_ok=True)
            np.save(pjoin(args.save_dir, seq + '.npy'), feats)
            print(f'{seq}: {joints.shape[0]} frames (up {up}) -> {feats.shape} saved')
        else:
            ref = np.load(ref_path)
            print(f'\n{seq}: ours from {joints.shape[0]} frames (detected up {up}), UniMuMo {ref.shape}')
            for mirror in mirrors:
                j = joints.copy()
                if mirror:
                    j[..., 0] *= -1
                for thre in args.feet_thre:
                    ours = positions_to_hml263(j, tgt, thre)
                    s = best_shift(ours, ref)
                    print(f'  mirror={str(mirror):5s} feet_thre={thre:<9g} shift={s:+d} {fmt(compare(ours, ref, s))}')
            if args.resample_check:
                # 60 -> 30 -> 60 fps against the direct 60 fps conversion (estimates FineDance's resampling error)
                m = pickle.load(open(pkl, 'rb'))
                q = axis_angle_to_quat(np.asarray(m['q'], dtype=np.float64).reshape(len(m['q']), -1, 3))
                pos = np.asarray(m['pos'], dtype=np.float64) / np.asarray(m['scale']).reshape(-1)[0]
                q30, p30 = q[::2], pos[::2]
                q60, p60 = resample(q30, p30, 30, 60)
                j60 = to_y_up(smpl_fk(q60, p60, offsets, parents)[:, :JOINTS_NUM])[0]
                direct = positions_to_hml263(joints, tgt, args.feet_thre[0])
                resampled = positions_to_hml263(j60, tgt, args.feet_thre[0])
                print(f'  resample 60->30->60 vs direct (mirror=False, feet_thre={args.feet_thre[0]:g}): '
                      f'{fmt(compare(resampled, direct))}')
        done += 1
        if args.n and done >= args.n:
            break
    print(f'\nprocessed {done} sequences')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    st = sub.add_parser('selftest', help='UniMuMo features -> joints -> our features')
    st.add_argument('--files', nargs='*', default=None)
    st.add_argument('--ref', default=None, help='HumanML3D feature file defining the target skeleton')
    st.add_argument('--feet_thre', type=float, nargs='+', default=[0.002, 0.002 / 9])

    ai = sub.add_parser('aist', help='EDGE AIST++ pkl -> features (compare with UniMuMo or save)')
    ai.add_argument('--edge_root', required=True)
    ai.add_argument('--split', default='data/motion/aist_val.txt')
    ai.add_argument('--motion_dir', default='data/motion/val/joint_vecs')
    ai.add_argument('--ref', default=None)
    ai.add_argument('--n', type=int, default=10)
    ai.add_argument('--mirror', choices=['both', 'yes', 'no'], default='both')
    ai.add_argument('--feet_thre', type=float, nargs='+', default=[0.002, 0.002 / 9])
    ai.add_argument('--resample_check', action='store_true')
    ai.add_argument('--no_compare', action='store_true', help='just convert and save')
    ai.add_argument('--save_dir', default=None)

    args = ap.parse_args()
    if args.ref is None:
        args.ref = default_ref()
    if args.cmd == 'selftest':
        cmd_selftest(args)
    else:
        if args.no_compare:
            assert args.save_dir, '--no_compare needs --save_dir'
            assert args.mirror != 'both' and len(args.feet_thre) == 1, 'pick one --mirror and one --feet_thre'
            args.n = 0
        cmd_aist(args)
