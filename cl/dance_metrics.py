"""Dance metrics for the UniMuMo FineDance sessions, copied from DiscoForcing so the numbers are comparable:

  * window features, frozen-normalisation FID, StyleFD   <- DiscoForcing metrics/cl_metrics.py
  * kinetic features (Bailando), geometric features      <- DiscoForcing metrics/m2d_features/{kinetic,manual_new,utils}.py
    (fairmotion, BSD licence)
  * Frechet distance, diversity                          <- DiscoForcing metrics/tools/utils.py (text-to-motion, MIT)
  * BAS, foot skating ratio                               <- DiscoForcing metrics/m2d_features/beat_align.py and
                                                             metrics/tools/utils.py, i.e. LODGE's FineDance metrics
Upstream quirks are kept on purpose (the kinetic acceleration term, librosa's default sr in the start tempo,
the foot-skating tangent mean, BAS on root-relative joints) so values stay comparable with the literature.

Every function takes global joint positions (T, 22, 3): SMPL joint order, y-up, metres, 30 fps.
"""
import numpy as np
import scipy.linalg
import scipy.signal
from scipy.ndimage import gaussian_filter as G

if not hasattr(scipy.signal, 'hann'):  # librosa 0.10.1's beat tracker calls scipy.signal.hann, removed in SciPy 1.13
    scipy.signal.hann = scipy.signal.windows.hann
from scipy.signal import argrelextrema
from scipy.spatial.distance import pdist

FPS, WIN, STRIDE = 30, 150, 30


# ---------------------------------------------------------------- geometric (manual) features, verbatim
def distance_between_points(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def distance_from_plane(a, b, c, p, threshold):
    ba = np.array(b) - np.array(a)
    ca = np.array(c) - np.array(a)
    cross = np.cross(ca, ba)
    pa = np.array(p) - np.array(a)
    return np.dot(cross, pa) / np.linalg.norm(cross) > threshold


def distance_from_plane_normal(n1, n2, a, p, threshold):
    normal = np.array(n2) - np.array(n1)
    pa = np.array(p) - np.array(a)
    return np.dot(normal, pa) / np.linalg.norm(normal) > threshold


def angle_within_range(j1, j2, k1, k2, range):
    j = np.array(j2) - np.array(j1)
    k = np.array(k2) - np.array(k1)
    angle = np.degrees(np.arccos(np.dot(j, k) / (np.linalg.norm(j) * np.linalg.norm(k))))
    return bool(range[0] < angle < range[1])


def velocity_direction_above_threshold(j1, j1_prev, j2, j2_prev, p, p_prev, threshold, time_per_frame=1 / 120.0):
    velocity = np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    direction = np.array(j2) - np.array(j1)
    velocity_along_direction = np.dot(velocity, direction) / np.linalg.norm(direction)
    return velocity_along_direction / time_per_frame > threshold


def velocity_direction_above_threshold_normal(j1, j1_prev, j2, j3, p, p_prev, threshold, time_per_frame=1 / 120.0):
    velocity = np.array(p) - np.array(j1) - (np.array(p_prev) - np.array(j1_prev))
    direction = np.cross(np.array(j3) - np.array(j1), np.array(j2) - np.array(j1))
    velocity_along_direction = np.dot(velocity, direction) / np.linalg.norm(direction)
    return velocity_along_direction / time_per_frame > threshold


def velocity_above_threshold(p, p_prev, threshold, time_per_frame=1 / 120.0):
    return np.linalg.norm(np.array(p) - np.array(p_prev)) / time_per_frame > threshold


SMPL_JOINT_NAMES = [
    "root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", "rankle", "chest", "ltoes", "rtoes", "neck",
    "linshoulder", "rinshoulder", "head", "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist",
    "lhand", "rhand",
]


class ManualFeatures:
    def __init__(self, positions, joint_names=SMPL_JOINT_NAMES):
        self.positions = positions
        self.joint_names = joint_names
        self.frame_num = 1
        self.hl = distance_between_points([1.99113488e-01, 2.36807942e-01, -1.80702247e-02],   # lshoulder
                                          [4.54445392e-01, 2.21158922e-01, -4.10167128e-02])   # lelbow
        self.sw = distance_between_points([1.99113488e-01, 2.36807942e-01, -1.80702247e-02],   # lshoulder
                                          [-1.91692337e-01, 2.36928746e-01, -1.23055102e-02])  # rshoulder
        self.hw = distance_between_points([5.64076714e-02, -3.23069185e-01, 1.09197125e-02],   # lhip
                                          [-6.24834076e-02, -3.31302464e-01, 1.50412619e-02])  # rhip

    def next_frame(self):
        self.frame_num += 1

    def transform_and_fetch_position(self, j):
        if j == "y_unit":
            return [0, 1, 0]
        elif j == "minus_y_unit":
            return [0, -1, 0]
        elif j == "zero":
            return [0, 0, 0]
        elif j == "y_min":
            return [0, min([y for (_, y, _) in self.positions[self.frame_num]]), 0]
        return self.positions[self.frame_num][self.joint_names.index(j)]

    def transform_and_fetch_prev_position(self, j):
        return self.positions[self.frame_num - 1][self.joint_names.index(j)]

    def f_move(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]]
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return velocity_direction_above_threshold(j1, j1_prev, j2, j2_prev, j3, j3_prev, range)

    def f_nmove(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]]
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return velocity_direction_above_threshold_normal(j1, j1_prev, j2, j3, j4, j4_prev, range)

    def f_plane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return distance_from_plane(j1, j2, j3, j4, threshold)

    def f_nplane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return distance_from_plane_normal(j1, j2, j3, j4, threshold)

    def f_angle(self, j1, j2, j3, j4, range):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return angle_within_range(j1, j2, j3, j4, range)

    def f_fast(self, j1, threshold):
        j1_prev = self.transform_and_fetch_prev_position(j1)
        j1 = self.transform_and_fetch_position(j1)
        return velocity_above_threshold(j1, j1_prev, threshold)


def extract_manual_features(positions, reduce=True):
    """Per-frame geometric features (32) averaged over the sequence; reduce=False returns them per frame."""
    assert len(positions.shape) == 3
    features = []
    f = ManualFeatures(positions)
    for _ in range(1, positions.shape[0]):
        pf = [
            f.f_nmove("neck", "rhip", "lhip", "rwrist", 1.8 * f.hl),
            f.f_nmove("neck", "lhip", "rhip", "lwrist", 1.8 * f.hl),
            f.f_nplane("chest", "neck", "neck", "rwrist", 0.2 * f.hl),
            f.f_nplane("chest", "neck", "neck", "lwrist", 0.2 * f.hl),
            f.f_move("belly", "chest", "chest", "rwrist", 1.8 * f.hl),
            f.f_move("belly", "chest", "chest", "lwrist", 1.8 * f.hl),
            f.f_angle("relbow", "rshoulder", "relbow", "rwrist", [0, 110]),
            f.f_angle("lelbow", "lshoulder", "lelbow", "lwrist", [0, 110]),
            f.f_nplane("lshoulder", "rshoulder", "lwrist", "rwrist", 2.5 * f.sw),
            f.f_move("lwrist", "rwrist", "rwrist", "lwrist", 1.4 * f.hl),
            f.f_move("rwrist", "root", "lwrist", "root", 1.4 * f.hl),
            f.f_move("lwrist", "root", "rwrist", "root", 1.4 * f.hl),
            f.f_fast("rwrist", 2.5 * f.hl),
            f.f_fast("lwrist", 2.5 * f.hl),
            f.f_plane("root", "lhip", "ltoes", "rankle", 0.38 * f.hl),
            f.f_plane("root", "rhip", "rtoes", "lankle", 0.38 * f.hl),
            f.f_nplane("zero", "y_unit", "y_min", "rankle", 1.2 * f.hl),
            f.f_nplane("zero", "y_unit", "y_min", "lankle", 1.2 * f.hl),
            f.f_nplane("lhip", "rhip", "lankle", "rankle", 2.1 * f.hw),
            f.f_angle("rknee", "rhip", "rknee", "rankle", [0, 110]),
            f.f_angle("lknee", "lhip", "lknee", "lankle", [0, 110]),
            f.f_fast("rankle", 2.5 * f.hl),
            f.f_fast("lankle", 2.5 * f.hl),
            f.f_angle("neck", "root", "rshoulder", "relbow", [25, 180]),
            f.f_angle("neck", "root", "lshoulder", "lelbow", [25, 180]),
            f.f_angle("neck", "root", "rhip", "rknee", [50, 180]),
            f.f_angle("neck", "root", "lhip", "lknee", [50, 180]),
            f.f_plane("rankle", "neck", "lankle", "root", 0.5 * f.hl),
            f.f_angle("neck", "root", "zero", "y_unit", [70, 110]),
            f.f_nplane("zero", "minus_y_unit", "y_min", "rwrist", -1.2 * f.hl),
            f.f_nplane("zero", "minus_y_unit", "y_min", "lwrist", -1.2 * f.hl),
            f.f_fast("root", 2.3 * f.hl),
        ]
        features.append(pf)
        f.next_frame()
    features = np.array(features, dtype=np.float32)
    return features.mean(axis=0) if reduce else features


# ---------------------------------------------------------------- kinetic features (Bailando)
def _avg_velocity_horizontal(positions, i, joint_idx, sliding_window, frame_time):
    n, v = 0, np.zeros(3)
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        v += positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        n += 1
    return np.linalg.norm(np.array([v[0], v[2]]) / (n * frame_time))


def _avg_velocity_vertical(positions, i, joint_idx, sliding_window, frame_time):
    n, v = 0, np.zeros(3)
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(positions):
            continue
        v += positions[i + j][joint_idx] - positions[i + j - 1][joint_idx]
        n += 1
    return np.linalg.norm(np.array([v[1]]) / (n * frame_time))


def _avg_acceleration(positions, i, joint_idx, sliding_window, frame_time):
    n, a = 0, np.zeros(3)
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j + 1 >= len(positions):
            continue
        v2 = (positions[i + j + 1][joint_idx] - positions[i + j][joint_idx]) / frame_time
        v1 = positions[i + j][joint_idx] - positions[i + j - 1][joint_idx] / frame_time  # upstream precedence quirk
        a += (v2 - v1) / frame_time
        n += 1
    return np.linalg.norm(a / n)


def extract_kinetic_features(positions, frame_time=1.0 / FPS, sliding_window=2):
    """Reference (loop) implementation, y-up: per joint horizontal KE, vertical KE, energy expenditure (66)."""
    T = len(positions)
    out = []
    for joint in range(positions.shape[1]):
        out += [sum(_avg_velocity_horizontal(positions, i, joint, sliding_window, frame_time) ** 2 for i in range(1, T)) / (T - 1.0),
                sum(_avg_velocity_vertical(positions, i, joint, sliding_window, frame_time) ** 2 for i in range(1, T)) / (T - 1.0),
                sum(_avg_acceleration(positions, i, joint, sliding_window, frame_time) for i in range(1, T)) / (T - 1.0)]
    return np.array(out, dtype=np.float32)


def kinetic_per_frame(pos, frame_time=1.0 / FPS, sliding_window=2):
    """Vectorised per-frame terms of extract_kinetic_features (frames 1..T-1), shape (T-1, J*3)."""
    T, sw, dt = len(pos), sliding_window, frame_time
    i = np.arange(1, T)
    d = np.zeros_like(pos, dtype=np.float64)
    d[1:] = pos[1:] - pos[:-1]
    cs = np.concatenate([np.zeros((1,) + pos.shape[1:]), np.cumsum(d, 0)])
    lo, hi = np.maximum(i - sw, 1), np.minimum(i + sw, T - 1)
    v = (cs[hi + 1] - cs[lo]) / ((hi - lo + 1)[:, None, None] * dt)
    ke_h = v[..., 0] ** 2 + v[..., 2] ** 2
    ke_v = v[..., 1] ** 2
    term = np.zeros_like(pos, dtype=np.float64)
    k = np.arange(1, T - 1)
    term[k] = ((pos[k + 1] - pos[k]) / dt - (pos[k] - pos[k - 1] / dt)) / dt
    ct = np.concatenate([np.zeros((1,) + pos.shape[1:]), np.cumsum(term, 0)])
    lo2, hi2 = np.maximum(i - sw, 1), np.minimum(i + sw, T - 2)
    ee = np.linalg.norm((ct[hi2 + 1] - ct[lo2]) / (hi2 - lo2 + 1)[:, None, None], axis=-1)
    return np.stack([ke_h, ke_v, ee], -1).reshape(T - 1, -1)


def root_relative(joints):
    """M2DMetrics convention: joints 1.. relative to the root, the root keeps its own position."""
    j = joints.astype(np.float64).copy()
    j[:, 1:, :] -= j[:, 0:1, :]
    return j


def window_features(joints_rr, win=WIN, stride=STRIDE):
    """(n_windows, 66) kinetic and (n_windows, 32) geometric features of one root-relative sequence, averaged over
    5 s windows with 1 s stride (a window over frames [s, s+win) uses the per-frame rows of frames s+1..s+win-1)."""
    per = [kinetic_per_frame(joints_rr), extract_manual_features(joints_rr, reduce=False).astype(np.float64)]
    T = len(joints_rr)
    spans = [(s, s + win - 1) for s in range(0, T - win + 1, stride)] or [(0, T - 1)]
    out = []
    for p in per:
        c = np.concatenate([np.zeros((1, p.shape[1])), np.cumsum(p, 0)])
        out.append(np.stack([(c[b] - c[a]) / (b - a) for a, b in spans]))
    return out[0], out[1]


# ---------------------------------------------------------------- Frechet distance, diversity
def activation_statistics(x):
    return np.mean(x, axis=0), np.cov(x, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def frozen_fid(ref, gen, mean, std):
    """FID after ONE fixed normalisation shared by every session and model (DiscoForcing cl_metrics)."""
    m1, c1 = activation_statistics((ref - mean) / std)
    m2, c2 = activation_statistics((gen - mean) / std)
    return frechet_distance(m1, c1, m2, c2)


def mean_pairwise_distance(x):
    """Diversity: mean Euclidean distance over all pairs (LODGE calculate_avg_distance, vectorised)."""
    return float(pdist(x).mean()) if len(x) > 1 else float('nan')


# ---------------------------------------------------------------- BAS (Bailando / LODGE)
def music_beats(wav, sr):
    """Beat frames at 30 fps of a waveform (LODGE get_music_beat_fromwav, taking an array instead of a path)."""
    import librosa
    try:
        from librosa.feature.rhythm import tempo as _tempo
    except ImportError:
        from librosa.beat import tempo as _tempo
    SR = FPS * 512
    data = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=SR)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)
    start_bpm = _tempo(y=data)[0]  # upstream calls it without sr (librosa's 22050 default): kept
    _, beat_idxs = librosa.beat.beat_track(onset_envelope=envelope, sr=SR, hop_length=512,
                                           start_bpm=start_bpm, tightness=100)
    return beat_idxs


def dance_beats(joints):
    """Local minima of the smoothed mean joint speed (LODGE calc_db)."""
    kinetic_vel = np.mean(np.sqrt(np.sum((joints[1:] - joints[:-1]) ** 2, axis=2)), axis=1)
    return argrelextrema(G(kinetic_vel, 5), np.less)


def beat_align(m_beats, d_beats):
    """LODGE BA(); NaN when the clip has no music beat, 0 when the dance has no beat (upstream would crash)."""
    if len(m_beats) == 0:
        return float('nan')
    if len(d_beats[0]) == 0:
        return 0.0
    return float(np.mean([np.exp(-np.min((d_beats[0] - bb) ** 2) / 2 / 9) for bb in m_beats]))


def bas(joints, wav, sr):
    """BAS of one clip; like LODGE M2DMetrics, dance beats are taken on the root-relative joints."""
    return beat_align(music_beats(wav, sr), dance_beats(root_relative(joints)))


# ---------------------------------------------------------------- foot skating ratio (LODGE)
def foot_skating_ratio(joints, ground_height=0.0, ankle_threshold=0.08, toe_threshold=0.05):
    """Share of foot-contact frames (ankle <= 8 cm and toe <= 5 cm) whose foot slides > 1 cm per frame;
    numpy port of LODGE calculate_foot_skating_ratio (global joints, y-up, 30 fps)."""
    l_ankle, r_ankle, l_foot, r_foot = 7, 8, 10, 11
    j = np.asarray(joints, dtype=np.float32)
    p = j[:, [l_ankle, r_ankle, l_foot, r_foot], :]
    vel = np.zeros_like(p)
    vel[:-1] = p[1:] - p[:-1]
    ratios = []
    for a, t in ((0, 2), (1, 3)):  # left (ankle, toe), right (ankle, toe)
        contact = (p[:, a, 1] <= ground_height + ankle_threshold) & (p[:, t, 1] <= ground_height + toe_threshold)
        v = vel[:, [a, t], :].copy()
        v[~contact] = 0
        tangent = np.abs(np.mean(v[:, :, [0, 2]], axis=-1))  # upstream: |mean(vx, vz)|, kept
        slide = np.any(tangent > 0.01, axis=-1)
        ratios.append(float(slide.sum() / (contact.sum() + 1e-9)))
    return (ratios[0] + ratios[1]) / 2.0


# ---------------------------------------------------------------- one set of clips -> all metrics
class ClipSet:
    """Per-clip window features, style encodings, foot skating and BAS of a list of clips (computed once, so
    split-half noise floors can re-aggregate any subset of clips)."""

    bas_error = None

    def __init__(self, joints, wavs=None, sr=32000, encoder=None):
        self.n = len(joints)
        self.kin, self.geo, self.fsr, self.bas = [], [], [], []
        for i, j in enumerate(joints):
            k, g = window_features(root_relative(j))
            self.kin.append(k)
            self.geo.append(g)
            self.fsr.append(foot_skating_ratio(j))
            self.bas.append(self._bas(j, wavs[i], sr) if wavs is not None else float('nan'))
        self.prob = self.emb = self.owner = None
        if encoder is not None:
            e = encoder.encode(joints)
            self.prob, self.emb, self.owner = e['prob'], e['emb'], e['owner']

    @staticmethod
    def _bas(j, wav, sr):
        try:
            return bas(j, wav, sr)
        except Exception as e:  # BAS is a sanity metric: report the failure once and keep the other metrics
            if ClipSet.bas_error is None:
                ClipSet.bas_error = repr(e)
                print(f'WARNING: BAS failed and is reported as missing: {e!r}', flush=True)
            return float('nan')

    def windows(self, clips=None):
        clips = range(self.n) if clips is None else clips
        return np.concatenate([self.kin[c] for c in clips]), np.concatenate([self.geo[c] for c in clips])

    def style(self, clips=None):
        sel = np.ones(len(self.owner), bool) if clips is None else np.isin(self.owner, list(clips))
        return self.prob[sel], self.emb[sel]


def set_metrics(cs, family, norm, ref=None, encoder=None, clips=None):
    """All metrics of clips `clips` (default all) of ClipSet `cs`, scored as music of style `family`.
    norm: frozen normalisation; ref: {'kin','geo','emb'} of the session's real dance (FID / StyleFD skipped if None)."""
    kin, geo = cs.windows(clips)
    keep = norm['geo_keep']
    nk = (kin - norm['kin_mean']) / norm['kin_std']
    ng = (geo[:, keep] - norm['geo_mean']) / norm['geo_std']
    idx = range(cs.n) if clips is None else clips
    out = {'Div_k': mean_pairwise_distance(nk), 'Div_g': mean_pairwise_distance(ng),
           'BAS': float(np.nanmean([cs.bas[c] for c in idx])) if not np.isnan([cs.bas[c] for c in idx]).all() else float('nan'),
           'FSR': float(np.mean([cs.fsr[c] for c in idx])), 'n_clips': len(idx), 'n_windows': int(len(kin))}
    if ref is not None:
        out['FID_k'] = frozen_fid(ref['kin'], kin, norm['kin_mean'], norm['kin_std'])
        out['FID_g'] = frozen_fid(ref['geo'][:, keep], geo[:, keep], norm['geo_mean'], norm['geo_std'])
    if encoder is not None and cs.prob is not None:
        prob, emb = cs.style(clips)
        out.update(encoder.scores(prob, family))
        if ref is not None:
            out['StyleFD'] = frechet_distance(*activation_statistics(ref['emb']), *activation_statistics(emb))
    return out
