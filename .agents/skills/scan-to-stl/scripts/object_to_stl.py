#!/usr/bin/env python3
"""Turn a 3D Model Anything object scan into a watertight, millimetre-scaled STL for 3D printing.

Uses everything the phone recorded about where each measurement was taken:
  - LiDAR depth + per-pixel confidence (depth/frames.json), fused into a signed-distance volume
  - the ARKit camera pose (x, y, z + orientation) and intrinsics of every depth frame
  - the object box and support plane picked on the phone (manifest.json), or found automatically
  - IMU rotation rates (motion/imu.f64) to drop frames smeared by fast motion
  - the capture mat's ARKit scale estimate and ARKit feature points, as scale/consistency checks

  1. filter depth: confidence, range, flying pixels at silhouettes, fast-motion frames
  2. object frame: support plane (RANSAC on the depth) + gravity, so the base prints flat
  3. TSDF fusion with free-space carving; space no frame ever saw (inside, underneath) is solid
  4. pose refinement: Gauss-Newton alignment of every frame to the fused SDF, then refuse
  5. solid -> marching tetrahedra (closed and manifold by construction) -> Taubin smoothing
  6. millimetres, Z up, base on Z = 0 -> binary STL + PLY + report.json + preview.png

usage: object_to_stl.py <scan> [-o outdir] [--voxel-mm 1] [--calibrate x=52.3] [--auto-box] ...

Works on object-capture/1 scans and on house-capture/5 sessions (pass --room; the object
is found where the cameras converge). Needs numpy; Pillow for preview.png.
"""
import argparse
import json
import math
import os
import re
import struct
import sys
import time
import zlib

import numpy as np

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # camera: +y up/-z forward -> +y down/+z forward
F32 = np.float32


def mat(cm):
    return np.array(cm, dtype=np.float64).reshape(4, 4).T  # column-major -> row-major


def column_major(m):
    return [round(float(x), 7) for x in np.asarray(m, dtype=np.float64).T.ravel()]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def rodrigues(w):
    a = float(np.linalg.norm(w))
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def rotation_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


# --- scan input -------------------------------------------------------------------------

class Frame:
    """One depth frame, prepared for fusion (depth maps stored as float16 like the source)."""

    def __init__(self, record, cam_to_world, K, surf, carve, weight):
        self.record = record
        self.name = record['file']
        self.timestamp = record.get('timestamp', 0.0)
        self.cam_to_world = cam_to_world  # 4x4, CV camera convention
        self.K = K
        self.h, self.w = surf.shape
        self.surf = surf.astype(np.float16)    # depth used for the surface band (0 = unusable)
        self.carve = carve.astype(np.float16)  # nearest depth around each pixel, for free space
        self.weight = weight.astype(np.float16)
        self.T = None   # camera -> object frame (refined)
        self.T0 = None  # camera -> object frame (ARKit)


def read_depth(folder, record):
    w, h = record['width'], record['height']
    with open(os.path.join(folder, record['file']), 'rb') as f:
        raw = f.read()
    if record['file'].endswith('.f16z') and len(raw) != w * h * 2:  # the app stores raw Float16 if compression fails
        raw = zlib.decompress(raw, -15)  # raw DEFLATE (Apple's .zlib compression)
    depth = np.frombuffer(raw, dtype='<f2').astype(F32).reshape(h, w)
    conf_path = os.path.join(folder, record.get('confidence') or '')
    if record.get('confidence') and os.path.exists(conf_path):
        conf = np.fromfile(conf_path, dtype=np.uint8).reshape(h, w)
    else:
        conf = np.full((h, w), 2, np.uint8)
    return depth, conf


def prepare_depth(depth, conf, K, a):
    """Surface depth (trusted pixels away from silhouettes), conservative carving depth, weights."""
    z = np.where(np.isfinite(depth), depth, 0).astype(F32)
    valid = (z > a.min_depth) & (z < a.max_depth) & (conf >= min(1, a.min_confidence))
    z = np.where(valid, z, 0)
    # flying pixels: LiDAR mixes foreground and background along silhouettes
    p = np.pad(z, 1, mode='edge')
    jump = np.zeros_like(z)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        jump = np.maximum(jump, np.abs(z - p[dy:dy + z.shape[0], dx:dx + z.shape[1]]))
    edge = jump > 0.004 + 0.02 * z
    surf = np.where(valid & (conf >= a.min_confidence) & ~edge, z, 0)
    # carve only up to the nearest surface around each pixel: never into a silhouette
    q = np.pad(np.where(valid, z, np.inf), 1, mode='edge')
    carve = np.full_like(z, np.inf)
    for dy in range(3):
        for dx in range(3):
            carve = np.minimum(carve, q[dy:dy + z.shape[0], dx:dx + z.shape[1]])
    carve = np.where(valid & np.isfinite(carve), carve, 0)
    # weights: LiDAR noise grows with distance and at grazing angles
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(F32)
    X = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], -1)
    gx = np.zeros_like(X)
    gy = np.zeros_like(X)
    gx[:, 1:-1] = X[:, 2:] - X[:, :-2]
    gy[1:-1] = X[2:] - X[:-2]
    n = np.cross(gx, gy)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    ray = X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)
    cos = np.abs(np.sum(n * ray, -1))
    weight = np.clip((0.3 / np.maximum(z, 1e-3)) ** 2, 0.2, 2.0) * np.maximum(cos, 0.25)
    weight *= np.where(conf >= 2, 1.0, 0.4)
    return surf, carve, np.where(surf > 0, weight, 0).astype(F32)


class Scan:
    def __init__(self, folder, room=None):
        self.folder = folder
        path = os.path.join(folder, 'manifest.json')
        self.manifest = load_json(path) if os.path.exists(path) else {}
        self.format = self.manifest.get('format', 'unknown')
        frames_path = os.path.join(folder, self.manifest.get('depthFrames') or 'depth/frames.json')
        if not os.path.exists(frames_path):
            raise SystemExit(f'{frames_path} not found: this pipeline needs LiDAR depth frames')
        records = load_json(frames_path)
        if room is not None:
            records = [r for r in records if r.get('room', 0) == room]
        self.records = sorted(records, key=lambda r: r.get('timestamp', 0))
        self.motion = self._motion()

    def _motion(self):
        m = self.manifest.get('motion')
        if not m or not os.path.exists(os.path.join(self.folder, m['file'])):
            return None
        fields = m['fields']
        data = np.fromfile(os.path.join(self.folder, m['file']), dtype='<f8')
        data = data[:len(data) // len(fields) * len(fields)].reshape(-1, len(fields))
        return {name: data[:, i] for i, name in enumerate(fields)}

    def rotation_rate(self, t, window=1 / 30):
        """Peak gyro speed (rad/s) around time t, or None without IMU samples there."""
        if self.motion is None or 'rx' not in self.motion:
            return None
        ts = self.motion['t']
        i0, i1 = np.searchsorted(ts, [t - window, t + window])
        if i1 <= i0:
            return None
        r = np.sqrt(self.motion['rx'][i0:i1] ** 2 + self.motion['ry'][i0:i1] ** 2 + self.motion['rz'][i0:i1] ** 2)
        return float(r.max())


def select_frames(scan, a, stats):
    """Drop frames taken while the phone turned fast (gyro, else pose deltas); cap the count."""
    recs = scan.records
    keep = []
    prev = None
    for r in recs:
        rate = scan.rotation_rate(r.get('timestamp', 0))
        source = 'imu'
        if rate is None and prev is not None:
            dt = r.get('timestamp', 0) - prev.get('timestamp', 0)
            if 0 < dt < 0.5:
                R = mat(r['transform'])[:3, :3].T @ mat(prev['transform'])[:3, :3]
                rate, source = math.radians(rotation_deg(R)) / dt, 'poses'
        prev = r
        if rate is not None and rate > a.max_rotation_rate:
            stats['motion'] += 1
            stats['motionSource'] = source
            continue
        keep.append(r)
    if len(keep) > a.max_frames:
        idx = np.linspace(0, len(keep) - 1, a.max_frames).round().astype(int)
        stats['subsampled'] = len(keep) - len(idx)
        keep = [keep[i] for i in idx]
    frames = []
    for r in keep:
        try:
            depth, conf = read_depth(scan.folder, r)
        except (OSError, zlib.error, ValueError):  # missing or truncated file (e.g. storage ran out mid-scan)
            stats['unreadable'] = stats.get('unreadable', 0) + 1
            continue
        K = np.array(r['intrinsics'], dtype=np.float64).reshape(3, 3).T
        surf, carve, weight = prepare_depth(depth, conf, K, a)
        if (surf > 0).sum() < 200:
            stats['noDepth'] += 1
            continue
        frames.append(Frame(r, mat(r['transform']) @ ARKIT_TO_CV, K, surf, carve, weight))
    return frames


def frame_points(fr, stride=2, use='surf'):
    d = (fr.surf if use == 'surf' else fr.carve).astype(F32)[::stride, ::stride]
    ys, xs = np.mgrid[0:fr.h:stride, 0:fr.w:stride]
    ok = d > 0
    z = d[ok]
    pts = np.c_[(xs[ok] - fr.K[0, 2]) * z / fr.K[0, 0], (ys[ok] - fr.K[1, 2]) * z / fr.K[1, 1], z]
    return pts, fr.weight.astype(F32)[::stride, ::stride][ok]


def transform(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


# --- object frame -----------------------------------------------------------------------

def ransac_plane(P, thresh, rng, up=None, max_tilt_deg=None, above=None, iters=400):
    """Plane n.p = d with the most inliers (optionally near-horizontal, below a point)."""
    if len(P) < 50:
        return None
    best, best_count = None, 0
    min_cos = math.cos(math.radians(max_tilt_deg)) if max_tilt_deg is not None else -1
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if up is not None:
            if n @ up < 0:
                n = -n
            if n @ up < min_cos:
                continue
        d = n @ s[0]
        if above is not None and n @ above - d < 0.003:
            continue
        count = int((np.abs(P @ n - d) < thresh).sum())
        if count > best_count:
            best, best_count = (n, d), count
    if best is None:
        return None
    n, d = best
    for _ in range(3):  # least-squares refit on the inliers
        inl = P[np.abs(P @ n - d) < thresh]
        c = inl.mean(0)
        n2 = np.linalg.svd(inl - c, full_matrices=False)[2][-1]
        if n2 @ n < 0:
            n2 = -n2
        n, d = n2, n2 @ c
    res = P @ n - d
    inl = np.abs(res) < thresh
    return {'normal': n, 'd': float(d), 'inliers': int(inl.sum()), 'rmsMM': float(np.sqrt(np.mean(res[inl] ** 2)) * 1000)}


def frame_on_plane(n, origin, x_hint):
    """4x4 whose Y axis is the plane normal, origin on the plane, X close to x_hint."""
    x = x_hint - (x_hint @ n) * n
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0]) - n[0] * n
    x /= np.linalg.norm(x)
    z = np.cross(x, n)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, n, z, origin
    return M


def sample_world_points(frames, every=3, stride=4):
    pts = []
    for fr in frames[::every]:
        p, _ = frame_points(fr, stride)
        pts.append(transform(fr.cam_to_world, p))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def look_at_point(frames, max_depth=1.0, tol=0.05, iters=2000, seed=0):
    """Point most optical axes pass within `tol` of, in front of the camera and in depth range,
    refined by least squares over those frames: where an orbit was aimed, even when a room
    session also holds frames facing walls and furniture."""
    C = np.array([fr.cam_to_world[:3, 3] for fr in frames])
    F = np.array([fr.cam_to_world[:3, 2] for fr in frames])

    def lsq(m):
        P = np.eye(3)[None] - F[m, :, None] * F[m, None, :]
        return np.linalg.solve(P.sum(0) + 1e-9 * np.eye(3), np.einsum('nij,nj->i', P, C[m]))

    def support(x):
        v = x - C
        along = (v * F).sum(1)
        dist = np.linalg.norm(v - along[:, None] * F, axis=1)
        return (dist < tol) & (along > 0.05) & (along < max_depth + tol)

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters if len(frames) > 2 else 0):
        i, j = rng.choice(len(frames), 2, replace=False)
        b, w = F[i] @ F[j], C[i] - C[j]
        den = 1 - b * b
        if den < 1e-6:
            continue
        s = (b * (F[j] @ w) - F[i] @ w) / den
        t = (F[j] @ w - b * (F[i] @ w)) / den
        p, q = C[i] + s * F[i], C[j] + t * F[j]
        if np.linalg.norm(p - q) < 2 * tol:
            m = support((p + q) / 2)
            if best is None or m.sum() > best.sum():
                best = m
    if best is None or best.sum() < 10:
        best = np.ones(len(frames), bool)
    return lsq(best)


def object_blob(P, seed_lo, seed_hi, limit_lo, limit_hi, cell=0.004, min_count=3, min_height=0.004):
    """Extent of what stands on the plane (object frame, +Y up) around a seed rectangle: points
    above `min_height`, binned in `cell` squares, keeping the connected footprint blobs that touch
    the seed rectangle. Returns (lo, hi) with lo[1] = 0, or None."""
    inside = ((P[:, 1] > min_height) & (P[:, 1] < 0.5) & (P[:, 0] > limit_lo[0]) & (P[:, 0] < limit_hi[0])
              & (P[:, 2] > limit_lo[2]) & (P[:, 2] < limit_hi[2]))
    obj = P[inside]
    if len(obj) < 50:
        return None
    ij = np.floor((obj[:, [0, 2]] - limit_lo[[0, 2]]) / cell).astype(int)
    grid = np.zeros(ij.max(0) + 1, np.int32)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    labels = label_components(grid >= min_count)
    lab = labels[ij[:, 0], ij[:, 1]]
    seed = (lab >= 0) & np.all(obj[:, [0, 2]] > seed_lo[[0, 2]], 1) & np.all(obj[:, [0, 2]] < seed_hi[[0, 2]], 1)
    if not seed.any():
        return None
    o = obj[np.isin(lab, np.unique(lab[seed]))]
    lo, hi = np.percentile(o, 0.5, axis=0), np.percentile(o, 99.5, axis=0)
    lo[1] = 0.0
    return lo, hi


def object_frame(scan, frames, a, rng, info):
    """object -> world 4x4 (origin on the support plane, +Y up) and the object's extent in that
    frame as (lo, hi) corners in metres (lo[1] = 0, the plane)."""
    world_pts = sample_world_points(frames)
    mat_info = scan.manifest.get('mat') or {}
    if mat_info:
        info['mat'] = {k: mat_info.get(k) for k in ('physicalWidth', 'estimatedScaleFactor', 'samples', 'scaleSpread')}
    box = scan.manifest.get('box')
    if box and not a.auto_box:
        T = mat(box['transform'])
        size = np.array(box['size'], dtype=np.float64)
        info['box'] = 'manifest'
        info['planeSource'] = (scan.manifest.get('supportPlane') or {}).get('source', 'manifest')
        if a.box_size_mm:
            size = np.array([float(v) for v in a.box_size_mm.split(',')]) / 1000
            info['box'] = 'manifest frame, --box-size-mm'
        # ARKit plane anchors can sit millimetres off the real table: refit on the depth
        P = transform(np.linalg.inv(T), world_pts)
        near = (np.abs(P[:, 1]) < 0.012) & (np.abs(P[:, 0]) < size[0] / 2 + 0.06) & (np.abs(P[:, 2]) < size[2] / 2 + 0.06)
        plane = ransac_plane(P[near], 0.0015, rng, up=np.array([0, 1.0, 0]), max_tilt_deg=8)
        if plane and plane['inliers'] > 300:
            n, d = plane['normal'], plane['d']
            T = T @ frame_on_plane(n, d * n, np.array([1.0, 0, 0]))
            info['planeRefit'] = {'shiftMM': round(d * 1000, 2), 'tiltDeg': round(math.degrees(math.acos(min(1, n[1]))), 3),
                                  'rmsMM': round(plane['rmsMM'], 2), 'inliers': plane['inliers']}
            info['planeSource'] += ' + depth refit'
        else:
            info['planeRefit'] = None
        lo, hi = np.array([-size[0] / 2, 0, -size[2] / 2]), np.array([size[0] / 2, size[1], size[2] / 2])
        # the phone sizes the box from the one depth frame at the tap, which cannot see the far side:
        # grow it to whatever the whole scan saw standing on the plane and touching the box
        if a.grow_mm > 0 and info['planeRefit']:
            g = a.grow_mm / 1000
            blob = object_blob(transform(np.linalg.inv(T), world_pts), lo, hi, lo - g, hi + g)
            if blob is not None:
                new_lo, new_hi = np.minimum(lo, blob[0]), np.maximum(hi, blob[1])
                grown = np.r_[lo - new_lo, new_hi - hi] * 1000
                if grown.max() > 2:
                    info['boxGrowthMM'] = {k: round(float(v), 1) for k, v in zip(('-x', '-y', '-z', '+x', '+y', '+z'), grown) if v > 2}
                    lo, hi = new_lo, new_hi
        return T, lo, hi
    # no box: find the object where the cameras converge, on the plane below that point
    target = look_at_point(frames, a.max_depth)
    if a.center:
        target = np.array([float(v) for v in a.center.split(',')])
    radius = a.radius_mm / 1000
    near = world_pts[np.linalg.norm(world_pts - target, axis=1) < radius * 1.6]
    plane = ransac_plane(near, 0.002, rng, up=np.array([0, 1.0, 0]), max_tilt_deg=15, above=target)
    if plane is None:
        raise SystemExit('could not find a support plane under the object: pass --center x,y,z (world metres)')
    n, d = plane['normal'], plane['d']
    T = frame_on_plane(n, target - (target @ n - d) * n, np.array([1.0, 0, 0]))
    P = transform(np.linalg.inv(T), near)
    # footprint: the connected blob(s) at the aim point
    r = np.array([radius, 0.5, radius])
    blob = object_blob(P, np.array([-0.01, 0, -0.01]), np.array([0.01, 0, 0.01]), -r, r)
    if blob is None:  # nothing right under the aim point: take the blob nearest to it
        obj = P[(P[:, 1] > 0.004) & (np.hypot(P[:, 0], P[:, 2]) < radius)]
        if len(obj) < 100:
            raise SystemExit('no object points above the support plane near the aim point: pass --center or --radius-mm')
        c = obj[np.argmin(np.hypot(obj[:, 0], obj[:, 2]))]
        blob = object_blob(P, c - 0.004, c + 0.004, -r, r)
    if blob is None:
        raise SystemExit('object points are too sparse to outline: pass --center or --radius-mm')
    lo, hi = blob
    center = np.array([(lo[0] + hi[0]) / 2, 0, (lo[2] + hi[2]) / 2])
    T = T @ frame_on_plane(np.array([0, 1.0, 0]), center, np.array([1.0, 0, 0]))
    info['box'] = 'auto'
    info['planeSource'] = 'depth RANSAC'
    info['planeRefit'] = {'rmsMM': round(plane['rmsMM'], 2), 'inliers': plane['inliers']}
    return T, lo - center, hi - center


# --- connected components ---------------------------------------------------------------

def union_find(n, u, v):
    """Component root per node of a graph (vectorized hook + pointer jumping)."""
    parent = np.arange(n, dtype=np.int64)
    u = np.asarray(u, np.int64)
    v = np.asarray(v, np.int64)
    while len(u):
        pu, pv = parent[u], parent[v]
        diff = pu != pv
        if not diff.any():
            break
        u, v, pu, pv = u[diff], v[diff], pu[diff], pv[diff]
        np.minimum.at(parent, np.maximum(pu, pv), np.minimum(pu, pv))
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
    return parent


def label_components(mask):
    """Face-connected components of a boolean grid; -1 = background."""
    flat = mask.ravel()
    idx = np.flatnonzero(flat)
    pos = np.full(flat.size, -1, np.int64)
    pos[idx] = np.arange(len(idx))
    strides = [int(np.prod(mask.shape[k + 1:])) for k in range(mask.ndim)]
    us, vs = [], []
    for axis in range(mask.ndim):
        a = [slice(None)] * mask.ndim
        b = [slice(None)] * mask.ndim
        a[axis], b[axis] = slice(0, -1), slice(1, None)
        coords = np.nonzero(mask[tuple(a)] & mask[tuple(b)])
        first = np.ravel_multi_index(coords, mask.shape)
        us.append(pos[first])
        vs.append(pos[first + strides[axis]])
    roots = union_find(len(idx), np.concatenate(us) if us else [], np.concatenate(vs) if vs else [])
    labels = np.full(flat.size, -1, np.int64)
    labels[idx] = roots
    return labels.reshape(mask.shape)


# --- TSDF volume ------------------------------------------------------------------------

class Volume:
    def __init__(self, lo, hi, voxel, trunc):
        self.voxel = voxel
        self.trunc = trunc
        self.dims = (np.ceil((hi - lo) / voxel).astype(int) + 1)
        self.origin = lo.astype(np.float64)
        nx, ny, nz = self.dims
        self.axes = [(self.origin[k] + np.arange(self.dims[k]) * voxel).astype(F32) for k in range(3)]
        n = nx * ny * nz
        self.tsdf = np.zeros(n, F32)
        self.weight = np.zeros(n, F32)       # all observations (surface band + free space)
        self.band = np.zeros(n, F32)         # surface-band observations only
        self.free = np.zeros(n, np.uint16)   # frames that saw straight through the voxel

    @property
    def size(self):
        return int(np.prod(self.dims))

    def integrate(self, fr, T, free_weight=0.5, min_depth=0.05):
        """Projective TSDF update from one frame; T = camera -> object."""
        Ti = np.linalg.inv(T)
        R, t = Ti[:3, :3].astype(F32), Ti[:3, 3].astype(F32)
        fx, fy, cx, cy = (F32(v) for v in (fr.K[0, 0], fr.K[1, 1], fr.K[0, 2], fr.K[1, 2]))
        X, Y, Z = self.axes
        nx, ny, nz = self.dims
        surf, carve, weight = (m.astype(F32).ravel() for m in (fr.surf, fr.carve, fr.weight))
        trunc = F32(self.trunc)
        step = max(1, int(4_000_000 // (ny * nz)))  # x-slabs bound the temporaries
        for a in range(0, nx, step):
            xs = X[a:a + step, None, None]
            zc = (R[2, 0] * xs + R[2, 1] * Y[None, :, None] + R[2, 2] * Z[None, None, :] + t[2]).ravel()
            front = np.flatnonzero(zc > min_depth)
            if len(front) == 0:
                continue
            xc = (R[0, 0] * xs + R[0, 1] * Y[None, :, None] + R[0, 2] * Z[None, None, :] + t[0]).ravel()[front]
            yc = (R[1, 0] * xs + R[1, 1] * Y[None, :, None] + R[1, 2] * Z[None, None, :] + t[1]).ravel()[front]
            z = zc[front]
            u = np.rint(fx * xc / z + cx)
            v = np.rint(fy * yc / z + cy)
            inside = (u >= 0) & (u < fr.w) & (v >= 0) & (v < fr.h)
            sel = front[inside] + a * ny * nz
            z = z[inside]
            pix = (v[inside] * fr.w + u[inside]).astype(np.int64)
            ds = surf[pix]
            dc = carve[pix]
            sdf = ds - z
            band = (ds > 0) & (sdf >= -trunc) & (sdf <= trunc)
            free = ~band & (dc > 0) & (dc - z > trunc)
            i = sel[band]
            w = weight[pix[band]]
            W = self.weight[i]
            self.tsdf[i] = (self.tsdf[i] * W + (sdf[band] / trunc) * w) / (W + w)
            self.weight[i] = W + w
            self.band[i] += w
            j = sel[free]
            W = self.weight[j]
            self.tsdf[j] = (self.tsdf[j] * W + free_weight) / (W + free_weight)
            self.weight[j] = W + free_weight
            self.free[j] = np.minimum(self.free[j].astype(np.int32) + 1, 65535).astype(np.uint16)

    def grid(self, arr):
        return arr.reshape(tuple(self.dims))

    def sampler(self):
        """Stacked (tsdf, weight, grad) per voxel for trilinear lookups during alignment."""
        t = self.grid(self.tsdf)
        g = np.gradient(t, self.voxel)
        return np.stack([self.tsdf, self.weight] + [x.ravel() for x in g], -1).astype(F32)

    def sample(self, S, p):
        g = (p - self.origin) / self.voxel
        i0 = np.floor(g).astype(np.int64)
        ok = np.all((i0 >= 0) & (i0 < self.dims - 1), axis=1)
        i0, f = i0[ok], (g[ok] - i0[ok]).astype(F32)
        nyz, nz = self.dims[1] * self.dims[2], self.dims[2]
        base = i0[:, 0] * nyz + i0[:, 1] * nz + i0[:, 2]
        out = np.zeros((len(base), S.shape[1]), F32)
        wmin = np.full(len(base), np.inf, F32)
        for dx in (0, 1):
            fx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                fy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    fz = f[:, 2] if dz else 1 - f[:, 2]
                    vals = S[base + dx * nyz + dy * nz + dz]
                    out += (fx * fy * fz)[:, None] * vals
                    wmin = np.minimum(wmin, vals[:, 1])
        return ok, out, wmin


def align_frame(pts, T, vol, S, center, iters=10):
    """Gauss-Newton: move the frame's points onto the model's zero level set (point-to-SDF)."""
    trunc = vol.trunc
    T = T.copy()
    rms0 = rms = None
    for it in range(iters):
        p = transform(T, pts)
        ok, s, wmin = vol.sample(S, p)
        val = s[:, 0] * trunc
        grad = s[:, 2:5] * trunc
        gn = np.linalg.norm(grad, axis=1)
        use = (wmin > 0) & (np.abs(s[:, 0]) < 0.95) & (gn > 0.3) & (gn < 3)
        if use.sum() < 150:
            return None, rms0, rms
        q = p[ok][use] - center
        r = val[use].astype(np.float64)
        g = grad[use].astype(np.float64)
        k = 0.3 * trunc
        w = np.where(np.abs(r) < k, 1.0, k / np.maximum(np.abs(r), 1e-12))
        rms = float(np.sqrt(np.sum(w * r * r) / np.sum(w)))
        if rms0 is None:
            rms0 = rms
        J = np.c_[np.cross(q, g), g]
        H = (J * w[:, None]).T @ J
        b = (J * w[:, None]).T @ r
        H += np.eye(6) * (1e-4 * np.trace(H) / 6 + 1e-12)
        delta = -np.linalg.solve(H, b)
        D = np.eye(4)
        D[:3, :3] = rodrigues(delta[:3])
        D[:3, 3] = center - D[:3, :3] @ center + delta[3:]
        T = D @ T
        if np.linalg.norm(delta[3:]) < 2e-6 and np.linalg.norm(delta[:3]) < 2e-5:
            break
    return T, rms0, rms


def correction(T, T0, point):
    """How far a refined pose moves what the frame saw at `point` (m), and its rotation (deg)."""
    C = T @ np.linalg.inv(T0)  # object -> object: where ARKit put the frame's points -> where they fit
    return float(np.linalg.norm(transform(C, point[None])[0] - point)), rotation_deg(C[:3, :3])


def refit_plane(frames, lo, hi, rng):
    """Support plane from the refined frames' table points; returns new-object -> old-object."""
    pts = []
    for fr in frames[::2]:
        p, _ = frame_points(fr, 4)
        p = transform(fr.T, p)
        pts.append(p[(np.abs(p[:, 1]) < 0.008) & np.all((p[:, [0, 2]] > lo[[0, 2]] - 0.05) & (p[:, [0, 2]] < hi[[0, 2]] + 0.05), 1)])
    plane = ransac_plane(np.concatenate(pts), 0.0015, rng, up=np.array([0, 1.0, 0]), max_tilt_deg=5)
    if not plane or plane['inliers'] < 300:
        return None, None
    n, d = plane['normal'], plane['d']
    return frame_on_plane(n, d * n, np.array([1.0, 0, 0])), {
        'shiftMM': round(d * 1000, 2), 'tiltDeg': round(math.degrees(math.acos(min(1, n[1]))), 3), 'rmsMM': round(plane['rmsMM'], 2)}


def refine_poses(frames, lo, hi, a, info):
    voxel = max(2 * a.voxel_mm / 1000, 0.002)
    trunc = max(3 * voxel, a.trunc_mm / 1000 * 1.5)
    center = (lo + hi) / 2
    passes = []
    for p in range(a.refine):
        t0 = time.time()
        vol = Volume(lo, hi, voxel, trunc)
        for fr in frames:
            vol.integrate(fr, fr.T, min_depth=a.min_depth)
        S = vol.sampler()
        before, after, shifts, turns, dropped = [], [], [], [], 0
        for fr in frames:
            pts, _ = frame_points(fr, 2)
            local = transform(fr.T, pts)
            keep = np.all((local > lo) & (local < hi), axis=1)
            T, r0, r1 = align_frame(pts[keep], fr.T, vol, S, center)
            if T is None:
                continue
            # a correction ARKit could not plausibly need means the alignment slipped
            moved, turn = correction(T, fr.T0, center)
            if moved > a.max_shift_mm / 1000 or turn > a.max_turn_deg:
                dropped += 1
                fr.dropped = True
                continue
            fr.T = T
            before.append(r0)
            after.append(r1)
        frames[:] = [fr for fr in frames if not getattr(fr, 'dropped', False)]
        for fr in frames:
            moved, turn = correction(fr.T, fr.T0, center)
            shifts.append(moved * 1000)
            turns.append(turn)
        passes.append({'residualMM': {'before': round(float(np.median(before)) * 1000, 3) if before else None,
                                      'after': round(float(np.median(after)) * 1000, 3) if after else None},
                       'droppedFrames': dropped, 'seconds': round(time.time() - t0, 1)})
        print(f'  refine pass {p + 1}: median residual {passes[-1]["residualMM"]["before"]} -> '
              f'{passes[-1]["residualMM"]["after"]} mm, dropped {dropped} ({passes[-1]["seconds"]}s)')
    if frames and passes:
        info['poseRefinement'] = {
            'passes': passes,
            'medianShiftMM': round(float(np.median(shifts)), 3), 'maxShiftMM': round(float(np.max(shifts)), 3),
            'medianTurnDeg': round(float(np.median(turns)), 4), 'maxTurnDeg': round(float(np.max(turns)), 4)}


# --- solid + mesh -----------------------------------------------------------------------

def solid_field(vol, a, info):
    """Signed field (< 0 inside) whose sign is a single closed, cavity-free solid."""
    t = vol.grid(vol.tsdf).copy()
    seen = vol.grid((vol.band > 0) | (vol.free >= a.min_free))
    t[~seen] = -1  # never seen through: inside the object or underneath it
    solid = t < 0
    # the table: everything lower than the cut height above the support plane goes
    ys = vol.axes[1]
    cut_layer = int(np.searchsorted(ys, a.cut_mm / 1000))
    solid[:, :cut_layer] = False
    for axis in range(3):  # a closed border of air
        idx = [slice(None)] * 3
        idx[axis] = 0
        solid[tuple(idx)] = False
        idx[axis] = -1
        solid[tuple(idx)] = False
    labels = label_components(solid)
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(ids) == 0:
        raise SystemExit('nothing solid left above the support plane: check the box, --cut-mm and the depth range')
    keep = ids[np.argmax(counts)]
    info['solidComponents'] = {'found': int(len(ids)), 'keptVoxels': int(counts.max()),
                               'droppedVoxels': int(counts.sum() - counts.max())}
    solid = labels == keep
    air = label_components(~solid)
    outside = air[0, 0, 0]
    voids = (~solid) & (air != outside)
    info['filledVoidVoxels'] = int(voids.sum())
    solid |= voids
    # cells whose sign was forced (cut, border, dropped parts, filled voids) get a clear value, so
    # the surface crosses well inside the cell instead of in micron-thin slivers
    forced = solid != (t < 0)
    field = np.where(solid, np.where(forced, -0.5, np.minimum(t, -1e-3)),
                     np.where(forced, 0.5, np.maximum(t, 1e-3))).astype(F32)
    # the outermost solid cells (not the base, which is always the flat extension to the table)
    # that no depth frame measured: filled in, not scanned. The interior never counts.
    shell = np.zeros_like(solid)
    for axis in range(3):
        for shift in (1, -1):
            shell |= np.roll(~solid, shift, axis)  # the border is air, so wrapping is harmless
    shell &= solid
    if cut_layer < solid.shape[1]:
        shell[:, cut_layer] = False
    info['unobservedSurfaceFraction'] = round(float((shell & ~seen).sum() / max(1, shell.sum())), 4)
    # the object ran into a side or the top of the volume and was cut flat there
    walls = {'-x': solid[1], '+x': solid[-2], '-z': solid[:, :, 1], '+z': solid[:, :, -2], 'top': solid[:, -2]}
    info['boxWallContact'] = [k for k, v in walls.items() if v.sum() > 20]
    return field, ys[cut_layer] if cut_layer < len(ys) else ys[-1]


def thin_fraction(solid, r):
    """Share of the solid that a cube of 2r+1 cells cannot reach (removed by a morphological opening):
    walls and parts thinner than that."""
    m = np.pad(solid, r)

    def morph(m, op):
        for axis in range(3):
            out = m.copy()
            for k in range(1, r + 1):
                out = op(out, op(np.roll(m, k, axis), np.roll(m, -k, axis)))
            m = out
        return m

    opened = morph(morph(m, np.logical_and), np.logical_or)
    return float((m & ~opened).sum() / max(1, m.sum()))


# Kuhn split of a cube into 6 tetrahedra around its main diagonal (corner c = x + 2y + 4z):
# the same split in every cube makes a consistent tetrahedral mesh, so the iso-surface is
# closed and manifold with no ambiguous cases.
TETS = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
CORNERS = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)], dtype=np.float64)


def tet_cases():
    cases = {}
    for code in range(1, 15):
        inside = [i for i in range(4) if code >> i & 1]
        out = [i for i in range(4) if not code >> i & 1]
        if len(inside) == 1:
            k = inside[0]
            cases[code] = [((k, out[0]), (k, out[1]), (k, out[2]))]
        elif len(inside) == 3:
            k = out[0]
            cases[code] = [((k, inside[0]), (k, inside[1]), (k, inside[2]))]
        else:
            p, q = inside
            r, s = out
            cases[code] = [((p, r), (p, s), (q, s)), ((p, r), (q, s), (q, r))]
    return cases


def marching_tetrahedra(field, origin, voxel):
    nx, ny, nz = field.shape
    f = field.ravel()
    N = f.size
    neg = field < 0
    any_neg = np.zeros((nx - 1, ny - 1, nz - 1), bool)
    all_neg = np.ones_like(any_neg)
    offsets = []
    for c in range(8):
        dx, dy, dz = (int(v) for v in CORNERS[c])
        corner = neg[dx:dx + nx - 1, dy:dy + ny - 1, dz:dz + nz - 1]
        any_neg |= corner
        all_neg &= corner
        offsets.append(dx * ny * nz + dy * nz + dz)
    offsets = np.array(offsets, np.int64)
    base = np.ravel_multi_index(np.nonzero(any_neg & ~all_neg), (nx, ny, nz))
    cases = tet_cases()
    keys, grads = [], []
    for tet in TETS:
        vids = base[:, None] + offsets[list(tet)][None, :]
        fv = f[vids]
        code = ((fv < 0) * (1 << np.arange(4))).sum(1)
        P = CORNERS[list(tet)]
        Minv = np.linalg.inv(P[1:] - P[0])  # gradient of the linear interpolant in this tet
        for c, tris in cases.items():
            rows = np.flatnonzero(code == c)
            if len(rows) == 0:
                continue
            g = (fv[rows, 1:] - fv[rows, :1]) @ Minv.T
            for tri in tris:
                k = np.empty((len(rows), 3), np.int64)
                for j, (p, q) in enumerate(tri):
                    a, b = vids[rows, p], vids[rows, q]
                    k[:, j] = np.minimum(a, b) * N + np.maximum(a, b)
                keys.append(k)
                grads.append(g)
    if not keys:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    keys = np.concatenate(keys)
    grads = np.concatenate(grads)
    uniq, inverse = np.unique(keys.ravel(), return_inverse=True)
    F = inverse.reshape(-1, 3).astype(np.int64)
    lo, hi = uniq // N, uniq % N
    t = (f[lo] / (f[lo] - f[hi]))[:, None]
    plo = np.c_[np.unravel_index(lo, field.shape)].astype(np.float64)
    phi = np.c_[np.unravel_index(hi, field.shape)].astype(np.float64)
    V = plo + t * (phi - plo)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = np.einsum('ij,ij->i', n, grads) < 0  # outward = towards increasing field
    F[flip] = F[flip][:, ::-1]
    return origin + V * voxel, F


def mesh_edges(F):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def taubin(V, F, iters, pinned, lam=0.5, mu=-0.53):
    """Volume-preserving smoothing of the voxel steps; pinned vertices keep their Y."""
    if iters <= 0:
        return V
    e = mesh_edges(F)
    n = len(V)
    deg = np.bincount(e.ravel(), minlength=n).astype(np.float64)
    deg[deg == 0] = 1
    V = V.copy()
    for _ in range(iters):
        for factor in (lam, mu):
            avg = np.stack([np.bincount(e[:, 0], V[e[:, 1], k], n) + np.bincount(e[:, 1], V[e[:, 0], k], n)
                            for k in range(3)], 1) / deg[:, None]
            y = V[pinned, 1].copy()
            V += factor * (avg - V)
            V[pinned, 1] = y
    return V


def mesh_report(V, F):
    nV, nF = len(V), len(F)
    directed = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und = np.sort(directed, axis=1)
    _, counts = np.unique(und[:, 0] * nV + und[:, 1], return_counts=True)
    oriented = len(np.unique(directed[:, 0] * nV + directed[:, 1])) == len(directed)
    closed = bool(np.all(counts == 2))
    roots = union_find(nV, directed[:, 0], directed[:, 1])
    components = len(np.unique(roots[np.unique(F)]))
    euler = nV - len(counts) + nF
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return {
        'vertices': nV, 'triangles': nF, 'watertight': closed, 'consistentOrientation': bool(oriented),
        'components': int(components), 'genus': int((2 * components - euler) // 2) if closed else None,
        'volumeMM3': float(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum() / 6),
        'areaMM2': float(np.linalg.norm(cross, axis=1).sum() / 2),
        'degenerateTriangles': int((np.linalg.norm(cross, axis=1) < 1e-12).sum()),
    }


# --- output -----------------------------------------------------------------------------

def robust_extent(P, F, k, base_mm=2.0, cone_deg=10.0, band_mm=2.0, q=0.002):
    """Extent along axis k (mm) as calipers would measure it, robust to LiDAR noise bumps: on each
    side, the area-weighted median position of the outermost surface facing that way (within
    `cone_deg`, within `band_mm` of the extreme), so a flat face counts at its mean position. Where
    nothing faces that way (a tip or an edge), the outermost `q` of the area is ignored instead.
    For X/Y the extended base flare is left out."""
    tri = P[F]
    c = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    n /= np.maximum(2 * area[:, None], 1e-20)
    keep = area > 0
    if k < 2 and np.ptp(P[:, 2]) > 4 * base_mm:
        keep &= c[:, 2] > P[:, 2].min() + base_mm
    ends = []
    for sign in (1.0, -1.0):
        x = sign * c[:, k]
        facing = keep & (sign * n[:, k] > math.cos(math.radians(cone_deg)))
        if facing.any():
            facing &= x > x[facing].max() - band_mm
        sel = facing if area[facing].sum() > 2.0 else keep
        xs, ws = x[sel], area[sel]
        o = np.argsort(xs)
        cw = np.cumsum(ws[o]) / ws.sum()
        ends.append(sign * xs[o][np.searchsorted(cw, 0.5 if sel is facing else 1 - q)])
    return float(ends[0] - ends[1])


def footprint_yaw(P):
    """Rotation about Z (rad) that makes the footprint's bounding rectangle smallest, long side on X."""
    xy = P[:: max(1, len(P) // 20000), :2]
    best = None
    for deg in np.arange(0, 90, 0.25):
        r = math.radians(deg)
        c, s = math.cos(r), math.sin(r)
        q = xy @ np.array([[c, s], [-s, c]])
        ext = q.max(0) - q.min(0)
        if best is None or ext[0] * ext[1] < best[0]:
            best = (ext[0] * ext[1], r, ext)
    _, r, ext = best
    return r + (math.pi / 2 if ext[1] > ext[0] else 0)


def write_stl(path, V, F, header):
    tri = V[F].astype(F32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(F), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    rec['n'], rec['v'] = n, tri
    with open(path, 'wb') as out:
        out.write(header.encode('ascii', 'replace')[:80].ljust(80, b' '))
        out.write(struct.pack('<I', len(F)))
        rec.tofile(out)


def read_stl(path):
    with open(path, 'rb') as f:
        f.seek(80)
        (n,) = struct.unpack('<I', f.read(4))
        rec = np.fromfile(f, dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')], count=n)
    return rec['v']


def write_ply(path, V, F):
    with open(path, 'wb') as out:
        out.write(f'ply\nformat binary_little_endian 1.0\ncomment units mm, Z up\nelement vertex {len(V)}\n'
                  'property float x\nproperty float y\nproperty float z\n'
                  f'element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n'.encode())
        V.astype('<f4').tofile(out)
        faces = np.zeros(len(F), dtype=[('n', 'u1'), ('i', '<i4', 3)])
        faces['n'], faces['i'] = 3, F
        faces.tofile(out)


def render_preview(path, V, F, size_mm, title):
    """Four shaded orthographic views (front, right, top, iso) with a millimetre scale bar."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    c = (V.min(0) + V.max(0)) / 2
    extent = float(np.max(V.max(0) - V.min(0)))
    px = 420
    # rows: screen right, screen up, towards the viewer
    views = {'front (-Y)': np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]),
             'right (+X)': np.array([[0, 1.0, 0], [0, 0, 1], [1, 0, 0]]),
             'top (+Z)': np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])}
    az, el = math.radians(35), math.radians(30)
    eye = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    right = np.cross([0, 0, 1.0], eye)
    right /= np.linalg.norm(right)
    views['iso'] = np.array([right, np.cross(eye, right), eye])
    paper, ink = (246, 243, 235), (30, 33, 30)
    sheet = Image.new('RGB', (px * 2, px * 2 + 40), paper)
    draw = ImageDraw.Draw(sheet)
    scale = px * 0.82 / max(extent, 1e-6)
    # splat a barycentric grid of points into every triangle, finer than a pixel, then z-buffer
    edge_px = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1) * scale
    steps = np.clip(np.ceil(edge_px * 1.5).astype(int), 1, 40)
    grids = {}
    for k, (label, M) in enumerate(views.items()):
        light = M[2] * 0.8 + M[1] * 0.35 + M[0] * 0.2
        light /= np.linalg.norm(light)
        shade = np.where(n @ M[2] > 0, 0.25 + 0.75 * np.clip(n @ light, 0, 1), 0.15)
        screen = (tri - c) @ M.T
        parts, cols = [], []
        for m in np.unique(steps):
            if m not in grids:
                i, j = np.meshgrid(np.arange(m + 1), np.arange(m + 1))
                keep = i + j <= m
                grids[m] = np.c_[i[keep], j[keep], m - i[keep] - j[keep]] / m
            rows = np.flatnonzero(steps == m)
            parts.append(np.einsum('sb,tbc->tsc', grids[m], screen[rows]).reshape(-1, 3))
            cols.append(np.repeat(shade[rows], len(grids[m])))
        pts, col = np.concatenate(parts), np.concatenate(cols)
        x = np.round(pts[:, 0] * scale + px / 2).astype(int)
        y = np.round(-pts[:, 1] * scale + px / 2).astype(int)
        ok = (x >= 0) & (x < px) & (y >= 0) & (y < px)
        pix, depth, col = (y * px + x)[ok], pts[ok, 2], col[ok]
        order = np.lexsort((-depth, pix))  # per pixel, nearest first
        pix, col = pix[order], col[order]
        first = np.r_[True, pix[1:] != pix[:-1]]
        tile = np.empty((px * px, 3), np.uint8)
        tile[:] = paper
        g = col[first, None] * np.array([235.0, 224, 204])
        tile[pix[first]] = g.clip(0, 255).astype(np.uint8)
        sheet.paste(Image.fromarray(tile.reshape(px, px, 3)), ((k % 2) * px, (k // 2) * px))
        draw.text(((k % 2) * px + 10, (k // 2) * px + 8), label, fill=ink)
    bar = 10 ** math.floor(math.log10(max(extent / 2, 1e-3)))
    y0 = px * 2 + 22
    draw.line([(12, y0), (12 + bar * scale, y0)], fill=(194, 71, 31), width=3)
    draw.text((20 + bar * scale, y0 - 7), f'{bar:g} mm', fill=ink)
    draw.text((140 + bar * scale, y0 - 7), f'{title}  {size_mm[0]:.1f} x {size_mm[1]:.1f} x {size_mm[2]:.1f} mm', fill=ink)
    sheet.save(path)
    return True


def view_coverage(frames, center):
    """Fraction of 12 azimuth x 3 elevation view cells around the object that some frame used."""
    cells = set()
    for fr in frames:
        d = fr.T[:3, 3] - center
        el = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
        az = math.degrees(math.atan2(d[2], d[0])) % 360
        cells.add((int(az // 30) % 12, min(2, max(0, int(el // 30)))))  # same cells as the app
    return len(cells) / 36


def feature_check(scan, obj_to_world, V_obj):
    """Median distance (mm) from ARKit's own 3D feature points inside the box to the surface."""
    feat = scan.manifest.get('features')
    if not feat or not os.path.exists(os.path.join(scan.folder, feat['file'])):
        return None
    tracks = {}
    for entry in load_json(os.path.join(scan.folder, feat['file'])):
        for pid, p in zip(entry.get('ids', []), entry.get('points', [])):
            tracks.setdefault(pid, []).append(p)
    pts = np.array([np.median(np.array(v), 0) for v in tracks.values() if len(v) >= 3])
    if len(pts) == 0:
        return None
    P = transform(np.linalg.inv(obj_to_world), pts)
    lo, hi = V_obj.min(0) - 0.003, V_obj.max(0) + 0.003
    P = P[np.all((P > lo) & (P < hi), axis=1)]
    if len(P) == 0:
        return None
    d = np.full(len(P), np.inf)
    sample = V_obj[:: max(1, len(V_obj) // 60000)]
    for i in range(0, len(P), 256):
        d[i:i + 256] = np.sqrt(((P[i:i + 256, None] - sample[None]) ** 2).sum(-1)).min(1)
    return {'points': int(len(P)), 'medianDistanceMM': round(float(np.median(d)) * 1000, 2),
            'p90DistanceMM': round(float(np.percentile(d, 90)) * 1000, 2)}


# --- main -------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('scan', help='object scan folder (manifest.json + depth/), or a house-capture session')
    ap.add_argument('-o', help='output folder (default <scan>/print)')
    ap.add_argument('--name', help='STL file name (default: the scan name)')
    ap.add_argument('--room', type=int, help='house-capture sessions: only this room index')
    ap.add_argument('--voxel-mm', type=float, default=1.0, help='fusion resolution (LiDAR pixels are ~1.5 mm at 30 cm)')
    ap.add_argument('--trunc-mm', type=float, default=0, help='TSDF truncation (default max(4 mm, 3 voxels))')
    ap.add_argument('--min-confidence', type=int, default=2, help='ARKit depth confidence for surfaces (0-2)')
    ap.add_argument('--min-depth', type=float, default=0.1, help='metres')
    ap.add_argument('--max-depth', type=float, default=1.0, help='metres')
    ap.add_argument('--max-rotation-rate', type=float, default=1.2, help='rad/s: faster frames are smeared')
    ap.add_argument('--max-frames', type=int, default=500)
    ap.add_argument('--refine', type=int, default=2, help='pose refinement passes (0 = trust ARKit)')
    ap.add_argument('--max-shift-mm', type=float, default=15)
    ap.add_argument('--max-turn-deg', type=float, default=3)
    ap.add_argument('--margin-mm', type=float, default=10, help='space added around the object box')
    ap.add_argument('--grow-mm', type=float, default=60,
                    help="let the phone's box grow by up to this much to fit what the scan saw standing in it (0 = off)")
    ap.add_argument('--cut-mm', type=float, default=1.0, help='ignore the table below this height; the base is extended back to it')
    ap.add_argument('--min-free', type=int, default=2, help='frames that must see through a voxel to call it empty')
    ap.add_argument('--smooth', type=int, default=16, help='Taubin smoothing iterations (0 = raw voxel surface)')
    ap.add_argument('--auto-box', action='store_true', help="ignore the phone's box; find the object automatically")
    ap.add_argument('--box-size-mm', help='override the box size: W,H,D')
    ap.add_argument('--center', help='auto box: world-space x,y,z (m) the object sits under')
    ap.add_argument('--radius-mm', type=float, default=150, help='auto box: search radius around the aim point')
    ap.add_argument('--scale', type=float, default=1.0, help='extra uniform scale')
    ap.add_argument('--calibrate', help='measured size along an output axis, e.g. x=52.3 (mm): rescales uniformly')
    ap.add_argument('--mat-scale', action='store_true', help="apply the capture mat's ARKit scale estimate")
    ap.add_argument('--mat-width-mm', type=float, help='measured width of your printed mat pattern (default: as declared, 180)')
    ap.add_argument('--keep-orientation', action='store_true', help="keep the box's yaw instead of squaring up the footprint")
    ap.add_argument('--no-preview', action='store_true')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    a.trunc_mm = a.trunc_mm or max(4.0, 3 * a.voxel_mm)
    scan = Scan(a.scan, a.room)
    out = a.o or os.path.join(a.scan, 'print')
    os.makedirs(out, exist_ok=True)
    name = a.name or re.sub(r'[^A-Za-z0-9._-]+', '-', scan.manifest.get('name') or 'object').strip('-') or 'object'
    info = {'format': 'object-print-report/1', 'scan': os.path.abspath(a.scan), 'inputFormat': scan.format}
    warnings = []

    # 1. frames
    stats = {'recorded': len(scan.records), 'motion': 0, 'noDepth': 0, 'subsampled': 0}
    frames = select_frames(scan, a, stats)
    stats['used'] = len(frames)
    info['frames'] = stats
    print(f'{len(frames)} of {len(scan.records)} depth frames usable ({stats["motion"]} too fast, {stats["noDepth"]} without depth'
          f'{", %d unreadable" % stats["unreadable"] if stats.get("unreadable") else ""})')
    if stats.get('unreadable'):
        warnings.append(f'{stats["unreadable"]} depth frame files were missing or damaged and were skipped')
    if len(frames) < 10:
        raise SystemExit('fewer than 10 usable depth frames')
    if scan.motion is None:
        warnings.append('no IMU samples: fast-motion frames were judged from pose deltas only')

    # 2. object frame
    obj_to_world, obj_lo, obj_hi = object_frame(scan, frames, a, rng, info)
    world_to_obj = np.linalg.inv(obj_to_world)
    for fr in frames:
        fr.T0 = world_to_obj @ fr.cam_to_world
        fr.T = fr.T0.copy()
    m = a.margin_mm / 1000
    lo = obj_lo - m
    lo[1] = -max(0.006, 4 * a.voxel_mm / 1000)
    hi = obj_hi + m
    voxel = a.voxel_mm / 1000
    while np.prod(np.ceil((hi - lo) / voxel) + 1) > 16e6:
        voxel *= 1.25
    if voxel > a.voxel_mm / 1000 * 1.01:
        warnings.append(f'box is large: voxel raised to {voxel * 1000:.2f} mm (shrink the box or pass --box-size-mm)')
    info['boxMM'] = [round(float(v) * 1000, 1) for v in obj_hi - obj_lo]
    info['objectToWorld'] = column_major(obj_to_world)
    print(f'object box {info["boxMM"]} mm ({info["box"]}), plane: {info["planeSource"]}')

    # 3-4. pose refinement on a coarser volume, then the final fusion
    if a.refine:
        refine_poses(frames, lo, hi, a, info)
        # the refined frames agree on a slightly different table: put the base on that one
        C, fit = refit_plane(frames, lo, hi, rng)
        if C is not None:
            Ci = np.linalg.inv(C)
            for fr in frames:
                fr.T, fr.T0 = Ci @ fr.T, Ci @ fr.T0
            obj_to_world = obj_to_world @ C
            info['objectToWorld'] = column_major(obj_to_world)
            info['planeAfterRefinement'] = fit
    vol = Volume(lo, hi, voxel, a.trunc_mm / 1000)
    t1 = time.time()
    for fr in frames:
        vol.integrate(fr, fr.T, min_depth=a.min_depth)
    print(f'fused {len(frames)} frames into {"x".join(map(str, vol.dims))} voxels of {voxel * 1000:.2f} mm ({time.time() - t1:.0f}s)')
    info['voxelMM'] = round(voxel * 1000, 3)
    info['truncationMM'] = round(vol.trunc * 1000, 2)

    # 5. solid -> mesh
    field, base_y = solid_field(vol, a, info)
    r = max(1, int(round(vol.trunc / voxel)) - 1)
    info['thinSolidFraction'] = round(thin_fraction(field < 0, r), 3)
    thin_mm = (2 * r + 1) * voxel * 1000
    V, F = marching_tetrahedra(field, vol.origin, voxel)
    if len(F) == 0:
        raise SystemExit('empty surface')
    base = V[:, 1] < base_y
    V[base, 1] = 0.0  # the cap the cut left becomes the flat base on the support plane
    V = taubin(V, F, a.smooth, base)

    # 6. print frame: millimetres, Z up, base on Z = 0, centred on the footprint
    P = np.c_[V[:, 0], -V[:, 2], V[:, 1]] * 1000
    if not a.keep_orientation:
        yaw = footprint_yaw(P)
        c, s_ = math.cos(yaw), math.sin(yaw)
        P[:, :2] = P[:, :2] @ np.array([[c, s_], [-s_, c]])
        info['footprintYawDeg'] = round(math.degrees(yaw), 2)
    scale, scale_source = a.scale, 'manual' if a.scale != 1 else 'none'
    mat_info = info.get('mat') or {}
    mat_error = None  # how much bigger ARKit's world is than reality, judged by the printed mat
    if mat_info.get('estimatedScaleFactor'):
        declared = (mat_info.get('physicalWidth') or 0.18) * 1000
        mat_error = mat_info['estimatedScaleFactor'] * declared / (a.mat_width_mm or declared)
    if a.mat_scale:
        if mat_error:
            scale /= mat_error
            scale_source = 'capture mat'
            if not a.mat_width_mm:
                warnings.append(f'--mat-scale assumed the mat printed at exactly {declared:.0f} mm: check the 100 mm bar '
                                'and pass --mat-width-mm if it is off')
        else:
            warnings.append('--mat-scale: this scan has no capture-mat scale estimate')
    if a.calibrate:
        axis, value = a.calibrate.split('=')
        k = 'xyz'.index(axis.strip().lower())
        raw = (P[:, k].max() - P[:, k].min()) * scale
        measured = robust_extent(P, F, k) * scale
        scale *= float(value) / measured
        scale_source = f'calibrated {axis}={value} mm (measured {measured:.2f}, bounding box {raw:.2f})'
    P *= scale
    P[:, :2] -= (P[:, :2].min(0) + P[:, :2].max(0)) / 2
    P[:, 2] -= P[:, 2].min()
    info['scale'] = {'applied': round(scale, 6), 'source': scale_source,
                     'matScaleError': round(mat_error, 5) if mat_error else None}
    if mat_error and not a.mat_scale and not a.calibrate and abs(mat_error - 1) > 0.02:
        if a.mat_width_mm:
            warnings.append(f'the capture mat suggests ARKit scale is off by {abs(mat_error - 1):.1%}: '
                            'measure the object and use --calibrate (or --mat-scale)')
        else:
            warnings.append(f'the capture mat measured {abs(mat_error - 1):.1%} {"smaller" if mat_error < 1 else "larger"} '
                            f'than its declared {declared:.0f} mm: usually the print was scaled (check the 100 mm bar and '
                            'pass --mat-width-mm); otherwise measure the object and use --calibrate')
    report = mesh_report(P, F)
    report['sizeMM'] = [round(float(v), 2) for v in P.max(0) - P.min(0)]
    info['mesh'] = report
    info['viewCoverage'] = round(view_coverage(frames, (obj_lo + obj_hi) / 2), 3)
    fc = feature_check(scan, obj_to_world, V)
    if fc:
        info['featurePointCheck'] = fc

    if info.get('box') != 'auto' and not info.get('planeRefit'):
        warnings.append("the table under the box could not be confirmed from the depth: the base may be tilted or "
                        'offset; check preview.png, or try --auto-box')
    if not report['watertight'] or not report['consistentOrientation']:
        warnings.append('mesh is not closed/consistently oriented: please report this scan')
    if info['viewCoverage'] < 0.5:
        warnings.append(f'only {info["viewCoverage"]:.0%} of view directions were recorded: orbit the object at 2-3 heights')
    if info['unobservedSurfaceFraction'] > 0.2:
        warnings.append(f'{info["unobservedSurfaceFraction"]:.0%} of the surface (not counting the base) was never measured '
                        'and was filled in as solid: rescan the missing sides, or treat those parts as guessed')
    if info.get('boxWallContact'):
        warnings.append(f'the object reaches the edge of the fusion volume ({", ".join(info["boxWallContact"])}) and was '
                        'cut flat there: rerun with a larger --margin-mm (e.g. 30) or --grow-mm, or --auto-box')
    if info.get('boxGrowthMM'):
        grown = ', '.join(f'{k} {v:g}' for k, v in info['boxGrowthMM'].items())
        warnings.append(f"the phone's box was too small and was grown to fit what stands on the table (mm: {grown}); "
                        'if that took in something next to the object, pass --grow-mm 0 or --box-size-mm W,H,D')
    if info['thinSolidFraction'] > 0.25:
        warnings.append(f'{info["thinSolidFraction"]:.0%} of the solid is thinner than ~{thin_mm:.0f} mm: fusion thickens such '
                        'walls by up to ~1.3 mm on their less-seen (inner) side, so inner sizes come out small and walls '
                        'under ~2 mm can get holes; measure a wall, or compare with --trunc-mm 2.5')
    if min(report['sizeMM']) < 15:
        warnings.append('features under ~3 mm are below what iPhone LiDAR resolves; expect rounded edges')
    pr = info.get('poseRefinement')
    if pr and pr['maxShiftMM'] > 8:
        warnings.append(f'ARKit poses needed up to {pr["maxShiftMM"]:.0f} mm of correction: tracking drifted; '
                        'a textured mat under the object helps')
    info['warnings'] = warnings

    stl = os.path.join(out, f'{name}.stl')
    write_stl(stl, P, F, f'3D Model Anything {name} mm Z-up')
    write_ply(os.path.join(out, f'{name}.ply'), P, F)
    if not a.no_preview and render_preview(os.path.join(out, 'preview.png'), P, F, report['sizeMM'], name):
        info['preview'] = 'preview.png'
    info['stl'] = os.path.basename(stl)
    info['units'] = 'mm, Z up, base on Z = 0'
    with open(os.path.join(out, 'poses.json'), 'w') as f:
        json.dump([{'file': fr.name, 'timestamp': fr.timestamp,
                    'arkit': column_major(obj_to_world @ fr.T0 @ ARKIT_TO_CV),
                    'refined': column_major(obj_to_world @ fr.T @ ARKIT_TO_CV)} for fr in frames], f)
    info['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(out, 'report.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f'{report["triangles"]:,} triangles, {report["sizeMM"][0]} x {report["sizeMM"][1]} x {report["sizeMM"][2]} mm, '
          f'{report["volumeMM3"] / 1000:.1f} cm3, watertight={report["watertight"]} -> {stl} ({info["seconds"]}s)')
    for w in warnings:
        print('warning:', w)
    return info


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
