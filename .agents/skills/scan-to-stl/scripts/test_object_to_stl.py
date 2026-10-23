#!/usr/bin/env python3
"""End-to-end check of object_to_stl.py on a synthetic object-capture/1 scan with known geometry.

A 50 x 30 x 25 mm block with a 24 mm cylinder rising to 45 mm sits on a table. An orbit of
256x192 LiDAR-like depth frames is rendered with depth noise, mixed pixels at silhouettes,
low-confidence dropouts, ARKit-like pose drift, a few fast-motion frames (flagged by the IMU,
with smeared depth), and a support plane recorded 3 mm high and 1 degree tilted. The STL must
be closed and within tolerance of the true shape, and pose refinement must reduce the error.

usage: test_object_to_stl.py [--keep DIR]   (needs numpy; Pillow optional)
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import object_to_stl as o2s  # noqa: E402

BOX = (np.array([-0.030, 0.0, -0.015]), np.array([0.020, 0.025, 0.015]))
CYL = (np.array([0.008, 0.0]), 0.012, 0.045)  # centre (x, z), radius, height
TRUE_SIZE_MM = np.array([50.0, 30.0, 45.0])   # print frame: x, y (= -z), z (= up)
TRUE_VOLUME_MM3 = 50 * 25 * 30 + math.pi * 12 ** 2 * 20
W, H, FX, CX, CY = 256, 192, 212.0, 127.5, 95.5


def true_sdf(p):
    """Signed distance (m) to the true object, object frame (Y up)."""
    c, h = (BOX[0] + BOX[1]) / 2, (BOX[1] - BOX[0]) / 2
    q = np.abs(p - c) - h
    box = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    (cx, cz), r, top = CYL
    d = np.c_[np.hypot(p[:, 0] - cx, p[:, 2] - cz) - r, np.abs(p[:, 1] - top / 2) - top / 2]
    cyl = np.minimum(d.max(1), 0) + np.linalg.norm(np.maximum(d, 0), axis=1)
    return np.minimum(box, cyl)


def cast(o, d):
    """Depth along each ray (rays scaled so camera z = 1, so t is depth) to plane/box/cylinder."""
    t = np.full(len(d), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        tp = -o[1] / d[:, 1]
        t = np.where((d[:, 1] < 0) & (tp > 0), tp, t)
        t0 = (BOX[0] - o) / d
        t1 = (BOX[1] - o) / d
        near = np.nanmax(np.minimum(t0, t1), axis=1)
        far = np.nanmin(np.maximum(t0, t1), axis=1)
        t = np.where((near <= far) & (near > 0), np.minimum(t, near), t)
        (cx, cz), r, top = CYL
        ox, oz = o[0] - cx, o[2] - cz
        a = d[:, 0] ** 2 + d[:, 2] ** 2
        b = 2 * (ox * d[:, 0] + oz * d[:, 2])
        c = ox ** 2 + oz ** 2 - r ** 2
        disc = b * b - 4 * a * c
        ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
        y = o[1] + ts * d[:, 1]
        t = np.where((disc > 0) & (ts > 0) & (y >= 0) & (y <= top), np.minimum(t, ts), t)
        tc = (top - o[1]) / d[:, 1]
        hx, hz = ox + tc * d[:, 0], oz + tc * d[:, 2]
        t = np.where((tc > 0) & (hx ** 2 + hz ** 2 <= r * r), np.minimum(t, tc), t)
    return t


def look_at(pos, target, roll):
    """ARKit camera-to-object pose: camera looks down -Z."""
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 1.0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cr, sr = math.cos(roll), math.sin(roll)
    x, y = cr * x + sr * y, -sr * x + cr * y
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, pos
    return T


def make_scan(folder, rng, drift=True, with_box=True):
    os.makedirs(os.path.join(folder, 'depth'))
    os.makedirs(os.path.join(folder, 'motion'))
    os.makedirs(os.path.join(folder, 'features'))
    yaw = math.radians(25)
    obj_to_world = np.eye(4)
    obj_to_world[:3, :3] = [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]]
    obj_to_world[:3, 3] = [0.42, 0.74, -0.31]
    target = np.array([0.0, 0.02, 0.0])
    poses = []
    for ring, (elev, radius) in enumerate([(15, 0.31), (35, 0.28), (55, 0.27), (75, 0.29)]):
        for k in range(30):
            az = math.radians(k * 12 + ring * 5 + rng.normal(0, 2))
            e = math.radians(elev + rng.normal(0, 2))
            pos = target + radius * np.array([math.cos(e) * math.cos(az), math.sin(e), math.cos(e) * math.sin(az)])
            poses.append(look_at(pos, target, rng.uniform(-math.pi, math.pi)))
    fast = set(rng.choice(len(poses), 6, replace=False).tolist())
    ys, xs = np.mgrid[0:H, 0:W]
    rays_cv = np.stack([(xs - CX) / FX, (ys - CY) / FX, np.ones_like(xs, float)], -1).reshape(-1, 3)
    records, imu = [], []
    offset, axis_drift = np.zeros(3), np.zeros(3)
    for i, cam_to_obj in enumerate(poses):
        cv = cam_to_obj @ o2s.ARKIT_TO_CV
        depth = cast(cv[:3, 3], rays_cv @ cv[:3, :3].T).reshape(H, W)
        seen = np.isfinite(depth) & (depth < 5)  # LiDAR range
        conf = np.where(seen, 2, 0).astype(np.uint8)
        depth = np.where(seen, depth, 0)
        # mixed pixels along silhouettes, then noise and dropouts
        jump = np.abs(np.diff(depth, axis=1))  # (sky pixels have depth 0)
        sil = np.zeros((H, W), bool)
        sil[:, 1:] |= jump > 0.01
        sil[:, :-1] |= jump > 0.01
        mix = sil & (rng.random((H, W)) < 0.5)
        blurred = (depth + np.roll(depth, 1, 1) + np.roll(depth, -1, 1)) / 3
        depth = np.where(mix, blurred, depth)
        conf[sil] = 1
        depth = depth + rng.normal(0, 1, depth.shape) * (0.001 + 0.003 * depth)
        conf[rng.random((H, W)) < 0.01] = 0
        if i in fast:
            depth = depth + 0.02 * np.sin(xs / 9.0)  # motion-smeared depth the IMU should reject
        depth = np.where(conf > 0, depth, 0).astype('<f2')
        name = f'depth/{i:05d}'
        comp = zlib.compressobj(6, zlib.DEFLATED, -15)
        with open(os.path.join(folder, name + '.f16z'), 'wb') as f:
            f.write(comp.compress(depth.tobytes()) + comp.flush())
        conf.tofile(os.path.join(folder, name + '.conf'))
        if drift:  # ARKit-like slow drift of the recorded pose around the object
            offset += rng.normal(0, 0.00015, 3)
            axis_drift += rng.normal(0, math.radians(0.015), 3)
        D = np.eye(4)
        D[:3, :3] = o2s.rodrigues(axis_drift)
        D[:3, 3] = target - D[:3, :3] @ target + offset
        recorded = obj_to_world @ D @ cam_to_obj
        k = np.array([[FX, 0, CX], [0, FX, CY], [0, 0, 1]])
        t = 10 + i * 0.15
        records.append({'file': name + '.f16z', 'confidence': name + '.conf', 'room': 0, 'timestamp': t,
                        'width': W, 'height': H, 'transform': o2s.column_major(recorded),
                        'intrinsics': [float(v) for v in k.T.ravel()]})
        for s in range(15):
            ts = t - 0.075 + s * 0.01
            rate = 3.0 if (i in fast and abs(ts - t) < 0.04) else 0.15
            imu.append([ts, 0, 0, 0, 1, 0, -1, 0, rate, 0, 0, 0, 0, 0])
    with open(os.path.join(folder, 'depth/frames.json'), 'w') as f:
        json.dump(records, f)
    np.array(imu, '<f8').tofile(os.path.join(folder, 'motion/imu.f64'))
    # ARKit feature points: noisy samples of the true surface
    pts = rng.uniform(BOX[0] - 0.005, BOX[1] + np.array([0.005, 0.025, 0.005]), (4000, 3))
    pts = pts[np.abs(true_sdf(pts)) < 0.0005][:80]
    features = [{'timestamp': 10 + j, 'ids': list(range(len(pts))),
                 'points': o2s.transform(obj_to_world, pts + rng.normal(0, 0.001, pts.shape)).tolist()}
                for j in range(4)]
    with open(os.path.join(folder, 'features/points.json'), 'w') as f:
        json.dump(features, f)
    # the phone's box: generous, its plane 3 mm high and tilted 1 degree
    tilt = o2s.rodrigues(np.array([math.radians(1.0), 0, 0]))
    B = np.eye(4)
    B[:3, :3] = tilt
    B[:3, 3] = [-0.005, 0.003, 0.0]
    manifest = {
        'format': 'object-capture/1', 'name': 'synthetic block', 'units': 'meters',
        'depthFrames': 'depth/frames.json', 'depthFrameCount': len(records),
        'motion': {'file': 'motion/imu.f64', 'rateHz': 100,
                   'fields': ['t', 'qx', 'qy', 'qz', 'qw', 'gx', 'gy', 'gz', 'rx', 'ry', 'rz', 'ax', 'ay', 'az']},
        'features': {'file': 'features/points.json'},
        'mat': {'physicalWidth': 0.18, 'estimatedScaleFactor': 1.004, 'samples': 40, 'scaleSpread': 0.002},
    }
    if with_box:
        manifest['box'] = {'transform': o2s.column_major(obj_to_world @ B), 'size': [0.060, 0.052, 0.040]}
        manifest['supportPlane'] = {'source': 'plane-anchor'}
    with open(os.path.join(folder, 'manifest.json'), 'w') as f:
        json.dump(manifest, f)
    return len(fast)


def surface_error_mm(stl_path):
    """Mean / 95th-percentile distance (mm) of STL vertices to the true surface, after the
    rigid fit that best places the STL on the truth (shape error, not where the scan put it)."""
    tri = o2s.read_stl(stl_path).reshape(-1, 3).astype(np.float64)
    v = np.unique(tri, axis=0)
    centre = np.array([(BOX[0][0] + BOX[1][0]) / 2, (BOX[0][2] + BOX[1][2]) / 2])
    p = np.c_[v[:, 0] / 1000 + centre[0], v[:, 2] / 1000, -v[:, 1] / 1000 + centre[1]]
    c = p.mean(0)
    eps = 1e-5
    for _ in range(15):  # Gauss-Newton on the analytic SDF
        r = true_sdf(p)
        g = np.stack([(true_sdf(p + eps * e) - r) / eps for e in np.eye(3)], 1)
        J = np.c_[np.cross(p - c, g), g]
        delta = -np.linalg.lstsq(J, r, rcond=None)[0]
        R = o2s.rodrigues(delta[:3])
        p = (p - c) @ R.T + c + delta[3:]
        if np.linalg.norm(delta) < 1e-8:
            break
    d = np.abs(true_sdf(p)) * 1000
    return float(d.mean()), float(np.percentile(d, 95))


def run(folder, out, *extra):
    return o2s.main([folder, '-o', out, *extra])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', help='write scans and results here instead of a temp folder')
    a = ap.parse_args()
    root = a.keep or tempfile.mkdtemp(prefix='object-stl-test-')
    os.makedirs(root, exist_ok=True)
    failures = []

    def check(ok, what):
        print(('PASS ' if ok else 'FAIL ') + what)
        if not ok:
            failures.append(what)

    try:
        scan = os.path.join(root, 'scan')
        if os.path.exists(scan):
            shutil.rmtree(scan)
        n_fast = make_scan(scan, np.random.default_rng(1))

        print('\n== refined ==')
        info = run(scan, os.path.join(root, 'refined'))
        m = info['mesh']
        size_err = np.abs(np.array(m['sizeMM']) - TRUE_SIZE_MM)
        mean_err, p95_err = surface_error_mm(os.path.join(root, 'refined', info['stl']))
        print(f'size {m["sizeMM"]} (error {np.round(size_err, 2)} mm), volume {m["volumeMM3"]:.0f} mm3 '
              f'(true {TRUE_VOLUME_MM3:.0f}), surface error mean {mean_err:.2f} / p95 {p95_err:.2f} mm')
        check(m['watertight'] and m['consistentOrientation'], 'STL is closed and consistently oriented')
        check(m['components'] == 1 and m['genus'] == 0, 'one solid without handles')
        check(m['degenerateTriangles'] == 0, 'no degenerate triangles')
        # the bounding box is a max statistic: LiDAR noise bumps inflate it more than the surface error
        check(bool(np.all(size_err < 1.6)), 'bounding box within 1.6 mm on every axis')
        check(abs(m['volumeMM3'] / TRUE_VOLUME_MM3 - 1) < 0.04, 'volume within 4%')
        check(mean_err < 0.5 and p95_err < 1.3, 'surface within 0.5 mm mean, 1.3 mm p95 of the truth')
        check(info['frames']['motion'] == n_fast, f'the {n_fast} fast-motion frames were rejected from IMU data')
        check(os.path.exists(os.path.join(root, 'refined', 'report.json')), 'report written')

        print('\n== no drift ==')
        still = os.path.join(root, 'scan-no-drift')
        if os.path.exists(still):
            shutil.rmtree(still)
        make_scan(still, np.random.default_rng(2), drift=False)
        exact = run(still, os.path.join(root, 'no-drift'), '--no-preview')
        fit = exact.get('planeRefit') or {}
        check(abs(fit.get('shiftMM', 0) + 3) < 0.3 and abs(fit.get('tiltDeg', 0) - 1) < 0.1,
              'recorded plane error (3 mm high, 1 deg) corrected from the depth')
        check(bool(np.all(np.abs(np.array(exact['mesh']['sizeMM']) - TRUE_SIZE_MM) < 1.2)), 'bounding box within 1.2 mm without drift')
        exact_mean, _ = surface_error_mm(os.path.join(root, 'no-drift', exact['stl']))
        check(exact_mean < 0.35, f'surface within 0.35 mm mean without drift ({exact_mean:.2f})')
        check(exact['featurePointCheck']['medianDistanceMM'] < 1.0, 'ARKit feature points lie on the surface')

        print('\n== ARKit poses only ==')
        raw = run(scan, os.path.join(root, 'raw'), '--refine', '0', '--no-preview')
        raw_mean, raw_p95 = surface_error_mm(os.path.join(root, 'raw', raw['stl']))
        print(f'surface error without refinement: mean {raw_mean:.2f} / p95 {raw_p95:.2f} mm')
        check(mean_err < raw_mean * 0.85, 'pose refinement reduces the surface error by 15% or more')

        print('\n== calibrated + automatic box ==')
        auto = run(scan, os.path.join(root, 'auto'), '--auto-box', '--calibrate', 'x=50', '--no-preview')
        am = auto['mesh']
        check(am['watertight'] and am['components'] == 1, 'automatic box finds one closed object')
        check(abs(am['sizeMM'][0] - 50) < 0.05, '--calibrate x=50 sets the X size exactly')
        check(bool(np.all(np.abs(np.array(am['sizeMM']) - TRUE_SIZE_MM) < 1.5)), 'automatic box size within 1.5 mm')
    finally:
        if not a.keep:
            shutil.rmtree(root, ignore_errors=True)
    if failures:
        print(f'\n{len(failures)} check(s) failed')
        return 1
    print('\nall object STL checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
