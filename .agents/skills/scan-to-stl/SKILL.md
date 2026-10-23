---
name: scan-to-stl
description: Turn 3D Model Anything iPhone object scans (LiDAR depth, camera poses, IMU) into accurate, watertight, millimetre-scaled STL files for 3D printing. Use for small objects (about 2–30 cm) captured with the app's object mode, or an object inside a room session.
---

# Scan to STL

The phone records evidence; this skill fuses it into a printable solid and tells you how far to trust it. LiDAR gives metric depth, ARKit gives every frame's position (x, y, z) and orientation, the IMU says which frames were smeared by motion, and the support plane and optional capture mat fix the base and scale. The result is a measured model, not a CAD-perfect one: report what was measured and what was filled in.

## Inputs

Read [object-capture.md](references/object-capture.md) for the `object-capture/1` format, coordinates and how each recorded signal is used. Check `manifest.json`'s `format` first.

- `object-capture/1` (app: **Scan something to 3D print**): has the object box, support plane, IMU and feature points. Use it directly.
- `house-capture/5` session or room export: depth frames and poses only. Pass `--room N`; the object is found where the cameras converge (`--auto-box`), or give `--center x,y,z` in session-world metres.

Keep scans in the ignored `data/captures-in/` (or anywhere outside Git). Never commit personal scans.

## Run

Set `S` to this skill's `scripts` directory. Needs Python 3 with NumPy (Pillow for previews).

```sh
python3 "$S/object_to_stl.py" data/captures-in/mug            # → data/captures-in/mug/print/
python3 "$S/object_to_stl.py" data/captures-in/mug --calibrate x=82.4   # caliper-measured X in mm
python3 "$S/test_object_to_stl.py"                              # synthetic end-to-end check
python3 "$S/make_capture_mat.py" -o capture-mat.pdf --paper letter     # printable mat
```

Outputs in `<scan>/print/` (or `-o`): `<name>.stl` (binary, mm, Z up, base on Z = 0, footprint centred and squared to X/Y), `<name>.ply`, `preview.png` (front/right/top/iso), `report.json`, `poses.json` (ARKit vs refined pose per frame).

## Check before you hand it over

1. Read `report.json`: `mesh.watertight`, `mesh.consistentOrientation`, `mesh.components` (1), `sizeMM`, `volumeMM3`, `frames` (used/rejected), `poseRefinement` (residual before → after, corrections), `planeRefit`, `unobservedSolidFraction`, `viewCoverage`, `warnings`.
2. Look at `preview.png`. Compare it with a few of the scan's `photos/*.jpg` (rotate 90° clockwise to view upright). Look for missing thin parts, blobs fused from neighbouring clutter, and a skirt at the base.
3. Ask for one caliper measurement whenever accuracy matters and rerun with `--calibrate axis=mm`. State the residual uncertainty: LiDAR fusion is typically within ~1 mm on matte objects; shiny, black, transparent or very thin features are worse.

## Fixing common problems

- Clutter fused onto the object or parts cut off: the phone's box is too small or too big. Use `--box-size-mm W,H,D` (keeps the box's frame) or `--auto-box`.
- Table remnants or a thick skirt: raise `--cut-mm` (default 1.0; the base is extended back down to the table). Lower it if a thin base flange is lost.
- Bumpy surface: raise `--smooth` (default 16). Faceted: lower `--voxel-mm` (default 1.0; LiDAR pixels are ~1.5 mm at 30 cm, so below ~0.7 mm gains little).
- Doubled or blurred surfaces with large `poseRefinement` corrections: tracking drifted. Rescan on the capture mat or a textured surface. Try `--refine 0` to compare against raw ARKit poses.
- Many frames rejected for motion: the scan was too fast; `--max-rotation-rate` relaxes the limit at the cost of smeared depth.
- Underside, deep holes or overhangs are filled solid when no frame saw into them (`unobservedSolidFraction`). That is deliberate (printable), not measured; say so.
- Scale: `--calibrate` from a caliper measurement is the most reliable. `--mat-scale` applies ARKit's capture-mat size estimate (pass `--mat-width-mm` if the print was not exactly 180 mm).

For fine detail beyond LiDAR resolution (text, threads, sharp edges), model the part in CAD using the STL and photos as reference rather than over-processing the scan.
