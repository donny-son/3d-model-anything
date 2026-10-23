---
name: scan-to-stl
description: Turn 3D Model Anything iPhone object scans (LiDAR depth, camera poses, IMU) into accurate, watertight, millimetre-scaled STL files for 3D printing. Use for small objects (about 2–30 cm) captured with the app's object mode, or an object inside a room session.
---

# Scan to STL

The phone records evidence; this skill fuses it into a printable solid and tells you how far to trust it. LiDAR gives metric depth, ARKit gives every frame's position (x, y, z) and orientation, the IMU says which frames were smeared by motion, and the support plane and optional capture mat fix the base and scale. The result is a measured model, not a CAD-perfect one: report what was measured and what was filled in.

## Inputs

Read [object-capture.md](references/object-capture.md) for the `object-capture/1` format, coordinates and how each recorded signal is used. Check `manifest.json`'s `format` first.

- `object-capture/1` (app: **Scan something to 3D print**): has the object box, support plane, IMU and feature points. Use it directly.
- `house-capture/5` session or room export: depth frames and poses only. Pass `--room N`; the object is found where most camera axes converge, which works when the scan includes a slow circle around it. If it stops with "could not find a support plane", give `--center x,y,z` in session-world metres.

Keep scans in the ignored `data/captures-in/` (or anywhere outside Git). Never commit personal scans.

## Run

Set `S` to this skill's `scripts` directory. Needs Python 3 with NumPy (Pillow for previews).

```sh
python3 "$S/object_to_stl.py" data/captures-in/mug            # → data/captures-in/mug/print/
python3 "$S/object_to_stl.py" data/captures-in/mug --calibrate x=82.4   # caliper-measured X (the longest side) in mm
python3 "$S/test_object_to_stl.py"                              # synthetic end-to-end check
python3 "$S/make_capture_mat.py" -o capture-mat.pdf --paper letter     # printable mat
```

Outputs in `<scan>/print/` (or `-o`): `<name>.stl` (binary, mm, Z up, base on Z = 0, footprint centred and squared to X/Y), `<name>.ply`, `preview.png` (front/right/top/iso), `report.json`, `poses.json` (ARKit vs refined pose per frame).

## Check before you hand it over

1. Read `report.json`: `mesh.watertight`, `mesh.consistentOrientation`, `mesh.components` (1), `sizeMM`, `volumeMM3`, `mesh.genus` (0 or a handful for an object without handles; tens or more means tunnels or holes), `sizeMM` (bounding box) and `mesh.caliperMM` (what calipers would read), `frames` (used/rejected/unreadable), `poseRefinement` (residual before → after, corrections), `planeRefit`, `boxGrowthMM`, `boxWallContact`, `unobservedSurfaceFraction`, `thinSolidFraction`, `viewCoverage`, `warnings`. Relay every warning.
2. Look at `preview.png`. Compare it with a few of the scan's `photos/*.jpg` (rotate 90° clockwise to view upright). Look for missing thin parts, blobs fused from neighbouring clutter, and a skirt at the base.
3. Ask for one caliper measurement whenever accuracy matters, ideally the longest dimension (between flat faces, or across a round body away from the base), compare it with `mesh.caliperMM` (not the bounding box `sizeMM`), and if they differ by more than a few tenths of a millimetre rerun with `--calibrate axis=mm` (axes as in the STL: X is the footprint's long side). State the residual uncertainty: LiDAR fusion is typically within ~1 mm on matte objects; shiny, black, transparent or very thin features are worse, and thin walls come out thick (below).

## Fixing common problems

- Parts cut off flat (`boxWallContact`): the measured surface reached the edge of the volume. Raise `--margin-mm` (e.g. 30) or `--grow-mm` (or `--box-size-mm` if you set one), or use `--auto-box`. The phone's box is only a starting point: the pipeline grows it by up to `--grow-mm` to what the scan saw standing on the table and touching it (`boxGrowthMM`).
- Clutter fused onto the object: something next to it touched the box and was grown into it. Pass `--grow-mm 0`, or `--box-size-mm W,H,D` (width, height, depth in the box's frame, used as is and never grown; the app shows W, D and H labelled), or move things apart and rescan.
- Table remnants or a thick skirt: raise `--cut-mm` (default 1.0; the base is extended back down to the table). Lower it if a thin base flange is lost.
- Bumpy surface: raise `--smooth` (default 16). Faceted: lower `--voxel-mm` (default 1.0; LiDAR pixels are ~1.5 mm at 30 cm, so below ~0.7 mm gains little).
- Doubled or blurred surfaces with large `poseRefinement` corrections: tracking drifted. Rescan on the capture mat or a textured surface. Try `--refine 0` to compare against raw ARKit poses.
- Many frames rejected for motion: the scan was too fast; `--max-rotation-rate` relaxes the limit at the cost of smeared depth.
- Underside, deep holes or overhangs are filled solid when no frame saw into them. That is deliberate (printable), not measured; `unobservedSurfaceFraction` says how much of the outer surface was filled in. Say so.
- Thin walls (`thinSolidFraction` warning; cups, bowls, boxes): fusion thickens them by about 0.5–1.5 mm on the less-seen inner side, so inner sizes are small and walls under ~2–3 mm can get holes (check `mesh.genus`). Calibrating on an outer size doesn't fix that. Measure a wall; `--trunc-mm 2.5` thins walls but opens holes in walls of ~3 mm or less, so keep it only if `mesh.genus` stays near the default run's. For a precise fit, model the walls in CAD from the measured outer shape.
- Scale: `--calibrate` with a caliper value on the longest dimension, after comparing it with `mesh.caliperMM`. `--mat-scale` applies ARKit's estimate of the capture mat's size; it is only as good as the print, so measure the mat's 100 mm bar and pass `--mat-width-mm` (the pattern is 180 mm wide at 100 %) if it is off.

For fine detail beyond LiDAR resolution (text, threads, sharp edges), model the part in CAD using the STL and photos as reference rather than over-processing the scan.
