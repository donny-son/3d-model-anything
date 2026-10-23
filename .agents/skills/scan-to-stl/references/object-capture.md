# Object capture: format and how the STL pipeline uses it

The app's object mode writes `object-capture/1` scans to `Documents/Objects/<yyyy-MM-dd HH.mm.ss>/`
(Files → 3D Model Anything → Objects). Share one as a ZIP, extract it, and point `object_to_stl.py`
at the folder that contains `manifest.json`. The Codable models are in
`apps/house-capture/HouseCapture/ObjectCapture.swift`.

## Files

```text
manifest.json          format, name, device, box, supportPlane, mat, file index, viewCoverage, events
depth/frames.json      [{ file, confidence, rgb, rgbIntrinsics, camera, room: 0, timestamp, width, height, transform, intrinsics }]
depth/NNNNN.f16z       256×192 Float16 depth (m), raw DEFLATE: zlib.decompress(data, -15)
depth/NNNNN.conf       256×192 UInt8 ARKit confidence (0 low, 1 medium, 2 high)
depth/NNNNN.jpg        640×480 colour of the same frame
photos/photos.json     [{ file, room: 0, timestamp, width, height, transform, intrinsics, exposure, blurPixels, pass: "object", camera }]
photos/NNNN.jpg        1920×1440 video frames, sensor orientation (EXIF 6)
motion/imu.f64         little-endian Float64 rows: t qx qy qz qw gx gy gz rx ry rz ax ay az (100 Hz)
features/points.json   [{ timestamp, ids: [UInt64], points: [[x, y, z]] }] ARKit feature points inside the box
```

Depth and photo records are the same as `house-capture/5` (see the scan-to-threejs format guide), so
existing tools read them. Depth frames are kept up to 10 per second, only while tracking is normal,
the object is in view and 12–80 cm away, the phone is turning slower than 1.2 rad/s, and the camera
has moved 1 cm or turned 2.3° since the last frame.

## Coordinates

Everything is in the scan's ARKit world: metres, +Y up (gravity), right-handed. `transform`s are
column-major 4×4. Photo/depth transforms are camera-to-world; the camera looks down −Z with +Y up in
sensor orientation, and `intrinsics` (`[fx, 0, 0, 0, fy, 0, cx, cy, 1]`, column-major) match the saved
pixels. Project a world point `P`:

```text
pc = inverse(transform) · [P, 1];  pc.z < 0 is in front
u  = fx · pc.x / −pc.z + cx;  v = fy · −pc.y / −pc.z + cy
```

`box.transform` maps the object frame to the world: origin at the centre of the box's bottom face on
the support plane, +Y the plane normal, X/Z along the box sides. `box.size` is width (x), height (y),
depth (z) in metres. The box is deliberately generous; the pipeline adds `--margin-mm` around it.

`supportPlane.source` says where the base came from: `mat` (tracked capture mat), `plane-anchor`
(ARKit horizontal plane, with `classification` such as `table`) or `raycast` (no plane: a level plane
through the tap). ARKit planes can sit millimetres off; the pipeline refits the plane on the depth.

`mat` (when the printed capture mat was seen): `physicalWidth` the app assumed (0.18 m), the last
tracked pose, and `estimatedScaleFactor`, the median of ARKit's estimated / declared mat size
(`scaleSpread` is half the 10–90 % range). About 1.00 means ARKit's metric scale agreed with the print.

`motion`: `t` is seconds since boot, the same clock as frame and photo `timestamp`s. `q` is the
CoreMotion attitude (xArbitraryZVertical reference, not the ARKit world), `g` gravity and `a` user
acceleration in g, `r` rotation rate in rad/s, all in the device frame.

## What each signal does in `object_to_stl.py`

| Signal | Used for |
| --- | --- |
| LiDAR depth | the surface (TSDF band from high-confidence pixels) and empty space (every valid pixel carves the space in front of it) |
| Confidence | surfaces need confidence 2 by default (`--min-confidence`); grazing, distant and medium-confidence pixels get lower weights |
| Depth discontinuities | flying pixels at silhouettes are dropped from surfaces; carving uses the nearest depth around each pixel so it never eats into an edge |
| Camera pose (x, y, z + rotation) | where every depth pixel lands; refined per frame by Gauss-Newton alignment to the fused signed-distance field, corrections reported and capped (`--max-shift-mm`, `--max-turn-deg`) |
| Intrinsics | per-frame projection (focus changes are recorded per frame) |
| IMU rotation rate | rejects frames taken while turning faster than `--max-rotation-rate` (falls back to pose deltas when absent) |
| Box + support plane | the fusion volume and the object frame; the plane is refit on depth, then again after pose refinement, so the base prints flat |
| Capture mat | texture for tracking, a flat base, and an independent scale check (`--mat-scale`, `--mat-width-mm`) |
| Feature points | `featurePointCheck`: median distance from ARKit's own landmarks to the final surface |
| View directions | `viewCoverage`: share of 12 × 3 directions around the object with frames |

Space no frame ever saw through (the underside, sealed cavities, deep occlusions) becomes solid, so the
result is always printable; `unobservedSolidFraction` says how much of the solid that is. The surface
is extracted with marching tetrahedra (closed and manifold by construction), the table below
`--cut-mm` is removed and the base extended back to the plane, and Taubin smoothing removes voxel
steps without shrinking the object. The output is millimetres with Z up.

## Accuracy

On the synthetic benchmark (`test_object_to_stl.py`: 256×192 depth, 1–2 mm noise, mixed pixels at
silhouettes, ARKit-like drift) the mean surface error is about 0.2 mm, bounding boxes are within
about 1.5 mm, and pose refinement cuts the error left by drift by about a quarter
(mean 0.24 → 0.18 mm, 95th percentile 0.68 → 0.50 mm). Real scans are noisier.
iPhone depth is about 1.5 mm per pixel at 30 cm and is smoothed by ARKit, so expect rounded edges
and lost detail below about 3 mm. Dark, shiny, transparent and very thin parts are unreliable. For
critical fits, calibrate against a caliper measurement and print a test piece.
