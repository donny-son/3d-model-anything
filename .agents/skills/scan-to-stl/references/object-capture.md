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
depth (z) in metres. The app sizes it from the single depth frame at the tap, extended away from the
camera to be at least as deep as it is wide, plus 1 cm, so it can still miss a handle or the far
side. The pipeline therefore grows it (by up to `--grow-mm`, default 60, on every side including the
top) to whatever the whole scan saw standing on the plane and touching it, adds `--margin-mm`, and
warns if the measured surface still reaches the edge of the volume (`boxWallContact`), where it would
be cut flat. When the table could not be confirmed from the depth (`planeRefit` null), growth ignores
the lowest 12 mm. An explicit `--box-size-mm` is used as is, never grown.

`supportPlane.source` says where the base came from: `mat` (tracked capture mat), `plane-anchor`
(an ARKit horizontal plane whose extent reaches the tap, with `classification` such as `table`) or
`depth` (no plane: the lowest well-populated layer of LiDAR points under the tap). If none is found the
app asks you to point at the table instead of guessing. ARKit planes can sit millimetres off; the
pipeline refits the plane on the depth.

`mat` (when the printed capture mat was seen): `physicalWidth` the app assumed (0.18 m), the last
tracked pose, and `estimatedScaleFactor`, the median of ARKit's estimated / declared mat size
(`scaleSpread` is half the 10–90 % range). About 1.00 means ARKit's metric scale agreed with the print.
It measures the print as much as ARKit: a mat printed at 97 % ("fit to page") reads about 0.97, so
pass the measured pattern width as `--mat-width-mm` before trusting `--mat-scale`.

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
result is always printable. `unobservedSurfaceFraction` is the share of the object's outer surface
(not counting the base) that no depth frame measured, i.e. filled in rather than scanned; a solid
interior is expected and not counted. `thinSolidFraction` is the share of the solid thinner than about
twice the TSDF truncation (a roughly round probe ≈ 7 mm across at the defaults); above 25 % the
report warns (see Accuracy). The surface
is extracted with marching tetrahedra (closed and manifold by construction), the table below
`--cut-mm` is removed and the base extended back to the plane, and Taubin smoothing removes voxel
steps without shrinking the object. The output is millimetres with Z up.

## Accuracy

On the synthetic benchmark (`test_object_to_stl.py`: 256×192 depth, 1–2 mm noise, mixed pixels at
silhouettes, ARKit-like drift) the mean surface error is about 0.2 mm, bounding boxes are within
about 1.5 mm, and pose refinement cuts the error left by drift by about a quarter
(mean 0.24 → 0.18 mm, 95th percentile 0.68 → 0.50 mm). Real scans are noisier.
iPhone depth is about 1.5 mm per pixel at 30 cm and is smoothed by ARKit, so expect rounded edges
and lost detail below about 3 mm. Dark, shiny, transparent and very thin parts are unreliable.

Thin walls are biased, not just noisy. A wall thinner than about twice the truncation (cups, bowls,
vases, boxes) is seen far more from outside than inside, so fusion thickens it by about 0.5–1.5 mm on
the inner side: inner diameters come out small, and walls under about 2–3 mm can get holes (watch
`mesh.genus`: a solid without real handles should be 0 or a handful; tens or more means tunnels).
Calibrating on an outer dimension does not fix this; measure the wall or an inner size when it
matters. `--trunc-mm 2.5` thins such walls by a few tenths of a millimetre, but it adds tens of
tunnels in poorly seen surfaces (cup floors and insides) whatever the wall thickness and opens real
holes in walls of about 2–3 mm, so keep it only if `mesh.genus` stays close to the default run's.

`mesh.caliperMM` is the size `--calibrate axis=mm` works with: on each side, the area-weighted median
position of the outer layer of surface facing that way (a band holding under 2 mm² is taken for a
bump and skipped, and the base flare doesn't count). The fused mesh sits slightly inside the true surface and this outer-layer reading
offsets that, so it averages within a few hundredths of a millimetre of the truth with about ±0.3 mm
of scatter per reading. `sizeMM` is the plain bounding box, which reads about 0.5–1.5 mm larger. On the synthetic block
and can, `caliperMM` comes within about 0.35 mm of the true sizes (49.90 × 30.12 × 44.94 mm for
50 × 30 × 45; 59.87 × 59.65 × 39.97 for 60 × 60 × 40), and calibrating on the true length or diameter
leaves the mean surface error the same or lower. Compare a caliper value with the same axis of `caliperMM` (X is the footprint's long side, Y its
short side, Z the height), not with `sizeMM`. Because each reading scatters by about ±0.3 mm, calibrate
only when they differ by more than about 0.5 mm; the report warns when a calibration is within that
0.5 mm, and when it would rescale by more than 3 % while the value matches a different axis much
better (ARKit's scale is rarely that far off, so that is usually a mix-up of axes). Calibrate on the longest dimension, between
flat faces or across a round body away from the base; a short side magnifies any error. For critical
fits, print a test piece.
