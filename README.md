# 3D Model Anything

[![Watch the video: 3D Model Anything](docs/images/thumb-1.png)](https://www.youtube.com/watch?v=JGQCcCzuS88)

Capture a room on an iPhone, then use its geometry and photographs to build an editable Three.js scene with an AI coding agent.

[Open the office viewer](https://wesbos.github.io/3d-model-anything/) · [App support](https://wesbos.github.io/3d-model-anything/support.html)

This repository includes the **3D Model Anything iPhone app**, a **Three.js viewer**, the **scan-to-threejs agent skill**, and **Wes’s detailed office demo**.

It also scans **small objects for 3D printing**: the app records LiDAR depth, the exact position of every frame, motion data and ARKit feature points around one object, and the **scan-to-stl** skill fuses them into a watertight, millimetre-scaled STL. See [Scan small objects to STL](#scan-small-objects-to-stl).

The app records the evidence. The skill guides an agent through measurement, modeling and matched-camera visual comparisons. This is an iterative reconstruction workflow, not a one-click scan-to-perfect-model converter. The current workflow focuses on rooms and interiors.

## Run the viewer

Use Node.js 22.12+ (or a newer supported release).

```sh
npm ci
npm run dev
```

Open http://localhost:5188. Explore the office in dollhouse, walking, overhead, wireframe and knolling views. The playground is at `/play.html?room=office-v4&mode=fpv`.

```sh
npm run build
npm run preview
```

The demo includes the reconstructed office and selected textures. Original scans and photo collections are not distributed; Scan and Overlay are disabled for this example. Those modes remain available for your own rooms with reference assets.

## Capture your own space

See [the iPhone app](apps/house-capture/README.md) for building with Xcode, device requirements, scanning and sharing. Scanning needs a LiDAR-equipped iPhone or iPad. Non-LiDAR devices can import and review scans.

The app is source-available here; an App Store release is a separate distribution step. Do not assume an App Store download is available yet.

Export a room or session through the system share sheet and extract it into an ignored `data/captures-in/<room>/` directory. Exports keep the existing `house-capture/5` contract; changing the app name does not change the files.

## Use the skill

The skill lives at `.agents/skills/scan-to-threejs/SKILL.md`. Open this repository in an agent that supports local skills and ask it to use **scan-to-threejs** with your extracted export. You can also copy that skill directory into your agent's skill directory; the companion viewer/runtime in this repo is still needed for its capture workflow.

Example request:

> Use scan-to-threejs to rebuild the room in data/captures-in/my-room. Inspect the measurements and photos, create an editable scene, and compare rendered views with the original photographs.

Start with [the format guide](.agents/skills/scan-to-threejs/references/house-capture.md) and [project layout](.agents/skills/scan-to-threejs/references/project-layout.md). Helpers need Python 3; photo comparisons need Pillow, and optional depth reconstruction needs NumPy, Open3D and Pillow. Capture scripts use the `agent-browser` CLI.

A new room goes in `demos/<id>/` and is registered in `demos/index.ts`. The office example retains its earlier reconstruction layers (`office-v2`, `office-latest`, `office-v4`) to preserve the final geometry.

## Scan small objects to STL

On a LiDAR iPhone or iPad: **Scan something to 3D print** → **Scan an object**. Tap the object to box it, walk slowly around it at 25–40 cm until the dots around the box turn green, then share the scan as a ZIP. Print the capture mat first for steadier tracking and a flat base:

```sh
python3 .agents/skills/scan-to-stl/scripts/make_capture_mat.py -o capture-mat.pdf --paper letter
```

Print it at 100 % ("Actual size") and check that the bar on the page measures 100 mm.

Extract the scan into `data/captures-in/<object>/` and convert it (Python 3 with NumPy; Pillow for the preview):

```sh
python3 .agents/skills/scan-to-stl/scripts/object_to_stl.py data/captures-in/mug
python3 .agents/skills/scan-to-stl/scripts/object_to_stl.py data/captures-in/mug --calibrate x=82.4   # caliper-measured length of the longest side (X)
```

`data/captures-in/mug/print/` then holds `mug.stl` (millimetres, Z up, flat base on Z = 0), a preview sheet and `report.json` with the size (bounding box and a caliper-like `caliperMM`, which is what `--calibrate` matches), volume, watertightness, pose corrections and warnings. The pipeline filters LiDAR depth by confidence and motion, refines every frame's ARKit pose against the fused model, re-levels the base on the table, and fills unseen space so the solid always prints. It also works on room sessions that include a slow circle around the object (`--room N`); if it can't find the object there, pass `--center x,y,z`. Read the [skill](.agents/skills/scan-to-stl/SKILL.md) for checks and fixes; `test_object_to_stl.py` runs a synthetic end-to-end accuracy test. Expect around 1 mm accuracy on matte objects. Shiny, black, transparent and very thin parts are harder, and thin walls (cups, boxes) come out up to ~1.5 mm too thick; measure one long dimension and calibrate when fit matters.

## Privacy and licensing

Only the office example is distributed. No other rooms, connected-house layout, original scan bundles, or development Git history are included. Keep your captures private; inspect assets before publishing your own demo.

Reusable source code and the skill are MIT licensed. The supplied office images/textures have separate [asset terms](ASSETS-LICENSE.md). Third-party marks, artwork and depicted products are not licensed by the code license.

See [release status](docs/release-status.md) for validation and iPhone distribution steps.
