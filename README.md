# devo_vo

A self-contained, portable subset of the [DEVO](https://github.com/tum-vision/DEVO) repo containing
everything `evals/eval_evs/eval_{tartan,mvsec,fpv,tumvie}_evs.py` need to run. Copy this whole folder
anywhere and the relative imports (`devo.*`, `utils.*`, `constants`) still resolve.

## License & Attribution

This is a derivative subset of [tum-vision/DEVO](https://github.com/tum-vision/DEVO), which is
MIT-licensed (Copyright (c) 2024 TUM Computer Vision Group & Princeton Vision & Learning Lab). DEVO
is a **visual odometry (VO)** method — frame-to-frame pose estimation from event data — and all
credit for that network architecture, training, and pretrained weights goes to the original DEVO
authors. This folder only repackages the pieces needed to run that VO front-end standalone, plus
the `dataset_api.py` convenience wrapper described below, so it can be dropped into any downstream
pipeline (SLAM, robotics, or otherwise) that needs event-based pose estimation. It was trimmed and
repackaged by Ekansh Bajpai as part of a master's thesis at Paderborn University (2026); see
[`LICENSE`](LICENSE) for the full terms, including the preserved original copyright notice.

If you use this code, please cite the original DEVO paper:

```bibtex
@inproceedings{klenk2023devo,
  title     = {Deep Event Visual Odometry},
  author    = {Klenk, Simon and Motzet, Marvin and Koestler, Lukas and Cremers, Daniel},
  booktitle = {International Conference on 3D Vision, 3DV 2024, Davos, Switzerland,
               March 18-21, 2024},
  pages     = {739--749},
  publisher = {{IEEE}},
  year      = {2024},
}
```

## Contents

```
devo_vo/
├── dataset_api.py                        # <-- unified Enum-based data/VO API, see below
├── evals/eval_evs/
│   ├── eval_tartan_evs.py
│   ├── eval_mvsec_evs.py
│   ├── eval_fpv_evs.py
│   └── eval_tumvie_evs.py
├── devo/                                 # DEVO VO package
│   ├── devo.py, enet.py, extractor.py, blocks.py, selector.py, ba.py,
│   │   projective_ops.py, utils.py, plot_utils.py, config.py
│   ├── altcorr/, fastba/, lietorch/      # CUDA/C++ source, unmodified from upstream DEVO --
│   │                                     # compiles to cuda_corr/cuda_ba/lietorch_backends
│   │                                     # (need building, see below)
├── utils/                                # data loading, eval, viz helpers
│   ├── load_utils.py     # <-- event voxel / event-frame loading for all 4 datasets
│   ├── eval_utils.py     # <-- runs VO + computes ATE
│   ├── transform_utils.py, viz_utils.py, event_utils.py, voxel_utils.py, pose_utils.py
├── config/
│   ├── default.yaml        # tartan
│   ├── eval_mvsec.yaml
│   ├── eval_fpv.yaml
│   └── eval_tumvie.yaml
├── splits/{tartan,mvsec,fpv,tumvie}/*.txt   # lists of validation scenes
├── constants.py
├── setup.py, requirements.txt, download_model.sh
└── DEVO.pth                              # pretrained weights (~40MB)
```

Left out on purpose (not needed by any of these 4 scripts): `devo/data_readers/` (training-only),
`evals/eval_rgb`, `evals/eval_e2v`, `evals/eval_evs_frame`, `evals/flow_depth`, other dataset splits
(eds, hku, rpg, vector), `thirdparty/*` (only needed for `--rpg_eval`, and those submodules aren't
checked out in the source repo either), `scripts/pp_*.py` preprocessing scripts (see note below),
`train.py`, `environment.yml` (upstream DEVO's own training conda env — pinned `cudatoolkit=11.3.1`
and other ancient versions; irrelevant here since there's no training code in this subset to run
with it).

## Setup

This is required for anything that touches `dataset_api.py` or the eval scripts — both import
`devo.devo.DEVO`, which unconditionally imports the compiled CUDA extensions
(`cuda_corr`/`cuda_ba`/`lietorch_backends`) at module level, so `import dataset_api` itself fails
without them. The only way to skip this build is bypassing `dataset_api.py` entirely and using
`utils/load_utils.py` / `utils/event_utils.py` directly for raw data loading, with no VO inference
(see "Where the eval-script code for each concept lives" below).

### Python dependencies

```bash
pip install -r requirements.txt
```

This installs `torch`, `torchvision`, `torch_scatter` (all pinned to a matching CUDA 12.8 build,
via the `--find-links` line at the top of the file), plus everything else `dataset_api.py` and the
eval scripts import at runtime (`numpy`, `scipy`, `natsort`, `yacs`, `evo`, `matplotlib`,
`opencv-python`, `h5py`, `hdf5plugin`, `numba`, `tabulate`). None of these are declared in
`setup.py` — its `install_requires` is empty; it only builds the CUDA extensions below.

### Building the CUDA extensions

`cuda_corr`, `cuda_ba`, and `lietorch_backends` are compiled from `devo/altcorr/`, `devo/fastba/`,
and `devo/lietorch/` respectively — unmodified CUDA/C++ source carried over from upstream
[tum-vision/DEVO](https://github.com/tum-vision/DEVO) (see License & Attribution above). They are
not something `devo_vo` or its author wrote; this section is just how to compile that existing
source against your own torch/CUDA build.

The `devo_vo` Python package itself needs no conda — the CUDA extensions build against whatever
`torch` is already installed in your active Python environment, via plain pip.

The three extensions need a CUDA **toolchain** (specifically `nvcc`) whose major version matches
your installed torch build — CUDA 12.x for `torch==2.8.0+cu128`. Ubuntu's own `apt` package
(`nvidia-cuda-toolkit`) is CUDA 11.5, too old to use. Two ways to get a matching `nvcc`, neither of
which requires conda:

**Option A — NVIDIA's official installer, no root needed.** Grab the Linux x86_64 runfile URL for
CUDA 12.8 from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
(select Linux → x86_64 → your distro → "runfile (local)"), then install it to a directory you own
— `--toolkit` skips the GPU driver (you already have one) and `--toolkitpath` avoids needing root:

```bash
sh cuda_12.8.*_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-12.8
export CUDA_HOME=$HOME/cuda-12.8
```

**Option B — a short-lived, build-only conda env**, if you already have (or don't mind installing)
conda — this is **only** a source of a compiler, not a runtime environment, and unrelated to any
project-level conda/venv:

```bash
conda create -n devo_build -c nvidia cuda-toolkit=12.8.2 -y
export CUDA_HOME=$(conda env list | awk '/devo_build/{print $NF}')
```

Either way, with `CUDA_HOME` set and your **project's own** venv/conda env active (not
`devo_build`/the runfile install — those are only a source of `nvcc`), fetch Eigen and build:

```bash
cd devo_vo
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch;print(".".join(map(str,torch.cuda.get_device_capability())))')"
pip install --no-build-isolation .
cd ..
```

Leave `CC`/`CXX` unset so the build uses your system `gcc`/`g++`, not one bundled alongside `nvcc`
(see Troubleshooting if this fails). `TORCH_CUDA_ARCH_LIST` must match your actual GPU
(`nvidia-smi --query-gpu=compute_cap --format=csv` also works — building for the wrong compute
capability is a common source of silent failures at runtime, not a build error). The build needs
`wheel` and benefits from `ninja` (`pip install wheel ninja`) if not already present.

## Running the eval scripts

All four take the same shape of command, from inside `devo_vo/` (same environment you built the
extensions into above):
```bash
python -m evals.eval_evs.eval_tartan_evs --datapath=<PATH> --weights="DEVO.pth" --stride=1 --trials=1 --expname=<name>
python -m evals.eval_evs.eval_mvsec_evs  --datapath=<PATH> --weights="DEVO.pth" --stride=1 --trials=1 --expname=<name>
python -m evals.eval_evs.eval_fpv_evs    --datapath=<PATH> --weights="DEVO.pth" --stride=1 --trials=1 --expname=<name>
python -m evals.eval_evs.eval_tumvie_evs --datapath=<PATH> --weights="DEVO.pth" --stride=1 --trials=1 --expname=<name>
```
Expected result on TartanAir `office2/Hard/P010`: ATE 27.32cm, matching the source DEVO repo.
`eval_mvsec_evs.py`, `eval_fpv_evs.py`, and `eval_tumvie_evs.py` share the same dependency chain
(`utils/load_utils.py`, `utils/eval_utils.py`, configs, splits) as `eval_tartan_evs.py`.

Expected per-scene directory layout:
```
tartan:  <datapath>/<scene>/evs_left/h5/*.h5, <datapath>/<scene>/pose_left.txt
mvsec:   <datapath>/<scene>/*_data.hdf5, calib_undist_<side>.txt, tss_imgs_us_<side>.txt,
         rectify_map_<side>.h5, <scene_base>_gt.hdf5   (GT poses)
fpv:     <datapath>/<scene>/events.txt, images_timestamps_us.txt, t_offset_us.txt,
         rectify_map.h5, calib_undist.txt, stamped_groundtruth_us_cam.txt (only "*_with_gt" scenes)
tumvie:  <datapath>/<scene>/*events_{left,right}.h5, {left,right}_images_undistorted/,
         calibration.json / rectify maps, mocap_data.txt (GT poses)
```
mvsec/fpv/tumvie need dataset-specific preprocessing first (undistortion, rectify maps, timestamp
alignment) — see `scripts/pp_mvsec.py` / `pp_fpv.py` / `pp_tumvie.py` in the source DEVO repo (not
copied here, since they're one-off preprocessing tools, not runtime dependencies of the eval scripts).
TartanAir-Events needs no preprocessing.

## `dataset_api.py` — unified loader for your own SLAM pipeline

A single file with no dependency on the `evals/*` scripts, meant to be dropped into (or imported
from) a different project. It wraps the same loading functions the eval scripts use, behind one
Enum-selected interface:

```python
import sys; sys.path.insert(0, "/path/to/devo_vo")
from dataset_api import Dataset, list_scenes, get_event_iterator, get_ground_truth, run_vo

scene = list_scenes(Dataset.TARTAN)[0]          # e.g. "office2/Hard/P010"

# 1) raw per-frame event voxel grids ("event_frames") + timestamps + intrinsics,
#    to feed into YOUR OWN VO front-end instead of DEVO's:
for voxel, intrinsics, t in get_event_iterator(Dataset.TARTAN, "/data/TartanAir", scene):
    ...  # voxel: [bins,H,W] cuda tensor, intrinsics: [fx,fy,cx,cy] cuda tensor, t: timestamp

# 2) ground-truth poses, if the dataset/scene has them:
tss_gt_us, poses_gt = get_ground_truth(Dataset.TARTAN, "/data/TartanAir", scene)  # ([N], [N,7] xyz+xyzw quat)

# 3) or let DEVO's pretrained VO produce the estimated trajectory, and feed
#    that into your own loop-closure / pose-graph back-end:
poses_est, tstamps_est = run_vo(Dataset.TARTAN, "/data/TartanAir", scene, weights="DEVO.pth")

# 4) same call, but also save a GT-vs-estimate comparison PDF with ATE/rotation metrics:
poses_est, tstamps_est = run_vo(Dataset.TARTAN, "/data/TartanAir", scene, weights="DEVO.pth",
                                 plot_path="office2_traj.pdf")
```

It can also be run directly as a smoke test (imports, iterates all frames, loads GT, runs VO), with
`--plot` to save the comparison PDF:
```bash
python dataset_api.py tartan --datapath=/data/TartanAir --scene=office2/Hard/P010 --plot=traj.pdf
```
On TartanAir `office2/Hard/P010`, this produces 557 event frames, 557 GT poses, and 557 estimated
VO poses, ATE ~26.5cm (matching `eval_tartan_evs.py`'s 27.3cm up to VO stochasticity/rounding).
`run_vo()` seeds `torch.manual_seed(1234)` before running, matching the eval scripts, so results
are reproducible run-to-run.

### Trajectory comparison plot (`--plot` / `plot_path=`)

`run_vo(..., plot_path="out.pdf")` fetches ground truth for the scene (skips the plot with a
warning if there isn't any, e.g. non-`_with_gt` FPV scenes) and calls `save_trajectory_plot()`,
which:
- time-syncs estimate against GT (skips fuzzy sync entirely when both already have the same
  length/timestamps, e.g. TartanAir; otherwise nearest-timestamp match within `max_diff_sec`,
  default 1.0s — same rule and tolerance as `utils/eval_utils.py::ate_real()`),
- aligns the estimate onto GT with Umeyama SE3/Sim3 (`align=`/`correct_scale=`, both default
  `True`),
- saves a one-page PDF: a 3D (x, y, z) plot of GT (dashed gray) vs aligned estimate (solid blue,
  equal aspect ratio on all 3 axes), with ATE (RMSE/mean/median/std, cm) and rotation RMSE (deg)
  printed under the plot,
- returns the metrics as a `dict`.

`save_trajectory_plot(tss_gt_us, poses_gt, tss_est_us, poses_est, out_pdf, ...)` is also exported
standalone — it takes plain `(timestamps, poses[N,7])` pairs, so you can use it to score *your own*
pipeline's final trajectory (after your loop closure / pose-graph optimization) against the same
`get_ground_truth()` output, not just DEVO's raw VO estimate.

Per-dataset notes baked into the registry (`_REGISTRY` in `dataset_api.py`):
- **image size**: tartan 480×640, mvsec/fpv 260×346, tumvie 720×1280 (all fixed, matching each
  dataset's default config yaml).
- **timestamps**: microseconds for mvsec/fpv/tumvie; for tartan they're just frame indices (the
  simulated dataset has no real clock — see `FREQ=50` in `eval_tartan_evs.py`).
- **poses**: `[N, 7]` = `xyz` + `xyzw` quaternion for both GT and estimate, in all four datasets.
- **kwargs forwarded** to the loaders: `stride=`, `timing=` (all), `scale=` (tartan only),
  `side=` (mvsec), `camID=` (tumvie, 2=left/3=right), `tss_gt_us=` (fpv, auto-crops voxels to the
  GT time range).
- FPV: only scenes ending in `_with_gt` (see `splits/fpv/fpv_val.txt`) have ground truth;
  `get_ground_truth` raises `FileNotFoundError` for the others.

If you need *raw* (x, y, t, polarity) events rather than pre-built voxel grids, see
`utils/event_utils.py` (`EventSlicer`, `to_voxel_grid`) — that's what the mvsec/fpv/tumvie loaders
call internally before voxelizing; `dataset_api.py` currently only exposes the voxel-grid stage,
since that's the common representation all four datasets and DEVO itself use.

## Where the eval-script code for each concept lives (for porting into your own pipeline)

### 1. Events / event-frames (voxel grids)
`utils/load_utils.py`
- `voxel_read()` / `voxel_iterator()` (tartan, ~L364/L439) — reads pre-baked `.h5` voxel grids.
- `mvsec_evs_iterator()` (~L826), `fpv_evs_iterator()` (~L1189), `tumvie_evs_iterator()` (~L84) —
  build voxel grids on the fly from raw event `.h5`/`.txt` files + rectify maps, via
  `utils/event_utils.py::to_voxel_grid` / `EventSlicer`.
- `dataset_api.get_event_iterator()` wraps all four behind one call.

### 2. Ground-truth poses
- tartan: `evals/eval_evs/eval_tartan_evs.py` L39-57 (`pose_left.txt`, NED→XYZ permutation, skips
  pose 0).
- mvsec: `utils/load_utils.py::load_mvsec_traj()` (~L560, reads `*_gt.hdf5`, converts 4×4 hom.
  matrices to quaternions via `utils/pose_utils.py::poses_hom_to_quatlist`).
- fpv: `utils/load_utils.py::load_gt_us()` (~L613, reads `stamped_groundtruth_us_cam.txt`).
- tumvie: `utils/load_utils.py::load_tumvie_traj()` (~L583, reads `mocap_data.txt`).
- `dataset_api.get_ground_truth()` wraps all four behind one call.

### 3. Estimated poses & their timestamps
`utils/eval_utils.py` → `run_voxel(...)` (~L109-139): builds a `devo.devo.DEVO` VO object, feeds
it each `(t, voxel, intrinsics)` from the relevant iterator, then:
```python
poses, tstamps = slam.terminate()
```
`poses` is `Nx7` (`xyz` + `xyzw` quaternion). `dataset_api.run_vo()` wraps this behind one call.

### 4. Tying it together / evaluation
`utils/eval_utils.py` → `log_results(...)` (~L314) takes `(traj_ref, tss_ref, traj_est, tss_est)`,
aligns them with `evo`, computes ATE, and (optionally) writes TUM-format trajectories via
`devo/plot_utils.py::save_trajectory_tum_format`. Not wrapped in `dataset_api.py` since it's specific
to ATE-style evaluation against ground truth, not something a downstream SLAM pipeline needs — for
loop closure / pose-graph optimization, use `run_vo()`'s output as your VO front-end input instead.

## Troubleshooting

**`ModuleNotFoundError` for `numpy`/`yacs`/`evo`/... right after building the CUDA extensions.**
`setup.py` only builds `cuda_corr`/`cuda_ba`/`lietorch_backends`; it declares no runtime
dependencies. Run `pip install -r requirements.txt` (see Setup) — this is a separate step from
building the extensions, not covered by `pip install .` alone.

**Build fails with a missing `pyconfig.h`.** Your `nvcc`'s bundled host compiler doesn't see
Ubuntu's multiarch Python headers (`/usr/include/x86_64-linux-gnu/python3.10`). Leave `CC`/`CXX`
unset so the build falls back to your system `gcc`/`g++` instead of a compiler bundled alongside
`nvcc` (e.g. inside a conda CUDA toolkit env).

**`ImportError: undefined symbol: ...` when importing `cuda_corr`/`cuda_ba`/`lietorch_backends`,
after they previously worked.** The extensions are compiled against a specific `torch` build's
ABI. If anything later reinstalls `torch` at a different version — e.g. `pip install torchvision`
with no version pin can silently pull in a newer `torch` from the default PyPI index instead of
your CUDA-matched build — the already-compiled `.so` files stop matching and imports fail with an
undefined-symbol error. Reinstall the exact pinned `torch`/`torchvision` versions from
`requirements.txt` (or rebuild the extensions) to fix it. This is why `requirements.txt` pins
`torchvision==0.23.0` via the same `--find-links` index as `torch_scatter`, rather than leaving it
unpinned.

**`ImportError: libc10.so: cannot open shared object file`** when importing `cuda_corr`/`cuda_ba`/
`lietorch_backends` directly (e.g. `python -c "import cuda_corr"`). Import `torch` first — this is
standard behavior for torch C++ extensions, not specific to this repo. `import dataset_api` and
`from devo.devo import DEVO` already do this correctly, since both import `torch` before anything
else.

**`NameError: name 'VONet' is not defined`** when constructing `DEVO(..., evs=False)`.
`devo/devo.py::DEVO.load_weights()` references `VONet` (the RGB/frame-based network), but its
import is commented out (`# from .net import VONet # TODO add net.py`) because `net.py` wasn't
carried over from upstream DEVO. Only the event-based path (`evs=True`, i.e. `eVONet`) is
supported in this subset — that's the only path this subset is built around and exercises.

**DEVO produces garbage poses, or gives no error but silently underperforms.** Check
`TORCH_CUDA_ARCH_LIST` was set to your actual GPU's compute capability during the build (see
Setup) — building for the wrong compute capability doesn't error, it just produces bad results.
