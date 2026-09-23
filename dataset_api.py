"""
Unified data-loading + VO front-end API for devo_vo.

Import this from a *different* project (add devo_vo to sys.path, or install
it as a package) to pull per-frame event voxel grids ("event_frames"),
timestamps, intrinsics, and ground-truth poses for one of the four supported
datasets, and/or to run DEVO's visual-odometry front-end to get estimated
poses. Feed the result into your own loop-closure / pose-graph back-end.

    import sys; sys.path.insert(0, "/path/to/devo_vo")
    from dataset_api import Dataset, get_event_iterator, get_ground_truth, run_vo

    for voxel, intrinsics, t in get_event_iterator(Dataset.TARTAN, "/data/TartanAir", "office2/Hard/P010"):
        my_frontend.process(voxel, intrinsics, t)   # your own VO/SLAM

    # or let DEVO do the VO for you, then post-process (loop closure, PGO, ...):
    poses, tstamps = run_vo(Dataset.TARTAN, "/data/TartanAir", "office2/Hard/P010", weights="DEVO.pth",
                             plot_path="office2_traj.pdf")  # optional: save a GT-vs-estimate comparison PDF

This module only wires together functions already in `utils/load_utils.py`,
`utils/eval_utils.py` and `devo/`; it adds no new data-loading logic of its
own, just a uniform per-dataset entry point selected by an Enum.
"""

import os.path as osp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
import torch

from devo.config import cfg as _default_cfg
from utils.eval_utils import run_voxel
from utils.load_utils import (
    voxel_iterator,
    mvsec_evs_iterator,
    load_mvsec_traj,
    fpv_evs_iterator,
    load_gt_us,
    tumvie_evs_iterator,
    load_tumvie_traj,
)

ROOT = Path(__file__).resolve().parent

EventIterator = Iterator[Tuple[torch.Tensor, torch.Tensor, float]]


class Dataset(Enum):
    TARTAN = "tartan"
    MVSEC = "mvsec"
    FPV = "fpv"
    TUMVIE = "tumvie"


def _tartan_iterator(scene_path: str, stride: int = 1, timing: bool = False, scale: float = 1.0, **_) -> EventIterator:
    return voxel_iterator(osp.join(scene_path, "evs_left", "h5"), stride=stride, timing=timing, scale=scale)


def _tartan_gt(scene_path: str, stride: int = 1, **_) -> Tuple[np.ndarray, np.ndarray]:
    # NED -> XYZ, and skip pose[0]: voxel t accumulates events between frame t-1 and t.
    PERM = [1, 2, 0, 4, 5, 3, 6]
    poses = np.loadtxt(osp.join(scene_path, "pose_left.txt"), delimiter=" ")[1::stride, PERM]
    # TartanAir-Events has no real timestamp file; the eval script uses the frame index
    # as timestamp (fixed simulated rate, see FREQ=50 in eval_tartan_evs.py).
    tss_us = np.arange(poses.shape[0], dtype=np.float64)
    return tss_us, poses


def _mvsec_iterator(scene_path: str, side: str = "left", stride: int = 1, timing: bool = False, **_) -> EventIterator:
    return mvsec_evs_iterator(scene_path, side=side, stride=stride, timing=timing, H=260, W=346)


def _mvsec_gt(scene_path: str, side: str = "left", **_) -> Tuple[np.ndarray, np.ndarray]:
    tss_us, poses = load_mvsec_traj(scene_path, side=side)
    return tss_us, poses


def _fpv_iterator(scene_path: str, stride: int = 1, timing: bool = False, tss_gt_us: Optional[np.ndarray] = None, **_) -> EventIterator:
    return fpv_evs_iterator(scene_path, stride=stride, timing=timing, H=260, W=346, tss_gt_us=tss_gt_us)


def _fpv_gt(scene_path: str, **_) -> Tuple[np.ndarray, np.ndarray]:
    # only scenes whose name ends in "_with_gt" ship ground truth, see splits/fpv/fpv_val.txt
    gt_file = osp.join(scene_path, "stamped_groundtruth_us_cam.txt")
    if not osp.isfile(gt_file):
        raise FileNotFoundError(f"No ground truth for this FPV scene (expected {gt_file}); "
                                 f"only '*_with_gt' scenes have it.")
    return load_gt_us(gt_file)


def _tumvie_iterator(scene_path: str, camID: int = 2, stride: int = 1, timing: bool = False, **_) -> EventIterator:
    return tumvie_evs_iterator(scene_path, camID=camID, stride=stride, timing=timing, H=720, W=1280)


def _tumvie_gt(scene_path: str, **_) -> Tuple[np.ndarray, np.ndarray]:
    return load_tumvie_traj(osp.join(scene_path, "mocap_data.txt"))


@dataclass(frozen=True)
class _DatasetSpec:
    height: int
    width: int
    default_config: str
    default_split: str
    make_iterator: Callable[..., EventIterator]
    load_gt: Callable[..., Tuple[np.ndarray, np.ndarray]]


_REGISTRY = {
    Dataset.TARTAN: _DatasetSpec(480, 640, "config/default.yaml", "splits/tartan/tartan_val.txt", _tartan_iterator, _tartan_gt),
    Dataset.MVSEC: _DatasetSpec(260, 346, "config/eval_mvsec.yaml", "splits/mvsec/mvsec_val.txt", _mvsec_iterator, _mvsec_gt),
    Dataset.FPV: _DatasetSpec(260, 346, "config/eval_fpv.yaml", "splits/fpv/fpv_val.txt", _fpv_iterator, _fpv_gt),
    Dataset.TUMVIE: _DatasetSpec(720, 1280, "config/eval_tumvie.yaml", "splits/tumvie/tumvie_val.txt", _tumvie_iterator, _tumvie_gt),
}


def list_scenes(dataset: Dataset, split_file: Optional[str] = None) -> List[str]:
    """Scene names in the given (or default) split file, e.g. for Dataset.TARTAN -> ['office2/Hard/P010', ...]."""
    spec = _REGISTRY[dataset]
    path = split_file or (ROOT / spec.default_split)
    return [s for s in Path(path).read_text().split() if s and "#" not in s]


def get_event_iterator(dataset: Dataset, datapath: str, scene: str, **kwargs) -> EventIterator:
    """
    Yields (voxel [bins,H,W] cuda tensor, intrinsics [fx,fy,cx,cy] cuda tensor, timestamp_us) per
    frame -- the event-voxel representation ("events" / "event_frames") DEVO consumes.

    kwargs are forwarded to the per-dataset loader, e.g. stride=, timing=, scale= (tartan only),
    side= (mvsec), camID= (tumvie), tss_gt_us= (fpv, to auto-crop to the GT time range).
    """
    spec = _REGISTRY[dataset]
    scene_path = osp.join(datapath, scene)
    return spec.make_iterator(scene_path, **kwargs)


def get_ground_truth(dataset: Dataset, datapath: str, scene: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (timestamps_us [N], poses_xyz_xyzw [N,7]) ground truth for a scene, if available."""
    spec = _REGISTRY[dataset]
    scene_path = osp.join(datapath, scene)
    return spec.load_gt(scene_path, **kwargs)


def save_trajectory_plot(tss_gt_us, poses_gt, tss_est_us, poses_est, out_pdf: str, title: str = "",
                          align: bool = True, correct_scale: bool = True, max_diff_sec: float = 1.0) -> dict:
    """
    Time-syncs `poses_est` against `poses_gt` (evo), aligns the estimate onto the ground truth
    (Umeyama SE3/Sim3), saves a PDF with a 3D (x, y, z) trajectory-comparison plot annotated with
    ATE/rotation metrics, and returns the metrics dict.

    Works with any (timestamps_us, poses[N,7] xyz+xyzw-quat) pair -- not just DEVO's output, so
    you can also use it to score your own SLAM pipeline's final (post loop-closure/PGO) trajectory
    against the same ground truth loaders `get_ground_truth()` gives you.

    If GT and estimate already have the exact same number of poses with identical timestamps (e.g.
    TartanAir, where DEVO returns one interpolated pose per input frame and "timestamps" are just
    frame indices, not real time), they're paired directly index-to-index and `max_diff_sec` is
    unused -- matching `utils/eval_utils.py::ate_real()`'s fast path. Otherwise poses are matched by
    nearest timestamp within `max_diff_sec` (default 1.0s, same tolerance `ate_real()` uses for the
    real-world datasets, where VO keyframes and mocap/IMU ground truth don't share a sample rate).

    Explicitly forces matplotlib's non-interactive "Agg" backend, so it works headless (this repo
    can't use evo's own plotting helpers for that reason, see constants.HAS_DESKTOP_ENV).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)
    from copy import deepcopy

    from evo.core import metrics, sync
    from evo.core.geometry import GeometryException
    from evo.core.trajectory import PoseTrajectory3D
    import evo.main_ape as main_ape

    def _to_evo(poses, tss_us) -> PoseTrajectory3D:
        poses = np.asarray(poses)
        return PoseTrajectory3D(
            positions_xyz=poses[:, :3],
            orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],  # xyzw -> wxyz
            timestamps=np.asarray(tss_us, dtype=np.float64) / 1e6,
        )

    traj_gt = _to_evo(poses_gt, tss_gt_us)
    traj_est = _to_evo(poses_est, tss_est_us)

    same_length_and_stamps = (traj_gt.timestamps.shape == traj_est.timestamps.shape
                               and np.allclose(traj_gt.timestamps, traj_est.timestamps))
    if not same_length_and_stamps:
        traj_gt, traj_est = sync.associate_trajectories(traj_gt, traj_est, max_diff=max_diff_sec)

    ape_trans = main_ape.ape(deepcopy(traj_gt), deepcopy(traj_est),
                              pose_relation=metrics.PoseRelation.translation_part,
                              align=align, correct_scale=correct_scale)
    ape_rot = main_ape.ape(deepcopy(traj_gt), deepcopy(traj_est),
                            pose_relation=metrics.PoseRelation.rotation_angle_deg,
                            align=align, correct_scale=correct_scale)

    metrics_dict = {
        "num_poses": traj_gt.num_poses,
        "path_length_m": traj_gt.path_length,
        "ate_rmse_cm": ape_trans.stats["rmse"] * 100,
        "ate_mean_cm": ape_trans.stats["mean"] * 100,
        "ate_median_cm": ape_trans.stats["median"] * 100,
        "ate_std_cm": ape_trans.stats["std"] * 100,
        "rot_rmse_deg": ape_rot.stats["rmse"],
        "rot_mean_deg": ape_rot.stats["mean"],
    }

    # for the plot, align a copy of the estimate the same way ape() did internally
    traj_est_aligned = deepcopy(traj_est)
    if align:
        try:
            traj_est_aligned.align(traj_gt, correct_scale=correct_scale)
        except GeometryException as e:
            print(f"[dataset_api] alignment failed, plotting unaligned: {e}")

    gt_xyz = traj_gt.positions_xyz
    est_xyz = traj_est_aligned.positions_xyz

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], "--", color="gray", label="Ground Truth")
    ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], "-", color="tab:blue", label="Estimate (aligned)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    ax.set_title(title)

    # equal aspect ratio in 3D: matplotlib has no native "equal" for 3D axes, so pad each axis
    # to the same half-range around its midpoint, matching how evo's own 3D plots look
    all_xyz = np.concatenate([gt_xyz, est_xyz], axis=0)
    mins, maxs = all_xyz.min(axis=0), all_xyz.max(axis=0)
    centers = (mins + maxs) / 2
    half_range = max((maxs - mins).max() / 2, 1e-3)
    ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim(centers[2] - half_range, centers[2] + half_range)

    metrics_str = (
        f"ATE RMSE: {metrics_dict['ate_rmse_cm']:.2f} cm | mean: {metrics_dict['ate_mean_cm']:.2f} cm | "
        f"median: {metrics_dict['ate_median_cm']:.2f} cm | std: {metrics_dict['ate_std_cm']:.2f} cm\n"
        f"Rot RMSE: {metrics_dict['rot_rmse_deg']:.2f} deg | "
        f"path length: {metrics_dict['path_length_m']:.2f} m | poses: {metrics_dict['num_poses']}"
    )
    fig.text(0.5, 0.02, metrics_str, ha="center", va="bottom", fontsize=9, family="monospace")
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    print(f"[dataset_api] saved trajectory plot to {out_pdf}\n{metrics_str}")

    return metrics_dict


def run_vo(dataset: Dataset, datapath: str, scene: str, weights: str = "DEVO.pth",
           config_path: Optional[str] = None, viz: bool = False, timing: bool = False,
           plot_path: Optional[str] = None, plot_title: Optional[str] = None,
           align: bool = True, correct_scale: bool = True, max_diff_sec: float = 1.0,
           **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs DEVO's visual-odometry front-end over one scene and returns
    (poses_xyz_xyzw [N,7], timestamps_us [N]) -- the estimated trajectory to hand to your own
    loop-closure / pose-graph back-end.

    kwargs are forwarded to the event iterator (see get_event_iterator).

    If `plot_path` is given and the dataset/scene has ground truth, also saves a PDF trajectory
    comparison plot with ATE/rotation metrics via `save_trajectory_plot()` (align/correct_scale/
    max_diff_sec control that alignment+sync; see `save_trajectory_plot` docstring). Missing
    ground truth (e.g. non-"_with_gt" FPV scenes) just skips the plot with a warning, it doesn't
    raise.
    """
    spec = _REGISTRY[dataset]
    cfg = _default_cfg.clone()
    cfg.merge_from_file(config_path or str(ROOT / spec.default_config))
    if dataset is Dataset.TUMVIE:
        cfg.camID = kwargs.get("camID", 2)

    # DEVO's patch selector samples stochastically; seed for reproducible trajectories/metrics,
    # matching the source repo's eval scripts (they all call this before evaluating).
    torch.manual_seed(1234)

    scene_path = osp.join(datapath, scene)
    iterator = spec.make_iterator(scene_path, timing=timing, **kwargs)
    poses, tstamps, _ = run_voxel(scene_path, cfg, weights, viz=viz, iterator=iterator,
                                   timing=timing, H=spec.height, W=spec.width)

    if plot_path is not None:
        try:
            tss_gt, poses_gt = get_ground_truth(dataset, datapath, scene, **kwargs)
        except FileNotFoundError as e:
            print(f"[dataset_api] skipping trajectory plot, no ground truth: {e}")
        else:
            save_trajectory_plot(tss_gt, poses_gt, tstamps, poses, plot_path,
                                  title=plot_title or f"{dataset.value}: {scene}",
                                  align=align, correct_scale=correct_scale, max_diff_sec=max_diff_sec)

    return poses, tstamps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=[d.value for d in Dataset])
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--scene", default=None, help="defaults to the first scene in the val split")
    parser.add_argument("--weights", default="DEVO.pth")
    parser.add_argument("--plot", metavar="OUT.pdf", default=None,
                         help="save a GT-vs-estimate trajectory comparison PDF with ATE/rotation metrics")
    args = parser.parse_args()

    ds = Dataset(args.dataset)
    scene = args.scene or list_scenes(ds)[0]
    print(f"[{ds.value}] scene={scene}")

    n = 0
    for voxel, intrinsics, t in get_event_iterator(ds, args.datapath, scene):
        n += 1
    print(f"[{ds.value}] iterated {n} event frames")

    try:
        tss_gt, poses_gt = get_ground_truth(ds, args.datapath, scene)
        print(f"[{ds.value}] ground truth: {poses_gt.shape[0]} poses")
    except FileNotFoundError as e:
        print(f"[{ds.value}] no ground truth: {e}")

    poses_est, tss_est = run_vo(ds, args.datapath, scene, weights=args.weights, plot_path=args.plot)
    print(f"[{ds.value}] DEVO VO estimate: {poses_est.shape[0]} poses")
