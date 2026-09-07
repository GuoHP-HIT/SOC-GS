# SOC-GS: Cross-Spectral Gaussian Splatting with Spatial Occupancy Consistency

Official code release for the AAAI-25 paper
**“Cross-Spectral Gaussian Splatting with Spatial Occupancy Consistency”**
(Haipeng Guo, Huanyu Liu, Jiazheng Wen, Junbao Li — Harbin Institute of
Technology).

This repository contains the complete training / evaluation / visualisation
code of both experiment pipelines of the paper in a **single installable
package**:

| directory | experiment | modelled spectral channels | features per point |
|---|---|---|---|
| `socgs/rgb_ms`    | **bimodal**  RGB + multispectral (MS) | R, G, B, MS                | 72  |
| `socgs/rgb_ir_ms` | **trimodal** RGB + infrared (IR) + MS | R, G, B, MS, IR            | 88  |

Everything shared by the two pipelines (scene representation, camera
structures, losses, pose estimation, metrics, checkpoint helpers) lives in
`socgs/common`; experiment-specific code (rasteriser, densification
controller, trainer, dataset loader) lives in the experiment packages. The
configuration file selects the experiment automatically
(`config/rgb_ms/...` vs `config/rgb_ir_ms/...`), so the original command-line
interface is kept unchanged.

**Weight compatibility.** Scene checkpoints of the original implementation
(the parquet files written during training) and the optimised cross-spectral
camera poses (the `ms_iter_*` / `ir_iter_*` `.npy` snapshots) are loaded
**directly and bit-exactly** by this code base — the attribute layout,
parquet column names, pose file naming and the `nn.Module` state-dict keys are
identical (see [Checkpoint compatibility](#checkpoint-compatibility)). No
retraining or format conversion is needed to continue experiments or to render
existing checkpoints.

---

## What is SOC-GS?

Using images captured by cameras with different light-spectrum sensitivities,
SOC-GS trains a *unified* 3D Gaussian representation that is shared across all
spectral bands (RGB / multispectral / infrared). Key ideas:

* **Shared explicit Gaussian surfaces across spectra** — one point cloud with
  per-channel spherical-harmonic colours (R/G/B/MS(/IR)) is optimised
  jointly, instead of fitting one model per spectrum;
* **Matching–optimising cross-spectral pose estimation** — the co-located MS
  and IR cameras share the viewpoint of the RGB camera but their poses are
  initialised from the RGB pose and refined per view (LoFTR matching +
  PnP initialisation, followed by a learnable-pose bundle adjustment and
  joint optimisation with the scene);
* **Improved adaptive densification for non-overlapping fields of view** —
  densification candidates are accumulated per modality and applied jointly
  across modalities so that regions visible to only one camera are still
  covered.

The whole pipeline is implemented in pure Python on top of
[Taichi](https://github.com/taichi-dev/taichi)
(rasteriser kernels) and PyTorch (optimisation), following the structure of
the unofficial Taichi 3DGS implementation by
[wanmeihuali/taichi_3d_gaussian_splatting](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting).

> Reproducibility note: quantitative results in the paper were computed with
> the evaluation pipeline below (same-view cross-spectral rendering +
> MATLAB-compatible PSNR/SSIM). Training-time validation curves additionally
> log the torch-based PSNR and the pytorch-msssim SSIM, which are the metrics
> reported in the TensorBoard logs — both metric families are provided in
> `socgs/common/metrics.py`.

---

## Installation

The code is tested with **Python 3.10, PyTorch 2.1 (CUDA), Taichi 1.7** on
Ubuntu with an NVIDIA GPU (CUDA driver required for training/rendering).

```bash
# 1. create an environment with a CUDA-enabled PyTorch
conda create -n socgs python=3.10
conda activate socgs
conda install pytorch=2.1.0 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 2. install the remaining dependencies and the package itself
git clone <this-repository> socgs && cd socgs
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` additionally lists `opencv-python`, `kornia` (used by the
LoFTR-based pose initialisation) and `nvidia-ml-py` (GPU-memory reporting
during the render-cost benchmark; skipped automatically when unavailable).
Taichi compiles its kernels on first use (a few minutes).

> Headless machines: training/rendering do not need a display. The optional
> interactive `visualizer.py` needs a desktop session.

---

## Repository layout

```
.
├── gaussian_point_train.py             # train entry (original CLI)
├── gaussian_point_render.py            # frame rendering entry (original CLI)
├── cross_spectral_render_w_same_view.py# same-view render + metrics + cost
├── calculate_PSNR_SSIM.py              # offline PSNR/SSIM for image pairs
├── visualizer.py                       # interactive RGB|MS|IR viewer
├── parquet_to_ply.py                   # checkpoint -> PLY exporter
├── config/
│   ├── rgb_ms/<scene>_train.yaml       # bimodal experiment (16 scenes)
│   └── rgb_ir_ms/<scene>_train.yaml    # trimodal experiment (16 scenes)
├── socgs/
│   ├── common/                         # shared modules (both experiments)
│   │   ├── GaussianPointCloudScene.py  # representation + parquet/PLY IO
│   │   ├── Camera.py, utils.py, SphericalHarmonics.py, LossFunction.py
│   │   ├── pose.py                     # learnable cross-spectral pose (BA)
│   │   ├── metrics.py                  # PSNR/SSIM (training + MATLAB-style)
│   │   └── checkpoints.py              # pose snapshot discovery/loading
│   ├── rgb_ms/                         # bimodal model stack
│   │   ├── GaussianPoint3D.py, GaussianPointCloudRasterisation.py,
│   │   ├── GaussianPointAdaptiveController.py, GaussianPointTrainer.py,
│   │   ├── ImagePoseDataset.py, render.py
│   └── rgb_ir_ms/                      # trimodal model stack (same names)
├── scripts/                            # batch training/rendering examples
└── tools/prepare_colmap.py             # optional: build a scene from COLMAP
```

---

## Dataset

Each scene `<scene>` (one of `bins, black, bluechair, cvlab, dino, fruits,
green, hall, hall2, hall3, hall4, orange, penguin, penguin2, puppets, tech`)
is stored as:

```
datasets/<scene>/
├── train.json         # RGB training views (25) -- poses & intrinsics
├── train_all.json     # all RGB views (30, incl. MS/IR pose-optimisation set)
├── val.json           # RGB validation views (5)
└── point_cloud.parquet# initial sparse point cloud (COLMAP, x/y/z/r/g/b)
datasets/split_source/<scene>/
├── rgb_imgs/          # RGB  images       4112 x 3008 (PNG)
├── ms_imgs/           # multispectral     510  x 254  (PNG, same view)
├── ir_imgs/           # infrared          1023 x 1023 (PNG, same view, trimodal only)
└── sparse/            # COLMAP reconstruction of the RGB views
```

The JSON files list **only the RGB views**; the MS/IR image of the same
viewpoint is loaded by replacing the directory name (`rgb_imgs` →
`ms_imgs` / `ir_imgs`), which is possible because the capture rig’s cameras
are co-located. The MS and IR intrinsics of the rig are fixed constants
(defined in `socgs/rgb_ms/ImagePoseDataset.py` / `socgs/rgb_ir_ms/ImagePoseDataset.py`)
and the recorded RGB poses come from COLMAP. The validation views of every
scene are the frame indices 8, 10, 14, 20, 22 (row indices into the COLMAP
model), i.e. the image files `0009/0011/0015/0021/0023.png` when the sequence
is numbered from 1.

**Downloading the data.** The preprocessed cross-spectral dataset used in the
paper is released separately (see the paper's project page). Place it so that
`datasets/<scene>/...` resolves from the repository root. To prepare a new
RGB scene from a COLMAP reconstruction you can use
`tools/prepare_colmap.py` (writes `train.json`/`train_all.json`/`val.json`
with the standard split and `point_cloud.parquet`); the MS/IR images need to
be registered to the RGB views of the capture rig.

---

## Training

```bash
# bimodal experiment (RGB + MS), e.g. scene "orange"
python gaussian_point_train.py --train_config config/rgb_ms/orange_train.yaml

# trimodal experiment (RGB + IR + MS)
python gaussian_point_train.py --train_config config/rgb_ir_ms/orange_train.yaml
```

The experiment is selected automatically from the config path
(`config/rgb_ms/…` → `rgb_ms`, `config/rgb_ir_ms/…` → `rgb_ir_ms`); pass
`--experiment` to override. `--gen_template_only` dumps a default
`TrainConfig` template instead of training.

All scenes at once: `bash scripts/train_rgb_ms_all_scenes.sh` (or
`train_rgb_ir_ms_all_scenes.sh`).

### Training pipeline

Both experiments follow the same multi-stage schedule (defaults are written
explicitly into the shipped yamls; iteration counts per scene are in the
per-scene configs, e.g. 50 000 / 60 000 for the trimodal experiment):

| stage | iterations (defaults) | what is optimised |
|---|---|---|
| **s1 — single-modality warm-up** | 0 … 5 000 | RGB-only 3DGS (geometry + RGB SH) with adaptive densification |
| **coarse pose initialisation** | once at iteration 5 000 | MS (and IR) poses per view: LoFTR matching of the rendered RGB view with the MS/IR image → ray/ellipsoid intersection → PnP; the per-channel SH colours are seeded from the RGB SH |
| **s2p2 — pose bundle adjustment** | 5 001 … 10 000 | MS/IR SH colours + the learnable per-view MS/IR poses (geometry frozen), alternating the modalities |
| **s2p3 — joint optimisation** | 10 001 … `num_iterations` | RGB/MS(/IR) alternating updates of geometry, colours and poses with joint cross-spectral densification |
| **ending** | last 500 iterations per modality | fine-tune the SH colours of one modality at a time |

The optimised poses are parametrised as axis-angle + translation per camera
(`socgs/common/pose.py::LearnPose`). Adam optimisers with exponential LR
decay are used for features, positions and poses (rates/decay in the yaml).

### Outputs

Everything is written under the config’s `summary_writer_log_dir`
(`logs/rgb_ms/<scene>` / `logs/rgb_ir_ms/<scene>` in the shipped configs):

| artifact | name |
|---|---|
| full scene snapshots | `scene_<iter>.parquet` (every validation) |
| best / final scenes | `best_scene.parquet` (by validation PSNR sum over the modelled channels), `final_scene.parquet` |
| learned MS/IR poses | `<ms|ir>_iter_<iter:06d>_r_.npy` (quaternion, (N,1,4)), `<ms|ir>_iter_<iter:06d>_t_.npy` (translation, (N,1,3)) |
| TensorBoard logs / images | `events.out.tfevents.*`, `images/` (validation predictions, GT and pose-initialisation keypoint figures) |

TensorBoard: `tensorboard --logdir logs`.

---

## Evaluation (same-view cross-spectral rendering)

```bash
# bimodal: render the validation views and report PSNR/SSIM + render cost
python cross_spectral_render_w_same_view.py --scene orange --render_view rgb \
    --log_dir logs/rgb_ms --output_prefix render --poses_dir datasets
python cross_spectral_render_w_same_view.py --scene orange --render_view ms \
    --log_dir logs/rgb_ms --output_prefix render --poses_dir datasets

# trimodal (adds the IR view)
python cross_spectral_render_w_same_view.py --scene orange --render_view ir \
    --log_dir logs/rgb_ir_ms --output_prefix render --poses_dir datasets
```

* `--render_view rgb` renders with the COLMAP RGB poses; `--render_view ms`
  (and `ir`) renders from the optimised poses — the latest pose snapshot in
  the log directory is used automatically (`--pose_iteration` overrides).
* Every validation view is rasterised once per run with the `val_all` stage
  (RGB+MS(/IR) in one pass); the per-channel predictions are written to
  `<output>/<scene>/<view>_view/{rgb,ms[,ir]}` and the ground-truth views of
  the chosen camera set to `<output>/<scene>/<view>_view/GT`, named
  `0008/0010/0014/0020/0022.png` (view labels of the original release).
* The script prints the average PSNR/SSIM (MATLAB-compatible, computed on the
  saved images) and the average rendering time / GPU memory (first frame
  excluded, matching the paper's render-cost measurement).

Batch scripts: `scripts/render_rgb_ms_all_scenes.sh`,
`scripts/render_rgb_ir_ms_all_scenes.sh`.

Additional tools:

```bash
# offline metrics on arbitrary image pairs
python calculate_PSNR_SSIM.py --pred_dir render/orange/rgb_view/rgb --gt_dir render/orange/rgb_view/GT

# legacy frame renderer (frame_000.png style outputs, arbitrary parquet/json)
python gaussian_point_render.py --parquet_path logs/rgb_ms/orange/best_scene.parquet \
    --poses datasets/orange/val.json --output_prefix result/orange --render_view rgb

# export a checkpoint to PLY (bimodal and trimodal checkpoints both supported)
python parquet_to_ply.py --parquet_path logs/rgb_ir_ms/orange/best_scene.parquet --ply_path out.ply
```

---

## Visualisation

```bash
python visualizer.py --parquet_path_list logs/rgb_ms/orange/best_scene.parquet --camera_modality RGB
```

Opens an interactive viewer (WASD/mouse navigation, `q/e` rotate, `h/p`
hide/show points). The number of panels follows the checkpoint: bimodal
scenes show RGB + MS, trimodal scenes show RGB + MS + IR. `--camera_modality`
selects the initial camera (intrinsics/image size).

---

## Checkpoint compatibility

The released checkpoints and pose snapshots produced by the original
implementations (`CS3DGS_RGB_MS` and `CS3DGS_RGB_IR_MS` trees) can be loaded
by this repository without conversion:

* **Scene checkpoint** (parquet written by `GaussianPointCloudScene.to_parquet`):
  columns `x,y,z` + `cov_q0-3, cov_s0-2, alpha0` + per-channel SH columns
  `{r,g,b,ms[,ir]}_sh0..15`. The `nn.Module` state dict uses the same keys as
  the original code (`point_cloud`, `point_cloud_features`,
  `point_invalid_mask`, `point_object_id`), and the feature layout is
  unchanged: 4 quaternion + 3 log-scale + 1 alpha + 16 SH coefficients per
  modelled channel (72 features for 4 channels, 88 for 5). Loaders detect the
  channel count automatically, so both bimodal and trimodal checkpoints are
  handled by the same code.
* **Pose snapshots** (`.npy`): `<modal>_iter_<iter:06d>_r_/t_.npy`, modal
  `ms` (bimodal) / `ms` and `ir` (trimodal), rows indexed by the absolute
  `camera_id`. The evaluation scripts pick the latest snapshot automatically
  instead of relying on hard-coded iteration numbers.
* **Configs**: yaml keys/values are those of the original release (paths
  adjusted to `logs/rgb_ms/...` / `logs/rgb_ir_ms/...`). One historic typo was
  cleaned up without changing the effective behaviour: the unused
  `position_learning_rateo` key (which was silently ignored, leaving the
  default 1e-5 active) was replaced by an explicit `position_learning_rate:
  0.00001`, and the effective stage defaults (`warmup` 5000, `fine BA` 10000,
  `ending` 500, `pose LR` 1e-3) are now written into the configs.

### Intentional fixes over the original trees

The original experiment trees contained several defects in code paths that
were not exercised by the published experiments; they are repaired here
“to the obviously intended behaviour”, without touching any validated
numerical path:

* rendering the MS/IR camera views of `gaussian_point_render.py` and
  `cross_spectral_render_w_same_view.py` referenced undefined
  `camera_info_ms/ir` names (NameError). The renderers now build the
  per-modality camera info in the constructor;
* render scripts hard-coded a specific pose-snapshot iteration
  (`030500`/`036500`/`049000`/`050000`), some of which never exist in the
  logs — the latest available snapshot is used automatically
  (`--pose_iteration` to pin one);
* `Scene.to_ply` assumed an RGB-only layout and crashed for the 72/88-feature
  checkpoints — the exporter now writes all spectral channels;
* the trimodal `best_scene.parquet` criterion added the RGB PSNR twice and
  omitted IR (RGB+MS+RGB) — fixed to RGB+MS+IR;
* the default `PointCloudSceneConfig.add_sphere` no longer appends helper
  points when loading a *trained* checkpoint (which previously produced
  NaN-feature rows); `add_sphere` only applies when initialising a fresh
  scene from a raw point cloud, as in training;
* dead code and experiment residue were removed (see below) and the
  duplicated metric/pose implementations were unified in `socgs/common`.

### What is intentionally not included

The original trees were forked from
[wanmeihuali/taichi_3d_gaussian_splatting](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting)
and carried upstream-only content that is unrelated to the paper experiments.
It is omitted here: `benchmark/`, `ci/`, `tests/`, `scratch/`,
`Dockerfile.aws`, the KITTI/Instant-NGP/ellipse-path tools, and a few
one-off experiment scripts (`stack4infrared.py` (fully commented-out file),
`stack_for_test.py`, `crop_and_resize_image.py`). Ground-truth cropping for
the legacy RGB comparison pipeline is no longer needed because the same-view
renderer writes GT and predictions at identical sizes.

---

## Code overview

| module | responsibility |
|---|---|
| `socgs/common/GaussianPointCloudScene.py` | point cloud + attribute tensors, initialisation (kNN covariance, SH seeding), parquet/PLY I/O, feature-layout constants |
| `socgs/common/Camera.py` | `CameraInfo` (+ per-modality intrinsics/sizes) |
| `socgs/common/SphericalHarmonics.py` | SH evaluation / jacobians (Taichi) |
| `socgs/common/LossFunction.py` | `(1-λ)L1 + λ(1-SSIM)` + optional scale regularisation |
| `socgs/common/utils.py` | Taichi + torch math (quaternions, SE3, 2D-Gaussian density…) |
| `socgs/common/pose.py` | axis-angle tools and `LearnPose` (learnable per-view poses) |
| `socgs/common/metrics.py` | torch PSNR/SSIM (training validation) and MATLAB-compatible PSNR/SSIM (evaluation) |
| `socgs/common/checkpoints.py` | pose-snapshot discovery/loading, GPU-memory helper |
| `socgs/<experiment>/GaussianPoint3D.py` | per-point Taichi struct and projection/covariance/colour math |
| `socgs/<experiment>/GaussianPointCloudRasterisation.py` | differentiable tiled rasteriser (forward + backward kernels, per-stage gradient masks) |
| `socgs/<experiment>/GaussianPointAdaptiveController.py` | per-modality accumulation and (cross-spectral) densification |
| `socgs/<experiment>/GaussianPointTrainer.py` | multi-stage training / validation / checkpointing |
| `socgs/<experiment>/ImagePoseDataset.py` | image+pose dataset with MS(/IR) directory pairing |
| `socgs/<experiment>/render.py` | same-view rendering/evaluation orchestration |

---

## Citation

```bibtex
@inproceedings{guo2025cross,
  title={Cross-Spectral Gaussian Splatting with Spatial Occupancy Consistency},
  author={Guo, Haipeng and Liu, Huanyu and Wen, Jiazheng and Li, Junbao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={3229--3237},
  year={2025},
  doi={10.1609/aaai.v39i3.32333}
}
```

## Acknowledgements

The rasteriser and the general training framework build on the unofficial
Taichi implementation of 3D Gaussian Splatting by
[wanmeihuali/taichi_3d_gaussian_splatting](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting)
(3D Gaussian Splatting for Real-Time Radiance Field Rendering, Kerbl et al.,
SIGGRAPH 2023), and on [Taichi](https://github.com/taichi-dev/taichi).

## License

Apache-2.0 (see `LICENSE`). The paper is © AAAI; when you use this code for
research purposes please cite the paper above.
