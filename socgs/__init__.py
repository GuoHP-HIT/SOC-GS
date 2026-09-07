"""SOC-GS: Cross-Spectral Gaussian Splatting with Spatial Occupancy Consistency.

Official code release of the AAAI-25 paper. This package contains the two
experiment pipelines of the paper in a single installable code base:

* ``socgs.rgb_ms``    -- the bimodal RGB + multispectral (MS) experiment;
* ``socgs.rgb_ir_ms`` -- the trimodal RGB + infrared (IR) + MS experiment.

Both pipelines share the *same* underlying representation (see
``socgs.common.GaussianPointCloudScene``): one shared Gaussian surface is
optimised across all spectral channels, with the only difference being the
number of modelled spectral channels (4 vs. 5, i.e. 72 vs. 88 feature slots
per point). ``socgs.common`` holds every module that is identical for the two
experiments (camera structures, spherical harmonics, losses, pose estimation,
metrics, scene storage); experiment-specific code (rasteriser, densification
controller, trainer, dataset loader) lives in the two experiment packages.

Run ``python gaussian_point_train.py --train_config config/rgb_ms/<scene>_train.yaml``
(or ``config/rgb_ir_ms/<scene>_train.yaml``) from the repository root -- the
config path selects the experiment automatically. See README.md for details.
"""

__version__ = "1.0.0"
