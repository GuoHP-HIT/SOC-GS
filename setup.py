from setuptools import setup, find_packages

setup(
    name="socgs",
    version="1.0.0",
    author="SOC-GS (AAAI-25) authors",
    description=(
        "Cross-Spectral Gaussian Splatting with Spatial Occupancy Consistency "
        "(SOC-GS): official implementation (Taichi-based, RGB+MS bimodal and "
        "RGB+IR+MS trimodal experiments)."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/",
    packages=find_packages(),
    # PyTorch / CUDA / Taichi are usually installed in a dedicated environment
    # (see requirements.txt and the README); installing them through setup.py
    # often breaks an existing environment, so nothing is declared here.
    python_requires=">=3.8",
)
