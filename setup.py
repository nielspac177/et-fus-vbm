from setuptools import setup, find_packages

setup(
    name="etfvbm",
    version="0.1.0",
    description="VBM of MRgFUS thalamotomy for essential tremor, from deepmriprep outputs",
    author="Niels Pacheco-Barrios",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy", "pandas", "scipy", "nibabel", "nilearn>=0.10",
        "statsmodels", "matplotlib", "seaborn", "pyyaml", "natsort", "tqdm",
    ],
)
