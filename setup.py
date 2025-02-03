from setuptools import find_packages, setup

setup(
    name="uc",
    version="0.1",
    python_requires=">=3.5",
    install_requires=["ply", "pytest", "graphviz", "pathlib", "llvmlite"],
    packages=find_packages(),
)
