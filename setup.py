from setuptools import setup

setup(
    name = 'ProsNet',
    version = '0.0.3',
    author = 'Benjamin Griffiths',
    description = 'A package for processing activPAL activity monitor data.',
    py_modules=["ProsNet"],
    package_dir={'': 'src'},
    install_requires = [
        "requests",
        "pandas",
        "matplotlib",
        "scipy",
        "uos-activpal",
        "numba",
        "seaborn",
        "numpy == 1.20"
    ],
)