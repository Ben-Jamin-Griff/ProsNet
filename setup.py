import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'ProsNet',
    version = '0.0.4',
    author = 'Benjamin Griffiths',
    description = 'A package for processing activPAL activity monitor data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ben-Jamin-Griff/ProsNet",
    project_urls={
        "Bug Tracker": "https://github.com/Ben-Jamin-Griff/ProsNet/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    py_modules=["ProsNet"],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src"),
    python_requires = ">=3.6",
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