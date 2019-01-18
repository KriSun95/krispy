I'm following this website:
'https://packaging.python.org/tutorials/packaging-projects/'
to help set this up.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="krispy",
    version="0.0.1",
    author="Kristopher Cooper",
    author_email="k.cooper.2@research.gla.ac.uk",
    description="This package is really only for personal use to work with, primarily, AIA and NuSTAR solar data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KriSun95/krispy/edit/master/krispy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
