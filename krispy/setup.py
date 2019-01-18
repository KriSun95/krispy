import setuptools
from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='krispy',
    version='0.0.1',
    author='Kris Cooper',
    author_email='k.cooper.2@research.gla.ac.uk',
    description='A small package with useful function for AIA and NuSTAR data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='N/A, eventually github url',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
