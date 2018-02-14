# -*- coding: utf-8 -*-


import os
from setuptools import setup

import numphy


this_dir = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(this_dir, "README.rst"), "r") as f:
    long_description = f.read()

keywords = ["numpy", "physics", "particle", "vector", "lorentz"]

classifiers = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Development Status :: 4 - Beta",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
]

install_requires = []
with open(os.path.join(this_dir, "requirements.txt"), "r") as f:
    install_requires.extend(line.strip() for line in f.readlines() if line.strip())

setup(
    name=numphy.__name__,
    version=numphy.__version__,
    author=numphy.__author__,
    author_email=numphy.__email__,
    description=numphy.__doc__.strip(),
    license=numphy.__license__,
    url=numphy.__contact__,
    keywords=" ".join(keywords),
    classifiers=classifiers,
    long_description=long_description,
    install_requires=install_requires,
    zip_safe=False,
    packages=["numphy"],
    package_data={
        "": ["LICENSE", "requirements.txt", "README.rst"],
    },
)
