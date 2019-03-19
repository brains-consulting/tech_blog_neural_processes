# -*- coding: utf-8 -*-

import pathlib
import setuptools


def get_requirements():
    req = pathlib.Path("requirements.txt")
    libs = [l.rstrip() for l in req.open(mode="r").readlines()]
    return libs


requirements = get_requirements()
readme = pathlib.Path("README.md").open(mode="r").read()


setuptools.setup(
    # Metadata
    name='npmodel',
    version="1.0.0",
    author='BCI Oshita',
    author_email='oshita.takehito@brains-consulting.co.jp',
    url='https://github.com/brains-consulting/tech_blog_neural_processes',
    description='an implementation of the neural processes',
    long_description=readme,
    license='MIT',

    # Package info
    packages=setuptools.find_packages(exclude=('test', 'train_toy.py', 'train_mnists.py')),

    zip_safe=True,
    install_requires=requirements,
    extras_require={},
)
