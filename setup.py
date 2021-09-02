#!/usr/bin/env python

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os.path


with open("README.rst", 'r') as f:
    long_description = f.read()

# Build cython files as extensions
def build_extensions(pkg_name, major_submodules):
    cwd = os.path.abspath(os.path.dirname(__file__))
    extensions = []
    for subm in major_submodules:
        for f in os.listdir(os.path.join(cwd, pkg_name, subm.replace(".", "/"))):
            if f.endswith(".pyx"):
                filename = os.path.splitext(f)[0]
                ext_name = f"{pkg_name}.{subm}.{filename}"
                ext_path = os.path.join(pkg_name, subm.replace(".", "/"), f)
                extensions.append(Extension(ext_name, [ext_path]))

    return extensions

extensions = build_extensions("pomdp_py", ["framework",
                                           "algorithms",
                                           "representations.distribution",
                                           "representations.belief"])
extensions += [
    Extension("pomdp_problems.tiger.cythonize", ["pomdp_problems/tiger/cythonize/tiger_problem.pyx"]),
    Extension("pomdp_problems.rocksample.cythonize", ["pomdp_problems/rocksample/cythonize/rocksample_problem.pyx"])
]

setup(name='pomdp-py',
      packages=find_packages(),
      version='1.2.4.6',
      description='Python POMDP Library.',
      long_description=long_description,
      long_description_content_type="text/x-rst",
      install_requires=[
          'Cython',
          'numpy',
          'scipy',
          'matplotlib',
          'pygame',        # for some tests
          'opencv-python',  # for some tests
          'networkx',
          'pygraphviz'
      ],
      license="MIT",
      author='Kaiyu Zheng',
      author_email='kzheng10@cs.brown.edu',
      keywords = ['Partially Observable Markov Decision Process', 'POMDP'],
      ext_modules=cythonize(extensions,
                            build_dir="build",
                            compiler_directives={'language_level' : "3"}),
      package_data={"pomdp_py": ["*.pxd", "*.pyx"],
                    "pomdp_problems": ["*.pxd", "*.pyx"]},
      zip_safe=False
)

