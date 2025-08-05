# cython 빌드를 위한 setup.py 코드
from setuptools import setup
from Cython.Build import cythonize # pip install cython을 통해서 Cython 설치 필요
import numpy

setup(
    ext_modules = cythonize("cython_module/c_example.pyx"),
    include_dirs=[numpy.get_include()]
)