from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

exts = [Extension(name='skrrf.tree._tree', sources=['skrrf/tree/_tree.pyx']),
        Extension(name='skrrf.tree._criterion', sources=['skrrf/tree/_criterion.pyx']),
        Extension(name='skrrf.tree._splitter', sources=['skrrf/tree/_splitter.pyx'])]

setup(
    name='skrrf',
    version='1',
    packages=['skrrf', 'skrrf.tree'],
    url='',
    license='',
    author='mmendez12',
    author_email='mickael.mendez@mail.utoronto.ca',
    description='',
    install_requires=['numpy', 'scikit-learn'],
    ext_modules=cythonize(exts),
    include_dirs=[numpy.get_include()]
)