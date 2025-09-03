#region modules
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
#endregion

#region classes
class NumpyInclude(build_ext):
    def finalize_options(self):
        super().finalize_options()
        for ext in self.extensions:
            ext.include_dirs.append(np.get_include())
            ext.extra_compile_args = ['-O3', '-fopenmp']

#endregion

#region variables
cython_extension  =  Extension(
    'paraprof.cython_ext',
    sources=['src/cython/main_cython.pyx'],
)

c_extension = Extension(
    'paraprof.c_ext',
    sources=['src/c/main_c.c']
)

extensions = [

]
#endregion

#region functions
setup(
    name='paraprof',
    version='1.0.0',
    description='Parallel profiler',
    long_description='Parallel profiler',
    author='Krishnaa Vadivel',
    author_email='krishnaa.vadivel@yale.edu',
    packages=find_packages(where='src/python'),
    package_dir={'': 'src/python'},
    requires=[
        'cython',
        'setuptools',
    ],
    ext_modules=cythonize(extensions),
    entry_points={
        'console_scripts': [
            'paraprof=paraprof.scripts.paraprof:main',
        ],
    },
    cmdclass={
        'build_ext': NumpyInclude,
    }
)
#endregion