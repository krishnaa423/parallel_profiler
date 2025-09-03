#region modules
from setuptools import setup, find_packages
#endregion

#region variables
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
    ],
    entry_points={
        'console_scripts': [
            'paraprof=paraprof.scripts.paraprof:main',
        ],
    },
)
#endregion

#region classes
#endregion