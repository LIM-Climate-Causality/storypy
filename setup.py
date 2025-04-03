from setuptools import setup, find_packages

setup(
    name='storypy',
    version='0.1.4',
    description='A Python package for climate storylines',
    author='Richard Alawode',
    author_email='richard.alawode@uni-leipzig.de',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'xarray',
        'matplotlib',
        'pandas',
        'numpy',
        'cartopy',
        'imageio',
        'netcdf4',
        'scipy',
        'regionmask',
        'scikit-learn'
        'ScalarMappable',
        'shapely',
        'fnmatch'
    ],
    url='https://github.com/LIM-Climate-Causality/storypy',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ],
)