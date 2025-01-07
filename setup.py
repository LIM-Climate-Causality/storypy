from setuptools import setup, find_packages

setup(
    name='storypy',
    version='0.1.0',
    description='A Python package for climate storylines',
    author='Richard Alawode',
    author_email='richard.alawode@uni-leipzig.de',
    license='MIT',
    packages=find_packages(),
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
    ],
    url='',
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ],
)