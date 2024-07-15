from setuptools import setup, find_packages

setup(
    name='vidar',
    version='0.1',
    packages=find_packages(include=[
        'vidar', 'vidar.*',
        'camviz', 'camviz.*',
        'efm_datasets', 'efm_datasets.*'
    ]),
    install_requires=[
        'torch',
        'numpy',
        'opencv-python',
        'fire'
    ],
)

