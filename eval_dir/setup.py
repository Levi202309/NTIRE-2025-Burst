from setuptools import setup, find_packages

setup(
    name='efficient_burst_hdr_and_restoration',
    version='1.0.0',
    description='A scoring program for Efficient Burst HDR and Restoration in NTIRE challenge',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'skimage',
    ],
)