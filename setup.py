from setuptools import setup, find_packages

setup(
    name='gsvd',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,  # include non-Python files specified in MANIFEST.in
    package_data={'gsvd': ['libgsvd.so', 'libgsvd.dylib', 'gsvd.dll']},
    install_requires=[
        'numpy'
    ],
    author="Your Name",
    description="A minimal Python wrapper for GSVD using LAPACKE_dggsvd3",
)
