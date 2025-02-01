from setuptools import setup, find_packages

setup(
    name="Neurocore",
    version="0.0.1",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scikit-learn',
        'pandas',
        'pybind11'
    ]
)


