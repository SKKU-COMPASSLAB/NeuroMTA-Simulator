from setuptools import setup, find_packages

setup(
    name='neuromta',
    version='2.0',
    description='NeuroMTA Simulator is for design space exploration of multi-core NPU architecture',
    author='Seongwook Kim',
    author_email='su8939@skku.edu',
    packages=find_packages(where="srcs"),
    package_dir={"": "srcs"},
)
