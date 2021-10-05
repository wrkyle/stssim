from setuptools import setup

setup(
    name='STSSim',
    version='0.1.0',
    author='Wes Kyle',
    author_email='wes@wrkyle.com',
    packages=['stssim'],
    scripts=[],
    url='https://github.com/wrkyle/stssim',
    license='LICENSE.txt',
    description='Tools to create simulated stochastic time series\' and causal networks thereof.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.17.0",
    ],
)