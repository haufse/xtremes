from setuptools import setup, find_packages

setup(
    name='xtremes',
    version='0.1.8',
    packages=find_packages(),
    install_requires=[
        # Add your project's dependencies here
        "numpy",
        "scipy",
        "pynverse",
        "tqdm",
        "matplotlib",
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here
            # e.g., 'xtremes-cli=xtremes.cli:main'
        ],
    },
    author='Erik Haufs',
    author_email='erik.haufs@rub.de',
    description='A package for the analysis of extreme events in time series data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/haufse/xtremes',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
