from setuptools import setup
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

# name is not fixed yet
# version 0 (i.e. not ready yet)
setup(name='multiple_relevant_CP',
      version='0.0.1',
      description='A package to detect relevant changes.',
     author='Rupsa Basu',
      author_email='r.basu@utwente.nl',
      long_description_content_type="text/markdown",
      long_description=README,
      license='MIT',
      packages=['multiple_relevant_CP'], # preliminary name
      include_package_data=True,
      install_requires=[
          'numpy',
          'ortools>=9.4',
          'scipy',
          'matplotlib',
           'matplotlib_scalebar',
           'scipy',
           'pandas',
           'multiprocessing',
           'decimal'
      ],
      zip_safe=False)