import os
import sys
import platform
from os import sep as dirsep
from os.path import isfile, join

from setuptools import setup
from setuptools import Extension

if sys.version_info[:2] <= (3, 8):
    sys.stderr.write('Python 3.8 and older is not supported\n')
    sys.exit()

INSTALL_REQUIRES=['setuptools']

__version__ = ''
with open('cryodrgn/__init__.py') as inp:
  for line in inp:
      if line.startswith('__version__'):
          exec(line.strip())
          break

with open('README.md') as inp:
    long_description = inp.read()


PACKAGES = ['cryodrgn',
            'cryodrgn.commands',
            'cryodrgn.sync_batchnorm',
            'cryodrgn.templates',
            'analysis_scripts',
            'testing',
            'testing.data',
            'utils']
PACKAGE_DATA = {
    'testing': ['data/*.cs',
                'data/*.pkl',
                'data/*.star',
                'data/*.mrcs',
                'data/*.txt']
}

PACKAGE_DIR = {}
for pkg in PACKAGES:
    PACKAGE_DIR[pkg] = join(*pkg.split('.'))

setup(
    name='opusdsd',
    version=__version__,
    author='Zhenwei Luo, Fengyun Ni, Qinghua Wang, Jianpeng Ma',
    author_email='jpma@fudan.edu.cn',
    description='OPUS-DSD: deep structural disentanglement for cryo-EM single-particle analysis',
    long_description=long_description,
    url='',
    packages=PACKAGES,
    #package_dir=PACKAGE_DIR,
    package_data=PACKAGE_DATA,
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    keywords=('Computational biophysics, '
              'Computational models, '
              'Cryoelectron microscopy, '
              'Proteins, '
              'Software'),
    classifiers=[
                 'Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GPL 3.0',
                 'Operating System :: POSIX',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3',
                 'Topic :: Scientific/Engineering :: Computational biophysics',
                 'Topic :: Scientific/Engineering :: Computer science',
                ],
    #scripts=SCRIPTS,
    #install_requires=INSTALL_REQUIRES,
    #provides=['cryodrgn ({0:s})'.format(__version__)]
)
