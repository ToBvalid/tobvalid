"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"

    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from setuptools import setup, find_packages

setup(name='tobvalid',
      version='0.9.1',
      description='',
      url=None,
      author='Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov',
      author_email='rmasmaliyeva@gmail.com, lekaveh@gmail.com, garib@mrc-lmb.cam.ac.uk',
      license='MPL-2.0',
      install_requires=['pandas', 'fire', 'matplotlib',
                        'numpy', 'scipy', 'gemmi>=0.3.8', 'seaborn', 'statsmodels'],
      entry_points={
          "console_scripts": [
              "tobvalid = tobvalid.run:main_func",
          ]},
      packages=find_packages(include=['tobvalid', 'tobvalid.*']),
      zip_safe=False,
      package_data={'tobvalid': ['templates/albe1.txt', 'templates/albe2.txt',
                                 'templates/xx.npy', 'templates/yy.npy', 'templates/albe_kde.npy']},
      python_requires='>=3',
      )
