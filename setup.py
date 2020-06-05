"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"

    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from setuptools import setup, find_packages

setup(name='tobvalid',
      version='0.9.1',
      description='Python library and a program for the statistical analysis and validation of ADPs (Atom Displacement Parameters)',
      url=None,
      author='Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov',
      author_email='rmasmaliyeva@gmail.com, lekaveh@gmail.com, garib@mrc-lmb.cam.ac.uk',
      license='MPL-2.0',
      long_description='''\
        ToBvalid is a Python library and a program for the statistical analysis and validation of ADPs (Atom Displacement Parameters). 

        This tool is designed for modelling of ADP distribution and their validation on both global and local levels. Main functionalities of ToBvalid include: 
            • Overall statistical analysis of ADP distribution
            • Parametrisation of ADP distribution (mixture) and validation of the distribution parameters 
            • Search for potential lighter and heavier atoms which may have been modelled wrongly
            • Validation of ligands''',

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
