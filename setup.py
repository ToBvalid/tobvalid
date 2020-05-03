from setuptools import setup, find_packages

setup(name='tobevalid',
      version='0.1',
      description='',
      url='https://github.com/Lekaveh/BFactor.git',
      author='Kaveh Babai',
      author_email='lekaveh@gmail.com',
      license='MIT',
      packages=find_packages(include=['tobevalid', 'tobevalid.*']),
      zip_safe=False,
      package_data={'tobevalid': ['templates/albe1.txt','templates/albe2.txt']}
      )