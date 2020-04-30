from setuptools import setup

setup(name='tobevalid',
      version='0.1',
      description='',
      url='https://github.com/Lekaveh/BFactor.git',
      author='Kaveh Babai',
      author_email='lekaveh@gmail.com',
      license='MIT',
      packages=['tobevalid', 'tobevalid.stats', 'tobevalid.parsers', 'tobevalid.mixture'],
      zip_safe=False,
      include_package_data=True)