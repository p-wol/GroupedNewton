from setuptools import setup, find_packages

setup(name='GroupedNewton',
      version='0.0.2',
      description="Variant of Newton's method with grouped parameters",
      author='Pierre Wolinski',
      author_email='pierre.wolinski@normalesup.org',
      url='https://github.com/p-wol/GroupedNewton',
      packages=find_packages('src'),
      package_dir={"": 'src'},
      license='LICENSE',
      long_description=open('README.md').read(),
     )

