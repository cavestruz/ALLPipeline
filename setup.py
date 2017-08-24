#  See: http://www.siafoo.net/article/77#id10 for more examples                                                                                                                                                     
from setuptools import setup, find_packages



setup(
    name='ALLPipeline',
    version='0.1dev',
    description='Machine Learning Pipeline',
    author='Camille Avestruz, Matthew Lightman, Hanjue Zhu',
    author_email='camille.avestruz@yale.edu',
    url='https://github.com/cavestruz/ALLPipeline',
    packages=find_packages(),
    license='The MIT License',
    long_description=open('README.md').read(),
)
