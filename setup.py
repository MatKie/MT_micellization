from setuptools import setup, find_packages

setup(
    name='MTM',
    version='0.1.0',
    url='https://tbd.com',
    author='Matthias Kiesel',
    author_email='m.kiesel18@imperial.ac.uk',
    description='Implement Molecular Thermodynamic theory of Micellization',
    packages=find_packages(),    
    install_requires=['numpy', 'matplotlib', 'scipy'],
)
