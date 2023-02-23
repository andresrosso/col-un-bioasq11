from setuptools import find_packages, setup

setup(
    name='unal-andresrosso-bioasq',
    packages=find_packages(),
    version='0.1.0',
    description='BioASQ10 participation code -Universidad Nacional (Bogotá-Colombia)-',
    author='ANDRES ROSSO / UN (BOGOTA-COLOMBIA)',
    url='https://github.com/andresrosso',
    license='MIT',
    keywords='development, setup, setuptools',
    python_requires='>=3.7, <4',
    install_requires=[
        'PyYAML',
        'pandas==0.23.3',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0',
        'jupyter'
    ]
)
