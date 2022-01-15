from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "QuantumInformation/README.md").read_text()

setup(name="QuantumInformation",
     version='1.1',
     description='Quantum Information and Computation package',
     author = 'M. S. Ramkarthik and Pranay Barkataki',
     email = 'pranay.barkataki@gmail.com',
     packages=['QuantumInformation'],
     long_description=long_description,
     long_description_content_type='text/markdown',
     zip_safe=False) 
