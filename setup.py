from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name="QuantumInformation",
     version='1.2',
     description='Quantum Information and Computation package',
     author = 'M. S. Ramkarthik and Pranay Barkataki',
     email = 'pranay.barkataki@gmail.com',
     packages=['QuantumInformation'],
     long_description=long_description,
     long_description_content_type='text/markdown',
     zip_safe=False, 
     install_requires = ["numpy >= 1.25.2", "scipy >= 1.11.1"],
     extras_require = {"dev": ["pytest >= 7.0", "twine >= 4.0.2"]}, 
     python_requires = ">= 3.7.9")
