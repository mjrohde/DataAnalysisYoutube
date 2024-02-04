import os
from setuptools import setup

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = f"{lib_folder}/requirements.txt"
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()
setup(name = "Analysis", installed_requires = install_requires)