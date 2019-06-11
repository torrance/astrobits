import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="astrobits",
    version="0.0.1",
    author="Torrance Hodgson",
    author_email="torrance.hodgson@postgrad.curtin.edu.au",
    description="An assortment of personal, astronomy-related utility functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/torrance/astrobits",
    packages=setuptools.find_packages(),
)
