import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="videocut",
    version="0.0.1",
    author="Xipeng Xie",
    author_email="xxp.prc@gmail.com",
    description="A package to cut a video automatically",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ageneinair",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)