import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="corescore", # Replace with your own username
    version="0.0.2",
    author="Zayad AlZaher, Jo Walsh",
    author_email="jowalsh@bgs.ac.uk",
    description="Detect and analyse fragmentation in core photos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BritishGeologicalSurvey/CoreScore",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: LGPL v3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

