""" Setuptools install-file for ftodtf"""
import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()


def find_dependencies():
    """ Dynamically computes the needed dependencies.
        This way we can list tensorflow as dependencie, but allow the user to keep an existing tensorflow-gpu installation
    """
    deps = ["numpy", "nltk", "fnvhash", "tqdm", "psutil"]
    try:
        # pylint: disable=unused-variable
        import tensorflow
    # Only list tensorflow as requirement if not already installed
    except ImportError:
        deps.append("tensorflow")
    return deps


setuptools.setup(
    name="ftodtf",
    version="0.0.1",
    description="Run FastText on distributed TensorFlow",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/dbaumgarten/FToDTF",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    entry_points={
        'console_scripts': ['fasttext=ftodtf.cli:cli_main'],
    },
    install_requires=find_dependencies()
)
