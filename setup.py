import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-helper", # Replace with your own username
    version="0.1",
    author="Santanu Bhattacharjee",
    author_email="mail.santanu94@gmail.com",
    description="A pytorch utility library to provide quick use functions while allowing low level customization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/santanu94/torch-helper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
