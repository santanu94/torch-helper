import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-helper",
    version="0.3-alpha",
    author="Santanu Bhattacharjee",
    author_email="mail.santanu94@gmail.com",
    description="A pytorch utility to provide customizable quick use functions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/santanu94/torch-helper.git",
    keywords = ['PyTorch', 'Deep Learning'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch>=1.0.0",
        "torchvision>=0.2.1",
        "sklearn"
    ]
)
