import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skyrim",
    version="1.0.0",
    author="huxiaoou",
    author_email="516984451@qq.com",
    description="Some highly frequently used tools in daily research.",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/huxiaoou/skyrim",
    install_requires=["numpy", "pandas", "matplotlib"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
