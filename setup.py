from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()
    install_requires += [
        "melloddy_tuner @ git+git://github.com/melloddy/MELLODDY-TUNER#egg=melloddy_tuner"
    ]


setup(
    name="zairachem",
    version="0.0.1",
    author="Miquel Duran-Frigola",
    author_email="miquel@ersilia.io",
    url="https://github.com/ersilia-os/zaira-chem",
    description="Automated QSAR modeling for chemistry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.7",
    install_requires=install_requires,
    packages=find_packages(exclude=("utilities")),
    entry_points={"console_scripts": ["zairachem=zairachem.cli:cli"]},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="qsar machine-learning chemistry computer-aided-drug-design",
    project_urls={"Source Code": "https://github.com/ersilia-os/zaira-chem"},
    include_package_data=True,
)
