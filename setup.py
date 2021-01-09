from setuptools import setup, find_packages

setup(
    name="MADpy",
    version="0.1",
    author="Christian Michelsen",
    author_email="christianmichelsen@gmail.com",
    description="Metagenomics Ancient Damage python: MADpy",
    packages=find_packages(),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        MADpy=MADpy.cli:main
    """,
)
