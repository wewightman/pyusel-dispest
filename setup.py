from setuptools import setup

# load requirements list
with open("requirements.txt", 'r') as f:
    reqs = f.readlines()

with open("req_depends.txt", 'r') as f:
    reqdeps = f.readlines()

with open("README.md", 'r') as f:
    readme = f.read()

# run setup tools
setup(
    name="dispest",
    description="Displacement estimators and shear wave speed estimators",
    author="Wren Wightman",
    author_email="wew12@duke.edu",
    install_requires=reqs,
    dependency_links=reqdeps,
    readme=readme,
    version="0.0.0"
)