from setuptools import setup

with open("README.md", 'r') as f:
    readme = f.read()

# run setup tools
setup(
    name="pyusel-dispest",
    description="Displacement estimators used in ultrasound elasticity",
    author="Wren Wightman",
    packages=['dispest'],
    author_email="wew12@duke.edu",
    long_description=readme,
    version="0.0.1"
)