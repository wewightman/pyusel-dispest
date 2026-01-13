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
    version="0.0.1",
    python_requires='>=3.11,<3.14',
    install_requires=[
        'numpy',
        'scipy',
        "pyusel-interp @ git+https://github.com/wewightman/pyusel-interp/tree/make_cpu_and_gpu_friendly"
    ],
    package_data={
        'dispest':["__xcorr__.cu"],
    },
)