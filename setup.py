from setuptools import Extension, setup

# load the C extentsion library
xcorr_cpu = Extension(
    name="dispest._xcorr_cpu",
    include_dirs=["dispest"],
    depends=["dispest/__xcorr__.h"],
    sources=["dispest/__xcorr__.c"]
)

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
    ext_modules=[xcorr_cpu]
)