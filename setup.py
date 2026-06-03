import setuptools

setuptools.setup(
    name="sf",
    version="0.0.1",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
