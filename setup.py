from setuptools import setup


ver_file = {}

with open("skyllh/_version.py") as f:

    exec(f.read(), ver_file)


setup(
    version=ver_file["_version"],
#    include_package_data=True,
#    package_data={"": extra_files},
)
