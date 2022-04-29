from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='facilyst',
    version='0.0.2',
    author='Parthiv Naresh',
    author_email='pnaresh.github@gmail.com',
    description='Make data analysis and machine learning tools more easily accessible.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ParthivNaresh/facilyst/',
    python_requires='>=3.7, <4',
    install_requires=open('requirements.txt').readlines(),
    tests_require=open('test-requirements.txt').readlines(),
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'facilyst = facilyst.__main__:cli'
        ]
    },
)