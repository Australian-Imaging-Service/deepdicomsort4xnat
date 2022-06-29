from setuptools import setup, find_packages


setup(
    name='dds4xnat',
    version='0.1',
    author='Mahdieh Dashtbani-Moghari',
    author_email='dashtbani.2009@gmail.com',
    packages=find_packages(),
    url='https://github.com/australian-imaging-service/deepdicomsort4xnat',
    license='CC0',
    description=('Renames the scan type for MR sessions stored on XNAT '
                 'using the DeepDicomSort algorithm'),
    long_description=open('README.md').read(),
    install_requires=[
        'tensorflow>=2.9.1',
        'xnat>=0.3.17',
        'pydra>=0.18',
        'pydicom>=1.0.2',
        'SimpleITK>=2.1.1.2'],
    extras_require={
        'test': [
            'pytest>=5.4.3',
            'xnat4tests>=0.1.4']},
    include_package_data=True,
    classifiers=(
        ["Development Status :: 4 - Beta",
         "Intended Audience :: Healthcare Industry",
         "Intended Audience :: Science/Research",
         "License :: OSI Approved :: Apache Software License",
         "Natural Language :: English",
         "Topic :: Scientific/Engineering :: Bio-Informatics",
         "Topic :: Scientific/Engineering :: Medical Science Apps."]
        + ["Programming Language :: Python :: " + str(v)
           for v in ('3.8', '3.9')]),
    keywords='repository neuroimaging workflows xnat')
