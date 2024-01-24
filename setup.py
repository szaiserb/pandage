from setuptools import setup

setup(name='pandage',
    version='1.0',
    description='A pandas based tool for data acquisition, storage, manipulation and visualization. ',
    author='Sebastian Zaiser',
    license='BSD',
    packages=['pandage'],
    package_data={'pandage': ['qtgui/*.ui', 'qtgui/*.py']},
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'six',
        'matplotlib',
        'PyQt5',
        'more_itertools'
    ],
    zip_safe=False)
