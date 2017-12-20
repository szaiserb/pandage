from setuptools import setup

setup(name='qutip_enhanced',
      version='1.0',
      description='Enhancements to qutip regarding NV spin physics',
      author='Sebastian Zaiser',
      author_email='s.zaiser@physik.uni-stuttgart.de',
      license='BSD',
      packages=['qutip_enhanced'],
      package_data={'qutip_enhanced': ['qtgui/*.ui', 'qtgui/*.py']},
      include_package_data=True,
      zip_safe=False)