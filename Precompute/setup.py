from distutils.core import setup,Extension
from Cython.Build import cythonize
import eigency

setup(
    author='nyLiao',
    version='0.0.1',
    install_requires=['Cython>=0.2.15','eigency>=1.77'],
    packages=['little-try'],
    python_requires='>=3',
    ext_modules=cythonize(Extension(
        name='propagation',
        sources=['propagation.pyx'],
        language='c++',
        extra_compile_args=["-std=c++11", "-O3", "-fopenmp"],
        include_dirs=[".", "module-dir-name"] + eigency.get_includes()[:2] + ["/usr/local/include/eigen3"] + ['/usr/local/include/Spectra'],
    ))
)