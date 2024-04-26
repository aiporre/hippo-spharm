from setuptools import setup, find_packages

setup(
    name='hippospharm',
    version='0.1',
    description='hippospharm: A package for the encoding of 3d objects in embeddings for machine learning tasks.',
    url='http://github.com/aiporre/hippo-spharm',
    author='Ariel Iporre Rivas',
    author_email='rivas@cbs.mpg.de',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        # Add other dependencies here
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    zip_safe=False
)