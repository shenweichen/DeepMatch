import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'requests',
    "deepctr~=0.9.4"
]

setuptools.setup(
    name="deepmatch",
    version="0.3.2",
    author="Weichen Shen",
    author_email="weichenswc@163.com",
    description="Deep matching model library for recommendations, advertising. It's easy to train models and to **export representation vectors** for user and item which can be used for **ANN search**.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shenweichen/deepmatch",
    download_url='https://github.com/shenweichen/deepmatch/tags',
    packages=setuptools.find_packages(
        exclude=["tests", "tests.models", "tests.layers"]),
    python_requires=">=3.7",
    install_requires=REQUIRED_PACKAGES,
    extras_require={
    },
    entry_points={
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license="Apache-2.0",
    keywords=['match', 'matching', 'recommendation'
                                   'deep learning', 'tensorflow', 'tensor', 'keras'],
)
