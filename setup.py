from setuptools import setup, find_packages

setup(
    name="nsclc_survival",
    version="1.0.0",
    description="NSCLC radiomics survival prediction — reproducible deep learning pipeline",
    author="Biruk Kirubel",
    author_email="birukkirubel@gmail.com",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "scripts*", "docs*"]),
    install_requires=[
        "torch==2.2.2",
        "torchvision==0.17.2",
        "nibabel==5.2.1",
        "scikit-image==0.22.0",
        "lifelines==0.29.0",
        "scikit-survival==0.22.2",
        "numpy==1.26.4",
        "scipy==1.13.0",
        "pandas==2.2.1",
        "scikit-learn==1.4.2",
        "huggingface_hub==0.22.2",
        "transformers==4.40.0",
        "matplotlib==3.8.4",
        "tqdm==4.66.2",
        "pyyaml==6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.1.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
