import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="luna_face_recog",
    version="1.0.0",
    author="Luna Face Recognition Team",
    author_email="",
    description="A Modern PyTorch-based Face Recognition Framework with GPU acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LSDJesus/Luna_face_recog",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    license="MIT",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "requests>=2.25.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
    ],
)