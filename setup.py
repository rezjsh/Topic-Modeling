from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A package for topic modeling using various algorithms."

__version__ = "0.0.1" 

REPO_NAME = "Topic-Modeling"
AUTHOR_USER_NAME = "rezjsh"
SRC_REPO = "topic_modeling" 
AUTHOR_EMAIL = "your.email@example.com" 

# Core dependencies required for the basic framework to function
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "spacy>=3.5.0",
    "gensim>=4.3.0",
    "scikit-learn>=1.2.0",
    "pandas>=1.5.0,<2.0.0", # Essential for pyLDAvis compatibility
    "numpy>=1.23.0",
    "tqdm>=4.65.0",
    "plotly>=5.14.0",
    "streamlit>=1.22.0",
    "pyLDAvis>=3.4.0",
]

# Optional dependencies for heavy embedding models
EXTRAS_REQUIRE = {
    "full": [
        "bertopic>=0.14.0",
        "top2vec>=1.0.34",
    ]
}

setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A modular framework for benchmarking 8 different Topic Modeling architectures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    
    # Package Discovery
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Metadata & Entry Points
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # Allows users to run the dashboard directly via command line
    entry_points={
        "console_scripts": [
            "topic-viz=topic_modeling.app:main",
        ],
    },
)