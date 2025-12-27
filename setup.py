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

INSTALL_REQUIRES = [
   
]

# --- 3. Setup Call ---
setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A package for topic modeling using various algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    
    # --- Package Discovery Improvement ---
    package_dir={"": "src"}, # Tells setuptools that packages are found inside the 'src' directory
    packages=find_packages(where="src"), # Looks for packages inside the 'src' directory (e.g., src/models, src/data)
    
    # --- Dependency Injection ---
    install_requires=INSTALL_REQUIRES,
    
    # --- General Metadata ---
    python_requires=">=3.8", # Updating to 3.8+ is common practice now
)