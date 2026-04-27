from setuptools import find_packages, setup

setup(
    name='solar_flare_classification',
    version='0.1.0',
    author='Himel Das',
    author_email='himeldas077@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'imbalanced-learn',
        'streamlit',
        'plotly',
        'shap',
        'seaborn',
        'matplotlib'
    ]
)
