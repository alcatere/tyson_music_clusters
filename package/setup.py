from setuptools import setup

setup(
    name='ml_package',
    version='0.1',
    description='Non supervised learning for Tyler the producer dataset',
    author='Rafael Arana',
    author_email='alberto2809.rb@gmail.com',
    packages= ['package.ml_training', 'package.utils', 'package.features'], 
    install_requires = ['pandas', 'numpy', 'scikit-learn', 'mlflow', 'requests', 'matplotlib', 'seaborn']
)