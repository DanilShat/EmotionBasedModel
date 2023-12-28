from setuptools import setup, find_packages

setup(
    name='EmotionBasedModel',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/DanilShat/EmotionBasedModel',
    author='Daniil Shatilov',
    author_email='purrrple24@gmail.com',
    description='A package for emotion-based classification, regression, and clusterization.',
    install_requires=[
        'numpy',
        'scikit-learn',
        'transformers',
        'tqdm',
        'matplotlib'
    ],
)
