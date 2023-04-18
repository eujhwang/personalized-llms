from setuptools import setup, find_packages

setup(name='callgpt',
      version='0.1',
      description="Library for running few-shot inference on GPT models",
      install_requires=[
          'openai'
      ],
      license='Apache License 2.0',
      long_description=open('README.md').read(),
)
