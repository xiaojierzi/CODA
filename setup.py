from setuptools import setup, find_packages

setup(
    name="icoda",
    version="2.0.0",
    description='Integrative cross-sample alignment and spatially differential gene analysis for spatial transcriptomics',
    author='Yecheng Tan',
    author_email='yctan21@m.fudan.edu.cn',
    url='https://github.com/xiaojierzi/CODA',
    packages=find_packages(), 
    install_requires=[],
    python_requires='>=3.9',
    license='GNU General Public License v3.0',
)
