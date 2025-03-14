from setuptools import setup,find_packages


setup(
    name='franka_real',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        # 'gymnasium==1.1.1',
    ],
    description='Real-Time Franka environments',
    author='Alireza Azimi',
    author_email='sazimi@ualberta.ca',
)