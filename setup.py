from setuptools import setup, find_packages

setup(
    name='slampy',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        #'jax',
        #'jaxlib'
        'opencv-contrib-python',
        'open3d'
    ], # slampy/visual_model/data/ */*.yml
    include_package_data=True,
    package_data={'slampy': ['visual_model/data/**/*.yml']}
)

"""
        'numpy==1.26.3',
        'jax==0.4.23',
        'jaxlib==0.4.23+cuda12.cudnn89'
        'opencv-contrib-python==4.9.0.80',
        'open3d==0.18.0'
"""