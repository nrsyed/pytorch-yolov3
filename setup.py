from setuptools import setup

setup(
    name="pytorch-yolov3",
    version="2.0",
    url="https://github.com/nrsyed/pytorch-yolov3",
    author="Najam R Syed",
    author_email="najam.r.syed@gmail.com",
    license="MIT",
    packages=["yolov3", "yolov3.devtools"],
    install_requires=[
        "numpy",
        "opencv-python",
        "torch"
    ],
    entry_points={
        "console_scripts": ["yolov3 = yolov3.__main__:main"]
    },
)
