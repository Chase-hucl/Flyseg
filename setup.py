from setuptools import setup, find_packages

setup(
    name="flyseg",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[],  # 或 ['numpy', 'torch'] 等
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "flyseg-predict = flyseg.prediction:main",
            'flyseg-clean-model = flyseg.nnunet_config:clean_model_cache',# 可选，定义命令行接口
        ]
    }
)
