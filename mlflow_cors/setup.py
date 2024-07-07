from setuptools import setup

setup(
    name="mlflow_cors",
    entry_points="""
        [mlflow.app]
        mlflow_cors=mlflow_cors:create_app
    """,
    version='0.0.1'
)