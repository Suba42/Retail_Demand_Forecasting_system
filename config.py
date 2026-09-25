import os


class Config:

    # Flask
    SECRET_KEY = os.environ.get(
        "SECRET_KEY",
        "retail-demand-forecasting-secret-key"
    )

    # MySQL
    MYSQL_HOST = os.environ.get(
        "MYSQL_HOST",
        "localhost"
    )

    MYSQL_PORT = int(
        os.environ.get(
            "MYSQL_PORT",
            3306
        )
    )

    MYSQL_USER = os.environ.get(
        "MYSQL_USER",
        "root"
    )

    MYSQL_PASSWORD = os.environ.get(
        "MYSQL_PASSWORD",
        "subasri14585"
    )

    MYSQL_DATABASE = os.environ.get(
        "MYSQL_DATABASE",
        "retail_demand_forecasting"
    )

    # Flask-SQLAlchemy
    SQLALCHEMY_DATABASE_URI = (
        "mysql+pymysql://"
        + MYSQL_USER
        + ":"
        + MYSQL_PASSWORD
        + "@"
        + MYSQL_HOST
        + ":"
        + str(MYSQL_PORT)
        + "/"
        + MYSQL_DATABASE
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False