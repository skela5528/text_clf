from configparser import ConfigParser


def get_config(config_path: str = 'config.ini') -> ConfigParser:
    config = ConfigParser()
    config.read(config_path)
    print(config.sections())
    # TODO validate sections
    return config


global CONFIG
CONFIG = get_config()
