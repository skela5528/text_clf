from configparser import ConfigParser


def get_config(config_path: str = 'config.ini') -> ConfigParser:
    config = ConfigParser()
    config.read(config_path)
    print(config.sections())
    # TODO validate sections
    return config


def parse_config_section(config_section) -> dict:
    params_dict = {}
    for param, value in config_section.items():
        if value.isnumeric():
            value = int(value)
        params_dict[param] = value
    return params_dict

global CONFIG
CONFIG = get_config()
