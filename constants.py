import logging
import os


def get_logger(log_path: str = None, level=logging.DEBUG) -> logging.Logger:
    global LOGGER
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handlers = []
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(log_format)
        handlers.append(file_handler)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(log_format)
    handlers.append(stdout_handler)
    logging.basicConfig(level=level, handlers=handlers, datefmt='%Y-%m-%d %H:%M:%S')
    LOGGER = logging.getLogger()
    return LOGGER


LOGGER = get_logger()

TOPICS = ['NO_TOPIC', 'earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'wheat', 'ship', 'corn',
          'oilseed', 'sugar', 'coffee', 'gold', 'veg-oil', 'money-supply', 'dlr', 'gnp', 'livestock', 'soybean',
          'nat-gas', 'bop', 'cpi', 'carcass', 'copper', 'reserves', 'cocoa', 'cotton', 'jobs', 'iron-steel', 'rice',
          'barley', 'alum', 'rubber', 'ipi', 'meal-feed', 'gas', 'palm-oil', 'yen', 'silver', 'zinc', 'pet-chem',
          'sorghum', 'rapeseed', 'strategic-metal', 'tin', 'wpi', 'orange', 'lead', 'retail', 'hog', 'heat', 'soy-oil',
          'housing', 'soy-meal', 'fuel', 'dmk', 'lei', 'lumber', 'nickel', 'stg', 'oat', 'tea', 'sunseed', 'sun-oil',
          'platinum', 'rape-oil', 'l-cattle', 'groundnut', 'plywood', 'jet', 'income', 'coconut', 'tapioca', 'propane',
          'potato', 'instal-debt', 'coconut-oil', 'inventories', 'linseed', 'copra-cake', 'palmkernel',
          'cornglutenfeed', 'wool', 'saudriyal', 'fishmeal', 'palladium', 'cpu', 'austdlr', 'naphtha', 'pork-belly',
          'lin-oil', 'rye', 'red-bean', 'groundnut-oil', 'citruspulp', 'rape-meal', 'can', 'dfl', 'corn-oil', 'peseta',
          'cotton-oil', 'nzdlr', 'rand', 'ringgit', 'castorseed', 'castor-oil', 'lit', 'rupiah', 'skr', 'nkr', 'dkr',
          'sun-meal', 'lin-meal']
