from enum import Enum
import sys
import logging

ENDC = '\033[0m'

black_ansi_color_code = 30


class Color(int, Enum):
    """ ANSI color codes """
    BLACK = 30
    RED = 31
    GREEN = 32
    BROWN = 33
    BLUE = 34
    PURPLE = 35
    CYAN = 36


def ansi_color(color: Color):
    return f"\033[0;{color.value}m"


class __CustomFormatter(logging.Formatter):
    """Logging Formatter for a simple unix style logger """

    def format(self, record):
        name: str = record.name
        level: str = record.levelname
        message: str = record.getMessage()

        level_color = {
            "INFO": Color.CYAN,
            "WARNING": Color.BROWN,
            "ERROR": Color.RED,
        }

        c = level_color.get(level)
        if c is None:
            c = 0
            print(f"Unknown color for level {level}")

        if c != 0 and sys.stdout.isatty():
            return ansi_color(c) + f"[{name}]  {message}{ENDC}"
        else:
            return f"[{name}]  {message}"


def __initialize_logger(name):
    logger = logging.getLogger(name)
    logger.propagate = False

    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(__CustomFormatter())
    logger.addHandler(ch)

    return logger


__loggers = {}


def get_logger(name) -> logging.Logger:
    global __loggers

    if name not in __loggers.keys():
        __loggers[name] = __initialize_logger(name)

    return __loggers[name]

def disable_logging(keep_main=False):
    for logger in __loggers:
        if keep_main and "MAIN" in logger:
            continue
        __loggers[logger].disabled = True
