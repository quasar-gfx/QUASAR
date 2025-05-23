import logging

class ColorFormatter(logging.Formatter):
    # ANSI color codes
    COLORS = {
        logging.INFO: '\033[32m',     # Green
        logging.WARNING: '\033[33m',  # Yellow (used for orange)
        logging.ERROR: '\033[31m',    # Red
        logging.CRITICAL: '\033[31m', # Red (same as error)
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelno, '')
        level_char = record.levelname[0]
        colored_level = f"{color}{level_char}{self.RESET}"
        record.levelname_colored = colored_level
        fmt = '[%(asctime)s] [%(levelname_colored)s] %(message)s'
        formatter = logging.Formatter(fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

formatter = ColorFormatter('[%(asctime)s] [%(levelname).1s] %(message)s', datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(handler)
