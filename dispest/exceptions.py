import logging as __logging
logger = __logging.getLogger(__name__)

class DispEstException(Exception):
    pass

class XCorrException(DispEstException):
    pass