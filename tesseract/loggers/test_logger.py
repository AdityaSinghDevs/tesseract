from logger import get_logger

logger = get_logger(__name__, log_file= "test.log" )

logger.critical("This test was succcess")