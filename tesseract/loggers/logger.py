import logging
import os
from typing import Optional

def get_logger(
        name :str, log_level: str = 'INFO',
        log_file : Optional[str]=None, console : bool=True,
        log_format : Optional[str]=None
) -> logging.Logger:

    '''
   Create and configure a logger for the application.

   Args:
       name (str): Name of the logger, typically the module's `__name__`.
       log_level (str, optional): Logging level (e.g., 'DEBUG', 'INFO'). Defaults to 'INFO'.
       log_file (Optional[str], optional): Relative path for log file. If None, file logging is disabled.
       console (bool, optional): Whether to log to console. Defaults to True.
       log_format (Optional[str], optional): Custom log format string. Defaults to standard console/file formats

   Returns:
       logging.Logger: Configured logger instance

   Notes:
       - Log files are stored under `tesseract/logs` when `log_file` is provided.
       - Prevents duplicate handlers if called multiple times for the same logger.

    '''
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.hasHandlers():

     if console:
        console_handler = logging.StreamHandler()

        if log_format:
           console_format = log_format
        else:
            console_format = logging.Formatter("{name}- {levelname} - {message}" , style='{')

            
    
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)


     base_log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
     base_log_dir = os.path.abspath(base_log_dir)

     if log_file:
       
       log_file  = os.path.join(base_log_dir, log_file)
       os.makedirs(os.path.dirname(log_file), exist_ok=True)

       file_handler = logging.FileHandler(log_file , mode='a' , encoding='utf-8')

       if log_format:
          file_format = log_format
       else:
          file_format = logging.Formatter("{asctime} - {levelname} - {name}:{funcName}:L{lineno}:{message}", style="{") 

       file_handler.setFormatter(file_format)
       logger.addHandler(file_handler)

    return logger