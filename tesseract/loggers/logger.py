import logging
import os
from typing import Optional

def get_logger(
        name :str, log_level: str = 'INFO',
        log_file : Optional[str]=None, console : bool=True,
        log_format : Optional[str]=None
) -> logging.Logger:
    
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