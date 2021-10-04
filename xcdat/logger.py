"""Logger module for setting up a logger."""
import logging
import logging.handlers


def setup_custom_logger(name: str, propagate: bool = False) -> logging.Logger:
    """Sets up a custom logger.

    Parameters
    ----------
    name : str
        Name of the file where this function is called.
    propagate : bool, optional
        Whether to propagate logger messages or not, by default False

    Returns
    -------
    logging.Logger
        The logger.

    Examples
    ---------
    Detailed information, typically of interest only when diagnosing problems:

    >>> logger.debug("")

    Confirmation that things are working as expected:

    >>> logger.info("")

    An indication that something unexpected happened, or indicative of some
    problem in the near future:

    >>> logger.warning("")

    The software has not been able to perform some function due to a more
    serious problem:

    >>> logger.error("")

    A serious error, indicating that the program itself may be unable to
    continue running:

    >>> logger.critical("")
    """
    log_format = "%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s"
    log_filemode = "w"  # w: overwrite; a: append

    # Setup
    logging.basicConfig(format=log_format, filemode=log_filemode, level=logging.INFO)
    logger = logging.getLogger(name)
    logger.propagate = propagate

    # Console output
    consoleHandler = logging.StreamHandler()
    logFormatter = logging.Formatter(log_format)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    return logger
