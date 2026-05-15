import logging
import platform
import psutil
import sys, os
import functools


logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, filename='msmate.log', encoding='utf-8', format='%(asctime)s %(levelname)s %(name)s %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.debug(f'{[os.cpu_count(), platform.uname(), psutil.net_if_addrs(), psutil.users()]}')

def log(f):
    @functools.wraps(f)
    def wr(*arg, **kwargs):
        try:
            logger.info(f'calling {f.__name__}')
            out = f(*arg, **kwargs)
            logger.info(f'done {f.__name__}')
            return out
        except Exception as e:
            logger.exception(f'{f.__name__}: {repr(e)}')
    return wr

import inspect
def logIA(f):
    @functools.wraps(f)
    def wr(*args, **kwargs):
        try:
            #print(f.__dict__)
            func_args = inspect.signature(f).bind(*args, **kwargs).arguments
            func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
            logger.info(f"{f.__qualname__} ({func_args_str})")
            out = f(*args, **kwargs)
            #logger.info(f'done {f.__name__}')
            return out

        except Exception as e:
            logger.exception(f'{repr(e)}')
    return wr