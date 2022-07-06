import logging
import functools
import inspect

# lev = logging.DEBUG
lev = logging.INFO
logging.basicConfig(level=lev)
logger = logging.getLogger()

def log(f):
    @functools.wraps(f)
    def wr(*arg, **kwargs):
        try:
            out = f(*arg, **kwargs)
            return out
        except Exception as e:
            logger.exception(f'{f.__name__}: {repr(e)}')
    return wr

# @log
# def hi(ins: str = 't') -> str:
#     return ins + 's'
# hi()
# hi(8)


def vtypes(f):
    @functools.wraps(f)
    def wr(*args, **kwargs):

        print('checking kwargs')
        d={**kwargs}
        print(d)
        print('printing done')
        return f(*args, **kwargs)
    return wr


@typechecked
@vtypes
def test(p: dict , s: int = 3, **kwargs) -> str:
    '''asd'''
    return ('t')

from typeguard import typechecked

test(s=3, p={'a':1})