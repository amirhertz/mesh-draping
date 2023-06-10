import os
import sys
IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None
EPSILON = 1e-4
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = f'{PROJECT_ROOT}/assets/'
RAW_ROOT = f'{DATA_ROOT}raw_meshes/'
CHECKPOINTS_ROOT = f'{DATA_ROOT}/checkpoints'
CACHE_ROOT = f'{DATA_ROOT}/cache/'
POINTS_CACHE = f'{CACHE_ROOT}/points'
GLOBAL_SCALE = 10
