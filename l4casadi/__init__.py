try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files   # type: ignore[no-redef]
import ctypes

from .l4casadi import L4CasADi, dynamic_lib_file_ending

from . import naive
from . import realtime


file_dir = files('l4casadi')
lib_path = file_dir / 'lib' / ('libl4casadi' + dynamic_lib_file_ending())
ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
