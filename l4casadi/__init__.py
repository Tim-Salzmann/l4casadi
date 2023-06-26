from importlib.resources import files
import ctypes
import platform

from .l4casadi import L4CasADi


file_dir = files('l4casadi')
lib_path = file_dir / 'lib' / ('libl4casadi' + ('.dylib' if platform.system() == 'Darwin' else '.so'))
ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
