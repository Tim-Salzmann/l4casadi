import os

try:
    from skbuild import setup
    import shutil
    import pathlib
except ImportError:
    raise ImportError('Please make sure all build requirements for L4CasADi are installed: `pip install -r '
                      'requirements_build.txt`')

try:
    import torch
except ImportError:
    raise ImportError('PyTorch is required to build and install L4CasADi. Please install PyTorch and install '
                      'l4casadi with `pip install l4casadi --no-build-isolation` or from source via `pip install . '
                      '--no-build-isolation`')


def compile_hook(manifest):
    lib = manifest[0]
    file_path = pathlib.Path(__file__).parent.resolve()
    (file_path / 'l4casadi' / 'lib').mkdir(exist_ok=True)
    (file_path / 'l4casadi' / 'include').mkdir(exist_ok=True)

    # Copy lib
    shutil.copy(lib, file_path / 'l4casadi' / 'lib')

    # Copy Header
    shutil.copy(file_path / 'libl4casadi' / 'include' / 'l4casadi.hpp', file_path / 'l4casadi' / 'include')
    return []


setup(
    cmake_process_manifest_hook=compile_hook,
    cmake_source_dir='libl4casadi',
    cmake_args=['-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_TORCH_PATH={os.path.dirname(os.path.abspath(torch.__file__))}'],
    include_package_data=True,
    package_data={'': [
        'lib/**.dylib',
        'lib/**.so',
        'lib/**.dll',
        'include/**.hpp',
        'template_generation/templates/casadi_function.in.cpp'
    ]},
)
