from skbuild import setup
import shutil
import pathlib


def compile_hook(manifest):
    lib = manifest[0]
    file_path = pathlib.Path(__file__).parent.resolve()
    (pathlib.Path(__file__).parent.resolve() / 'l4casadi' / 'lib').mkdir(exist_ok=True)
    (pathlib.Path(__file__).parent.resolve() / 'l4casadi' / 'include').mkdir(exist_ok=True)

    # Copy lib
    shutil.copy(lib, file_path / 'l4casadi' / 'lib')

    # Copy Header
    shutil.copy(file_path / 'libl4casadi' / 'include' / 'l4casadi.hpp', file_path / 'l4casadi' / 'include')
    return []


setup(
    cmake_process_manifest_hook=compile_hook,
    cmake_source_dir='libl4casadi',
    include_package_data=True,
    package_data={'': [
        'lib/**.dylib',
        'lib/**.so',
        'include/**.hpp',
        'template_generation/c_templates_tera/casadi_function.in.cpp'
    ]},
)
