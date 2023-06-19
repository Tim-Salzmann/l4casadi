from skbuild import setup
import shutil
import pathlib


def compile_hook(manifest):
    lib = manifest[0]
    shutil.copy(lib, pathlib.Path(__file__).parent.resolve() / 'l4casadi')
    return []


setup(
    cmake_process_manifest_hook=compile_hook,
    cmake_source_dir='l4casadi/cpp',
    include_package_data=True,
    package_data={'': [
        '**.dylib',
        '**.so',
        'cpp/include/l4casadi.hpp',
        'template_generation/c_templates_tera/casadi_function.in.cpp'
    ]},
)
