from skbuild import setup

setup(
    #cmake_args=['-DCMAKE_PREFIX_PATH=/Users/TimSalzmann/Documents/Study/PhD/Code/Ideas/L4CasADi/libtorch'],
    cmake_source_dir='l4casadi/cpp',
    #packages=["l4casadi", "l4casadi.template_generation"],
    include_package_data=True,
    package_data={'': [
        'cpp/include/l4casadi.hpp',
        'template_generation/c_templates_tera/casadi_function.in.cpp'
    ]},
)
