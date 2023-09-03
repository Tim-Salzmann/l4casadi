# Integrate L4CasADi into Pure c(++) Project

## Code Generation
- Run `python generate.py` to generate the c++ code.

## Compile
- Replace `<L4CASADI_LIB_DIR>` in `CMakeLists.txt` with the printed path.
- Create a build directory `mkdir build && cd build`.
- Run `cmake ..` to generate the makefile.
- Run `make` to compile the executable.

## Run
- Set `LD_LIBRARY_PATH` such that Torch and the generated library can be found.\
  `export LD_LIBRARY_PATH=<TORCH_LIB_DIR>:<L4CASADI_GEN_LIB_DIR>:$LD_LIBRARY_PATH`.\
  (Replace `<TORCH_LIB_DIR>` and `<L4CASADI_GEN_LIB_DIR>` with the printed paths.)
- Run `./l4cpp`
- Type in the input to the function and press enter.
- Output is printed to the console.