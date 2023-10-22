# Use L4CasADi in Matlab

## Prerequisites
- Download CasADi for Matlab from https://web.casadi.org/get/ and place it in this folder.

## Function Export
- Run `python export.py` to export the torch model as L4CasADi library.
- This will also output the `LIB_DIR` to export as environment variable.

## Set Environment Variables in Matlab
Make sure that `LD_LIBRARY_PATH` (`DYLD_LIBRARY_PATH` on MacOS) is set to `LIB_DIR` in Matlab on start.

```
getenv("LD_LIBRARY_PATH")
```

Setting the env with Matlab's `setenv` will most likely not work as it only sets the variable locally.
Instead, you will have to set the env variable in the shell before starting Matlab from the same shell.

Example for MacOS:
```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<LIB_DIR>
DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}" /Applications/MATLAB_R2023b.app/Contents/MacOS/MATLAB
```

## Run
Run the `use_l4casadi_f.m` script in Matlab.