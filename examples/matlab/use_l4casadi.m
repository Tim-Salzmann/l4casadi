addpath('./casadi-3')

disp(getenv("DYLD_LIBRARY_PATH"))

import casadi.*

f = external('sin_l4c', fullfile('..', '_l4c_generated', 'libsin_l4c.dylib'));

x = DM(pi/2);

disp(f(x))