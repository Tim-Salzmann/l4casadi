import casadi as cs
import l4casadi as l4c

# Declare variables
x = cs.MX.sym("x", 2)

# Form the NLP
f = lambda x: x[0] ** 2 + x[1] ** 2  # objective
f = l4c.L4CasADi(f, name='f', model_expects_batch_dim=False)(x)

g = lambda x: x[0] + x[1] - 10  # constraint
g = l4c.L4CasADi(g, name='g', model_expects_batch_dim=False)(x)

nlp = {'x': x, 'f': f, 'g': g}


# Pick an NLP solver
MySolver = "ipopt"
# MySolver = "worhp"
# MySolver = "sqpmethod"

# Solver options
opts = {}
if MySolver == "sqpmethod":
    opts["qpsol"] = "qpoases"
    opts["qpsol_options"] = {"printLevel": "none"}  # type: ignore[assignment]

# Allocate a solver
solver = cs.nlpsol("solver", MySolver, nlp, opts)

# Solve the NLP
sol = solver(lbg=0)

# Print solution
print("-----")
print("objective at solution = ", sol["f"])
print("primal solution = ", sol["x"])
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])
