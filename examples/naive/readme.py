import casadi as cs
import l4casadi as l4c


naive_mlp = l4c.naive.MultiLayerPerceptron(2, 128, 1, 2, 'Tanh')
l4c_model = l4c.L4CasADi(naive_mlp)

x_sym = cs.MX.sym('x', 3, 2)
y_sym = l4c_model(x_sym)
f = cs.Function('y', [x_sym], [y_sym])
df = cs.Function('dy', [x_sym], [cs.jacobian(y_sym, x_sym)])
ddf = cs.Function('ddy', [x_sym], [cs.jacobian(cs.jacobian(y_sym, x_sym), x_sym)])

x = cs.DM([[0., 2.], [0., 2.], [0., 2.]])
print(l4c_model(x))
print(f(x))
print(df(x))
print(ddf(x))
