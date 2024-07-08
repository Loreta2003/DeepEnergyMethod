import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class MLP(layers.Layer):
    ''' A feed-forward neural network '''
    def __init__(self, units, activation):
        super().__init__()
        self.ls = []
        for (u, a) in zip(units, activation):
            self.ls += [layers.Dense(u, activation=a)]

    def call(self, x):    
        for l in self.ls:
            x = l(x)
        return x

def gradient(f, x):
    ''' Computes the first derivative of a function f '''
    with tf.GradientTape() as g:
        g.watch(x)
        u = f(x)
    du = tf.squeeze(g.batch_jacobian(u, x), 2)
    return u, du

def second_gradient(f, x):
    ''' Computes the second derivative of a function f '''
    with tf.GradientTape() as g:
        g.watch(x)
        u, du = gradient(f, x)
    ddu = tf.squeeze(g.batch_jacobian(du, x), 2)
    return u, du, ddu

class PINN(layers.Layer):
    ''' Physics-informed neural network for the non-dimensional, linear Euler-Bernoulli beam '''
    def __init__(self, w_pel, w_dir, param, **kwargs):
        super().__init__()
        # Create a FFNN
        self.mlp = MLP(**kwargs)

        # Weights
        self.w_pel = tf.constant(w_pel, dtype=tf.float32)
        self.w_dir = tf.constant(w_dir, dtype=tf.float32)

        # Beam parameters
        self.EA = tf.constant(param[0] * param[1], dtype=tf.float32)
        self.E = tf.constant(param[0], dtype = tf.float32)
        self.L = tf.constant(param[2], dtype=tf.float32)

        # Value of the inhomogeneous Neumann BC (dimensional)
        self.F = tf.constant(5., dtype=tf.float32)

        # x-values of the Dirichlet BCs
        self.zero = tf.zeros([1,1])
        self.L_pos = tf.ones([1,1]) * param[2]

    def call(self, x):
        ''' Main PINN function '''
        # evaluate MLP and outputs
        u, du, ddu = second_gradient(self.mlp, x)

        u0, du0 = gradient(self.mlp, self.zero)
        uL, duL = gradient(self.mlp, self.L_pos)

        # Generate nodes and weights for Gauss-Legendre quadrature
        nodes, weights = np.polynomial.legendre.leggauss(1000)
        a = 0
        b = self.L.numpy()
        transformed_nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        transformed_nodes = tf.convert_to_tensor(transformed_nodes, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        transformed_nodes = tf.reshape(transformed_nodes, (1000, 1))

        # Evaluate the function at the transformed nodes using self.mlp
        _, func_at_nodes = gradient(self.mlp, transformed_nodes)
        func_at_nodes = tf.pow(func_at_nodes, 2)

        # Compute the internal energy
        inter_en = 0.5 * self.EA * 0.5 * (b - a) * tf.reduce_sum(weights * func_at_nodes)

        # Compute the external energy
        exter_en = self.F * uL

        # Calculate the resulting potential energy
        res_pel = inter_en - exter_en

        # Dirichlet boundary condition residuals
        res_dir_bc = tf.square(u0)

        # assemble and add loss term
        total_loss = self.w_pel * res_pel + self.w_dir * res_dir_bc
        self.add_loss(total_loss)

        return u, du

def build(*args, **kwargs):
    ''' Builds an evaluation model that returns the nodal displacements '''
    # define input shape
    x = tf.keras.Input(shape=(1,))
    # define which (custom) layers the model uses
    pinn = PINN(*args, **kwargs)
    u, du = pinn(x)
    # connect input and output
    model = tf.keras.Model(inputs = [x], outputs = [u, du])
    # define optimizer
    model.compile('adam')
    return model

def convert(x):
    return tf.expand_dims(tf.constant(x, dtype='float32'), axis=1)

def analytical_ebb(EA, F, L, x):
    ''' Data generation with the analytical solution of 
     the linear Euler-Bernoulli beam '''
    u = F * x / EA
    du = np.full_like(x, F / EA)
    x = convert(x)
    u = convert(u)
    du = convert(du)
    return x, u, du

# Beam properties
L = 1            # Length [m]
W = 0.02           # Width [m]
H = 0.02           # Height [m]
E = 50e2           # Young's modulus [Pa]

# Cross-section parameters (assuming rectangular cross-section)
A = W * H
I = H ** 3 * W / 12
param = [E, A, L]

# Compute the analytical solution
n = 1000 # number of collocation points
F = 5. # Transversal force at free end
x = np.linspace(0, L, n)

x, u, du = analytical_ebb(E * A, F, L, x)

# Plot data
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(x, u, 'b', linestyle='--')
axs[0].set_xlabel('x')
axs[0].set_ylabel('u')
axs[0].grid()

axs[1].plot(x, du, 'b', linestyle='--')
axs[1].set_xlabel('x')
axs[1].set_ylabel('du')
axs[1].grid()

plt.tight_layout()
plt.show()

# Loss weights
WPEL = 1.0
WDIR = 1000.0

# Load model
model = build(WPEL, WDIR, param,
    units=[8,  1],
    activation=['softplus', 'linear']
)
model.summary()

# Create collocation points
NCAL = 1000 # number of collocation points
x_cal = tf.expand_dims(tf.convert_to_tensor(np.linspace(0, L, NCAL), dtype=tf.float32), 1)

# Calibrate model
NEPOCHS = 2000
model.optimizer.learning_rate.assign(0.01)
h = model.fit([x_cal], [x_cal, x_cal],
              epochs=NEPOCHS,
              verbose=2,
              batch_size=32)

plt.semilogy(np.linspace(1, NEPOCHS, NEPOCHS), h.history['loss'], label='training loss')
plt.grid(which='both')
plt.legend()
plt.show()

u_pinn, du_pinn = model(x)

# Plot data and prediction
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(x, u_pinn, 'r', label='PINN')
axs[0].plot(x, u, 'b', linestyle='--', label='analytical')
axs[0].set_xlabel('x')
axs[0].set_ylabel('u')
axs[0].grid()
axs[0].legend()

axs[1].plot(x, du_pinn, 'r', label='PINN')
axs[1].plot(x, du, 'b', linestyle='--', label='analytical')
axs[1].set_xlabel('x')
axs[1].set_ylabel('du')
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.show()


