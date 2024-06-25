from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from scipy.integrate import quadrature


class MLP(layers.Layer):
    ''' A feed-forward neural network '''
    def __init__(self, units, activation):
        super().__init__()
        self.ls = []
        for (u, a) in zip(units, activation):
            self.ls += [layers.Dense(u, a)]

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

def third_gradient(f, x):
    ''' Computes the third derivative of a function f '''
    with tf.GradientTape() as g:
        g.watch(x)
        u, du, ddu = second_gradient(f, x)
    dddu = tf.squeeze(g.batch_jacobian(ddu, x), 2)
    return u, du, ddu, dddu

def fourth_gradient(f, x):
    ''' Computes the fourth derivative of a function f '''
    with tf.GradientTape() as g:
        g.watch(x)
        u, du, ddu, dddu = third_gradient(f, x)
    ddddu = tf.squeeze(g.batch_jacobian(dddu, x), 2)
    return u, du, ddu, dddu, ddddu

class PINN(layers.Layer):
    ''' Physics-informed neural network for the non-dimensional, linear Euler-Bernoulli beam '''
    def __init__(self, w_pel, w_dir, w_neu, w_data, param, labeled_data=None, **kwargs):
        super().__init__()
        # Create a FFNN
        self.mlp = MLP(**kwargs)

        # Weights
        self.w_pel = tf.constant(w_pel, dtype=tf.float32)
        self.w_dir = tf.constant(w_dir, dtype=tf.float32)
        self.w_neu = tf.constant(w_neu, dtype=tf.float32)
        self.w_data = tf.constant(w_data, dtype=tf.float32)

        # Beam parameters
        self.EA = tf.constant(param[0] * param[1], dtype=tf.float32)
        self.EI = tf.constant(param[0] * param[2], dtype=tf.float32)

        # Nondimensional parameters
        self.x_c = tf.constant(param[3], dtype=tf.float32)
        self.w_c = self.x_c / self.EI

        # x-values of the Dirichlet BCs
        self.zero = tf.zeros([1,1])
        self.L = tf.ones([1,1]) * param[3]

        # Value of the inhomogeneous Neumann BC (dimensional)
        self.F = tf.constant(5., dtype=tf.float32)

        # Labeled data of the form [x, w]
        self.labeled_data_x = tf.constant(labeled_data[0], dtype=tf.float32) / self.x_c
        self.labeled_data_w = tf.constant(labeled_data[1], dtype=tf.float32) * self.w_c ** (-1)

    def call(self, x):
        ''' Main PINN function '''
        # nondimensionalize input
        x = x / self.x_c

        # evaluate MLP and outputs
        w, dw, ddw, dddw, ddddw = fourth_gradient(self.mlp, x)
        M = self._moment(ddw)
        Q = self._transversal_force(dddw)

        w0, dw0 = gradient(self.mlp, self.zero)
        wL, _, ddwL, dddwL = third_gradient(self.mlp, self.L / self.x_c)

        res_pde = self._labeled_data_residual()
        #pel residual
        #-----------------------GAUSS QUADRATURE TRY---------------------------------#
        # Generate nodes and weights for Gauss-Legendre quadrature
        nodes, weights = np.polynomial.legendre.leggauss(1000)

        # Transform nodes to the desired interval [a, b]
        a = 0
        b = 1
        transformed_nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)

        # Convert to TensorFlow tensors
        transformed_nodes = tf.convert_to_tensor(transformed_nodes, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        # Reshape transformed_nodes to have shape (200, 1)
        transformed_nodes = tf.reshape(transformed_nodes, (1000, 1))

        # Evaluate the function at the transformed nodes using self.mlp
        _, _, func_at_nodes, _, _ = fourth_gradient(self.mlp, transformed_nodes)
        print("Func_at_nodes:")
        print(func_at_nodes*pow(self.w_c, 2)/pow(self.x_c, 4))
        func_at_nodes = tf.pow(func_at_nodes, 2)

        # Compute the internal energy
        inter_en = 0.5 * self.EI * (tf.pow(self.w_c, 2) / tf.pow(self.x_c, 3))* 0.5 * (b - a) * tf.reduce_sum(weights * func_at_nodes)
        print(inter_en)

        # Compute the external energy
        exter_en = self._external_energy(wL)

        # Calculate the resulting potential energy
        res_pel = inter_en - exter_en
        print("internal_energy:")
        print(inter_en)
        print("external_energy:")
        print(exter_en)
        res_pel = inter_en - exter_en
        print("res_pel:")
        print(res_pel)
        #----------------------------------------------------------------------------#
        #---------------------------NORMAL INTEGRATION-------------------------------#
        # w_x, _, func_at_nodes, _, _ = fourth_gradient(self.mlp, x)
        # func_at_nodes = tf.pow(func_at_nodes, 2)
        # h = x[1:] - x[:-1]

        # inter_en = 0.5 * self.EI * (tf.pow(self.w_c, 2) / tf.pow(self.x_c, 3))*tf.reduce_sum(0.5*h*(func_at_nodes[:-1] + func_at_nodes[1:]))
        # #print(inter_en)
        # exter_en = self._external_energy(w_x[-1])
        # print("internal_energy:")
        # print(inter_en)
        # print("external_energy:")
        # print(exter_en)
        # res_pel = inter_en - exter_en
        # print("res_pel:")
        # print(res_pel)
        #----------------------------------------------------------------------------#
        # Dirichlet boundary condition residuals
        res_dir_bc = (
            tf.square(w0)
            + tf.square(dw0)
        ) / 2

        # Neumann boundary condition residuals
        F = self._transversal_force(dddwL)
        res_neu_bc = (
            tf.square(F / self.F - 1)
            + tf.square(ddwL)
        ) / 2

        # labeled data residual
        res_data = self._labeled_data_residual()

        # assemble and add loss term
        total_loss = self.w_pel * (res_pel) + (self.w_dir * res_dir_bc + self.w_neu * res_neu_bc) / 2 + self.w_data * res_data
        self.add_loss(total_loss)

        return (
            w * self.w_c,
            dw * self.w_c / self.x_c,
            M, 
            Q,
            ddddw * self.w_c / tf.math.pow(self.x_c, 4)
        )
    
    def _pde_residual(self, ddddw):
        ''' Computes the residual of the differential equation'''
        return ddddw
    def third_der(self, x):
        _, _, _, ddw, _ = fourth_gradient(self.mlp, x)
        return ddw
    
    def _internal_energy(self, x):
        '''Computes the internal energy'''
        return 0.5 * self.EI * (pow(self.w_c, 2) / pow(self.x_c, 3))*quadrature(self.third_der, x[0], x[-1])
    #tf.reduce_sum(ddw ** 2) * (self.labeled_data_x[1] - self.labeled_data_x[0])
    
    def _external_energy(self, w_L):
        '''Computes the external energy'''
        #*self.w_c
        return self.F*self.w_c*w_L
    
    def _pel_residual(self, w_L, x):
        ''' Computes the residual of the variation of potential energy loss'''
        return self._internal_energy(x) - self._external_energy(w_L)

    def _labeled_data_residual(self):
        ''' Computes the data loss using the mean squared error '''
        w = self.mlp(self.labeled_data_x)
        return tf.reduce_mean(tf.square(w - self.labeled_data_w))

    def _transversal_force(self, dddw):
        ''' Computes the transversal force '''
        return - self.EI * self.w_c / tf.math.pow(self.x_c, 3) * dddw
    
    def _moment(self, ddw):
        ''' Computes the bending moment '''
        return - self.EI * self.w_c / tf.square(self.x_c) * ddw

def build(*args, **kwargs):
    ''' Builds an evaluation model that returns the nodal displacements '''
    # define input shape
    x = tf.keras.Input(shape=(1,))
    # define which (custom) layers the model uses
    pinn = PINN(*args, **kwargs)
    w, dw, M, Q, ddddw = pinn(x)
    # connect input and output
    model = tf.keras.Model(inputs = [x], outputs = [w, dw, M, Q, ddddw])
    # define optimizer
    model.compile('adam')
    return model
    # x = tf.keras.Input(shape=(1,))
    # # Define which (custom) layers the model uses
    # pinn = PINN(*args, **kwargs)
    # w, dw, M, Q, ddddw = pinn(x)
    # # Connect input and output
    # model = tf.keras.Model(inputs=[x], outputs=[w, dw, M, Q, ddddw])
    
    # # Define the optimizer as SGD
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    
    # Compile the model
    # model.compile(optimizer=optimizer, loss='mse')  # Assuming 'mse' as the loss function
    # return model

convert = lambda x: tf.expand_dims(tf.constant(x, dtype='float32'), axis=1)

def analytical_ebb(EI, F, L, x):
    ''' Data genereation with the analytical solution of 
     the linear Euler-Bernoulli beam '''
    # compute analytical solution
    w = (0.5 * F * L * x ** 2 - 1. / 6. * F * x ** 3) / EI
    dw = (F * L * x - 0.5 * F * x ** 2) / EI
    M = F * (x - L)
    Q = F * np.ones(x.shape)
    ddddw = np.zeros(x.shape)

    # convert to tensors
    x = convert(x)
    w = convert(w)
    dw = convert(dw)
    M = convert(M)
    Q = convert(Q)
    ddddw = convert(ddddw)

    return x, w, dw, M, Q, ddddw

# Beam properties
# L = 0.2            # Length [m]
# W = 0.02           # Width [m]
# H = 0.02           # Height [m]
# E = 50e6           # Young's modulus [Pa]

L = 1            # Length [m]
W = 0.02           # Width [m]
H = 0.02           # Height [m]
E = 50e6  

# Cross-section parameters (assuming rectangular cross-section)
A = W * H
I = H ** 3 * W / 12
print("I:")
print(I)
param = [E, A, I, L]

# Compute the analytical solution
n = 1000 # number of collocation points
F = 5. # Transversal force at free end
x = np.linspace(0, L, n)

x, w, dw, M, Q, ddddw = analytical_ebb(E * I, F, L, x)

# Plot data
fig, axs = plt.subplots(3, 2, figsize=(8, 8))
axs[0, 0].plot(x, w, 'b', linestyle='--')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('w')
axs[0, 0].grid()

axs[0, 1].plot(x, dw, 'b', linestyle='--')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('dw')
axs[0, 1].grid()

axs[1, 0].plot(x, M, 'b', linestyle='--')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('M')
axs[1, 0].grid()

axs[1, 1].plot(x, Q, 'b', linestyle='--')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('Q')
axs[1, 1].grid()

axs[2, 0].plot(x, ddddw, 'b', linestyle='--')
axs[2, 0].set_xlabel('x')
axs[2, 0].set_ylabel('ddddw')
axs[2, 0].grid()

axs[2, 1].axis('off')
plt.tight_layout()
plt.show()

# Loss weights
WPEL = 1.
WDIR = 1000.
WNEU = 0.
WDATA = 0.

# Load model
model = build(WPEL, WDIR, WNEU, WDATA, param,
    labeled_data=[x, w],
    units=[8, 1],
    activation=['softplus', 'linear']
)
model.summary()

# Create collocation points
NCAL = 1000 # number of collocation points
x_cal = tf.expand_dims(tf.convert_to_tensor(np.linspace(0, L, NCAL), dtype=tf.float32), 1)

# Calibrate model
NEPOCHS = 2500
model.optimizer.learning_rate.assign(0.01)
h = model.fit([x_cal], [x_cal, x_cal, x_cal, x_cal, x_cal],
              epochs=NEPOCHS,
              verbose=2,
              batch_size=32)

plt.semilogy(np.linspace(1, NEPOCHS, NEPOCHS), h.history['loss'], label='training loss')
plt.grid(which='both')
plt.legend()
plt.show()

# # Save weights
# model.save_weights('weights_lebb_01.h5')

# # Load weights to skip the training
# model.load_weights('weights_lebb_01.h5')

# Get model predictions
w_pinn, dw_pinn, M_pinn, Q_pinn, ddddw_pinn = model(x)

# Plot data and prediction
fig, axs = plt.subplots(3, 2, figsize=(10, 12))
axs[0, 0].plot(x, w_pinn, 'r', label='PINN')
axs[0, 0].plot(x, w, 'b', linestyle='--', label='analytical')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('w')
axs[0, 0].grid()
axs[0, 0].legend()

axs[0, 1].plot(x, dw_pinn, 'r', label='PINN')
axs[0, 1].plot(x, dw, 'b', linestyle='--', label='analytical')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('dw')
axs[0, 1].grid()
axs[0, 1].legend()

axs[1, 0].plot(x, M_pinn, 'r', label='PINN')
axs[1, 0].plot(x, M, 'b', linestyle='--', label='analytical')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('M')
axs[1, 0].grid()
axs[1, 0].legend()

axs[1, 1].plot(x, Q_pinn, 'r', label='PINN')
axs[1, 1].plot(x, Q, 'b', linestyle='--', label='analytical')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('Q')
axs[1, 1].grid()
axs[1, 1].legend()

axs[2, 0].plot(x, ddddw_pinn, 'r', label='PINN')
axs[2, 0].plot(x, ddddw, 'b', linestyle='--', label='analytical')
axs[2, 0].set_xlabel('x')
axs[2, 0].set_ylabel('ddddw')
axs[2, 0].grid()
axs[2, 0].legend()

axs[2, 1].axis("off")

plt.tight_layout()
plt.show()
