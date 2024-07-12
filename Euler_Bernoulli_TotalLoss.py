import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import math

class MLP(nn.Module):
    ''' A feed-forward neural network '''
    def __init__(self, units, activation):
        super(MLP, self).__init__()
        layers = []
        input_size = 1  # initial input size is 1

        for u, a in zip(units, activation):
            layers.append(nn.Linear(input_size, u))
            if a == 'relu':
                layers.append(nn.ReLU())
            elif a == 'tanh':
                layers.append(nn.Tanh())
            elif a == 'softplus':
                layers.append(nn.Softplus())
            elif a == 'linear':
                pass  # No activation for 'linear'
            input_size = u  # update input size for the next layer

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def gradient(f, x):
    ''' Computes the first derivative of a function f '''
    x = x.requires_grad_(True)
    u = f(x)
    du = autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return u, du

def second_gradient(f, x):
    ''' Computes the second derivative of a function f '''
    x = x.requires_grad_(True)
    u, du = gradient(f, x)
    ddu = autograd.grad(outputs=du, inputs=x, grad_outputs=torch.ones_like(du), create_graph=True)[0]
    return u, du, ddu

def third_gradient(f, x):
    ''' Computes the third derivative of a function f '''
    x = x.requires_grad_(True)
    u, du, ddu = second_gradient(f, x)
    dddu = autograd.grad(outputs=ddu, inputs=x, grad_outputs=torch.ones_like(ddu), create_graph=True)[0]
    return u, du, ddu, dddu

def fourth_gradient(f, x):
    ''' Computes the fourth derivative of a function f '''
    x = x.requires_grad_(True)
    u, du, ddu, dddu = third_gradient(f, x)
    ddddu = autograd.grad(outputs=dddu, inputs=x, grad_outputs=torch.ones_like(dddu), create_graph=True)[0]
    return u, du, ddu, dddu, ddddu

def getDisplacement(f, x):
    u = f(x)
    return u

def trapezoidalIntegration(y, x):
    dx = x[1] - x[0]
    result = torch.sum(y)
    result = result - (y[0] + y[-1]) / 2
    return result * dx

def GaussGlobalIntegration(y, a, b, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    transformed_nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    func_at_nodes = y(transformed_nodes)

    return 0.5*(b - a)*np.sum(weights*func_at_nodes)

def getInternalEnergyWithGaussGlobal(model, n, a, b, E, I):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    transformed_nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    transformed_nodes_tensor = torch.tensor(transformed_nodes, dtype=torch.float32).view(-1, 1)

    u = getDisplacement(model, transformed_nodes_tensor)
    _, _, func_at_nodes = second_gradient(lambda x: getDisplacement(model, x), transformed_nodes_tensor)
    func_at_nodes = func_at_nodes ** 2

    internal_energy = 0.5 * E * I * 0.5 * (b - a) * torch.sum(torch.tensor(weights, dtype=torch.float32).view(-1, 1) * func_at_nodes)
    return internal_energy

def getInternalEnergyWithGaussLocal(model, n, m, a, b, E, I):
    internal_energy = 0
    for i in range(n):
        internal_energy += getInternalEnergyWithGaussGlobal(model, m, a + ((b-a)*i/n), a + ((b-a)*(i+1)/n), E, I)
    return internal_energy

def getInternalEnergy(model, x, EI):
    u = getDisplacement(model, x)
    _, _, e = second_gradient(lambda x: getDisplacement(model, x), x)
    internal_energy_density = 0.5 * EI(x) * e ** 2
    internal_energy = trapezoidalIntegration(internal_energy_density, x)
    return internal_energy

def getExternalEnergy(uL):
    F = 5
    external_energy = F * uL
    return external_energy

def buildModel(**kwargs):
    ''' Builds an evaluation model that returns the nodal displacements '''
    pinn = MLP(**kwargs)
    return pinn

def analytical_ebb(EI, F, L, x):
    ''' Data generation with the analytical solution of the linear Euler-Bernoulli beam '''
    # compute analytical solution
    u = (0.5 * F * L * x ** 2 - 1. / 6. * F * x ** 3) / EI
    du = (F * L * x - 0.5 * F * x ** 2) / EI
    M = F * (x - L)
    Q = F * np.ones(x.shape)
    ddddu= np.zeros(x.shape)

    return x, u, du, M, Q, ddddu

# Define the model
model = buildModel(
    units=[10, 1],
    activation=['softplus', 'linear']
)
print(model)

# Define the input tensor
x = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)

# Beam properties
L = 1            # Length [m]
W = 0.02           # Width [m]
H = 0.02           # Height [m]
E = 50e6           # Young's modulus [Pa]

# Cross-section parameters (assuming rectangular cross-section)
A = W * H
I = H ** 3 * W / 12

# Define EI function
EI = lambda x: E * I + 0 * x

# Training parameters
learning_rate = 0.001
num_epochs = 20000

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Compute the displacement
    u = getDisplacement(model, x)
    
    # Compute the internal and external energies
    internal_energy = getInternalEnergy(model, x, EI)
    #internal_energy = getInternalEnergyWithGaussGlobal(model, 100, 0, 1, E, I)
    #internal_energy = getInternalEnergyWithGaussLocal(model, 10, 3, 0, 1, E, I)
    
    # Compute the displacement at x = 1
    uL = getDisplacement(model, torch.tensor([[1.0]], requires_grad=True))
    
    # Compute external energy
    external_energy = getExternalEnergy(uL)
    
    # Compute the potential energy (loss)
    potential_energy = internal_energy - external_energy

    #Dirichlet boundary condition
    u0 = getDisplacement(model, torch.tensor([[0.0]], requires_grad=True))
    _, du0 = gradient(lambda x: getDisplacement(model, x), torch.tensor([[0.0]], requires_grad=True))
    dir_loss = pow(u0, 2) + pow(du0, 2)

    #Total loss
    w_pe = 1.0
    w_dir = 1000.0
    total_loss = w_pe*potential_energy + w_dir*dir_loss
    
    # Backpropagate the loss
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {potential_energy.item():.4f}')

# Plot the results
u = getDisplacement(model, x).detach().numpy()
_, du, ddu, dddu, ddddu = fourth_gradient(lambda x: getDisplacement(model, x), x)
du, ddu, dddu, ddddu = du.detach().numpy(), ddu.detach().numpy(), dddu.detach().numpy(), ddddu.detach().numpy()
M = -E*I*ddu
Q = -E*I*dddu
x = x.detach().numpy()
_, u_analytical, du_a, M_a, Q_a, ddddu_a = analytical_ebb(E * I, 5, 1, x)

fig, axs = plt.subplots(3, 2, figsize=(8, 8))
axs[0, 0].plot(x, u, 'r')
axs[0, 0].plot(x, u_analytical, 'b', linestyle='--', label = 'analytical')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('w')
axs[0, 0].grid()

axs[0, 1].plot(x, du, 'r')
axs[0, 1].plot(x, du_a, 'b', linestyle='--', label = 'analytical')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('dw')
axs[0, 1].grid()

axs[1, 0].plot(x, M, 'r')
axs[1, 0].plot(x, M_a, 'b', linestyle='--', label = 'analytical')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('M')
axs[1, 0].grid()

axs[1, 1].plot(x, Q, 'r')
axs[1, 1].plot(x, Q_a, 'b', linestyle='--', label = 'analytical')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('Q')
axs[1, 1].grid()

axs[2, 0].plot(x, ddddu, 'r')
axs[2, 0].plot(x, ddddu_a, 'b', linestyle='--', label = 'analytical')
axs[2, 0].set_xlabel('x')
axs[2, 0].set_ylabel('ddddw')
axs[2, 0].grid()

axs[2, 1].axis('off')
plt.tight_layout()
plt.show()
