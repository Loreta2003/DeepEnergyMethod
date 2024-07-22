import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import math
import time

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

def getW(f, x):
    output = f(x)
    w = output[:, 0]
    return w
  
def getU(f, x):
    output = f(x)
    u = output[:, 1]
    return u

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

# def getInternalEnergyWithGaussGlobal(model, n, a, b, E, I):
#     nodes, weights = np.polynomial.legendre.leggauss(n)
#     transformed_nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
#     transformed_nodes_tensor = torch.tensor(transformed_nodes, dtype=torch.float32).view(-1, 1)

#     u = getDisplacement(model, transformed_nodes_tensor)
#     _, _, func_at_nodes = second_gradient(lambda x: getDisplacement(model, x), transformed_nodes_tensor)
#     func_at_nodes = func_at_nodes ** 2

#     internal_energy = 0.5 * E * I * 0.5 * (b - a) * torch.sum(torch.tensor(weights, dtype=torch.float32).view(-1, 1) * func_at_nodes)
#     return internal_energy

# def getInternalEnergyWithGaussLocal(model, n, m, a, b, E, I):
#     internal_energy = 0
#     for i in range(n):
#         internal_energy += getInternalEnergyWithGaussGlobal(model, m, a + ((b-a)*i/n), a + ((b-a)*(i+1)/n), E, I)
#     return internal_energy

def getInternalEnergy(model, x, EI, EA):
    _, dw, ddw = second_gradient(lambda x: getW(model, x), x)
    _, du = gradient(lambda x: getU(model, x), x)
    internal_energy_density = 0.5 * ((EA(x)*((du + 0.5*(dw**2))**2))+EI(x) * ddw ** 2)
    internal_energy = trapezoidalIntegration(internal_energy_density, x)
    return internal_energy

def getExternalEnergy(wL):
    F = 1
    external_energy = F * wL
    return external_energy

def buildModel(**kwargs):
    ''' Builds an evaluation model that returns the nodal displacements '''
    pinn = MLP(**kwargs)
    return pinn

def analytical_ebb(EI, F, L, x):
    ''' Data generation with the analytical solution of the linear Euler-Bernoulli beam '''
    # compute analytical solution
    u = -((F/EI)**2)*(x**3)*(((x**2)/40) - (L*x/8) + ((L**2)/6))
    w = -(F/EI)*(x**2)*((x/6) - (L/2))
    du = -((F/EI)**2)*(x**2)*(((x**2)/8) - (L*x/2) + ((L**2)/2))
    dw = -(F/EI)*x*((x/2) - L)
    M = F * (x - L)
    Q = F * np.ones(x.shape)
    N = np.zeros(x.shape)

    return u, w, du, dw, M, Q, N

# Define the model
units = [20, 2]
model = buildModel(
    units=units,
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
EA = lambda x: E * A + 0 * x

# Training parameters
learning_rate = 0.001
num_epochs = 40000
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Compute the displacement
    u = getU(model, x)
    w = getW(model, x)
    
    # Compute the internal and external energies
    internal_energy = getInternalEnergy(model, x, EI, EA)
    #internal_energy = getInternalEnergyWithGaussGlobal(model, 4, 0, 1, E, I)
    #internal_energy = getInternalEnergyWithGaussLocal(model, 3, 1, 0, 1, E, I)
    
    # Compute the displacement at x = 1
    wL = getW(model, torch.tensor([[1.0]], requires_grad=True))
    
    # Compute external energy
    external_energy = getExternalEnergy(wL)
    
    # Compute the potential energy (loss)
    potential_energy = internal_energy - external_energy

    #Dirichlet boundary condition
    u0 = getU(model, torch.tensor([[0.0]], requires_grad=True))
    w0, dw0 = gradient(lambda x: getW(model, x), torch.tensor([[0.0]], requires_grad=True))
    dir_loss = pow(u0, 2) + pow(w0, 2) + pow(dw0, 2)

    #Total loss
    w_pe = 1.0
    w_dir = 1000.0
    total_loss = w_pe*potential_energy + w_dir*dir_loss
    
    # Backpropagate the loss
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {potential_energy.item():.4f}')

end_time = time.time()
training_duration = end_time - start_time
print(f'Training Duration: {training_duration:.2f} seconds')

# Plot the results
u = getU(model, x).detach().numpy()
u = u.reshape(-1, 1)
w = getW(model, x).detach().numpy()
w = w.reshape(-1, 1)
_, du, ddu, dddu = third_gradient(lambda x: getU(model, x), x)
_, dw, ddw, dddw = third_gradient(lambda x: getW(model, x), x)
du, ddu, dddu = du.detach().numpy(), ddu.detach().numpy(), dddu.detach().numpy()
dw, ddw, dddw = dw.detach().numpy(), ddw.detach().numpy(), dddw.detach().numpy()

force = 1

N = E*A*(0.5*pow(dw, 2) + du)
M = -E*I*ddu
Q = -E*I*dddw + N*dw
x = x.detach().numpy()
u_analytical, w_analytical, du_analytical, dw_analytical, M_a, Q_a, N_a = analytical_ebb(E * I, force, 1, x)

x_deformed_analytical = x + u_analytical
y_deformed_analytical = w_analytical

x_deformed_model = x + u
y_deformed_model = w

fig, axs = plt.subplots(2, 3, figsize=(8, 8))
axs[0, 0].plot(x_deformed_model, y_deformed_model, 'r')
axs[0, 0].plot(x_deformed_analytical, y_deformed_analytical, 'b', linestyle='--', label = 'analytical')
axs[0, 0].set_xlabel('x_deformed')
axs[0, 0].set_ylabel('y_deformed')
axs[0, 0].grid()

axs[0, 1].plot(x, du, 'r')
axs[0, 1].plot(x, du_analytical, 'b', linestyle='--', label = 'analytical')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('du')
axs[0, 1].grid()

axs[0, 2].plot(x, dw, 'r')
axs[0, 2].plot(x, dw_analytical, 'b', linestyle='--', label = 'analytical')
axs[0, 2].set_xlabel('x')
axs[0, 2].set_ylabel('dw')
axs[0, 2].grid()

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

axs[1, 2].plot(x, N, 'r')
axs[1, 2].plot(x, N_a, 'b', linestyle='--', label = 'analytical')
axs[1, 2].set_xlabel('x')
axs[1, 2].set_ylabel('N')
axs[1, 2].grid()


fig.suptitle(f'Training Duration: {training_duration:.2f} seconds\nEpochs: {num_epochs}\nUnits: {units}', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.show()
