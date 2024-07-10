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

def getDisplacement(f, x):
    z = f(x)
    u = (x - 1) * x * z
    return u

def trapezoidalIntegration(y, x):
    dx = x[1] - x[0]
    result = torch.sum(y)
    result = result - (y[0] + y[-1]) / 2
    return result * dx

def getInternalEnergy(model, x, EA):
    u = getDisplacement(model, x)
    _, e = gradient(lambda x: getDisplacement(model, x), x)
    internal_energy_density = 0.5 * EA(x) * e ** 2
    internal_energy = trapezoidalIntegration(internal_energy_density, x)
    return internal_energy

def getExternalEnergy(u, p, x):
    external_energy_density = p(x) * u
    external_energy = trapezoidalIntegration(external_energy_density, x)
    return external_energy

def buildModel(**kwargs):
    ''' Builds an evaluation model that returns the nodal displacements '''
    pinn = MLP(**kwargs)
    return pinn

def u_a(x):
    return np.sin(2 * np.pi * x)

# Define the model
model = buildModel(
    units=[10, 1],
    activation=['softplus', 'linear']
)
print(model)

# Define the input tensor
x = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)

# Define EA and p functions
EA = lambda x: 1 + 0 * x
p = lambda x: 4 * math.pi**2 * torch.sin(2 * math.pi * x)

# Training parameters
learning_rate = 0.001
num_epochs = 10000

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Compute the displacement
    u = getDisplacement(model, x)
    
    # Compute the internal and external energies
    internal_energy = getInternalEnergy(model, x, EA)
    external_energy = getExternalEnergy(u, p, x)
    
    # Compute the potential energy (loss)
    potential_energy = internal_energy - external_energy
    
    # Backpropagate the loss
    potential_energy.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {potential_energy.item():.4f}')

# Plot the results
u = getDisplacement(model, x).detach().numpy()
x = x.detach().numpy()
u_analytical = u_a(x)
plt.plot(x, u, label='NN')
plt.plot(x, u_analytical, label = "Analytical")
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()
