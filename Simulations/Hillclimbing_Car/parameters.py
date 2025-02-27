import torch 
import math
from torch import autograd

#########################
### Design Parameters ###
#########################
m = 2
n = 1

m1x_0 = torch.ones(m, 1) 
m2x_0 = 0 * 0 * torch.eye(m)

# Define physics
dt = 0.1        # Time resolution
mass = 1200     # Vehicle mass (kg)
g = 9.81        # Gravity (m/s^2)
Cr = 0.01       # Rolling resistance coefficient
Cd = 0.3        # Aerodynamic drag coefficient
A = 2.2         # Frontal area (m^2)
rho = 1.225     # Air density (kg/m^3)
Fmax = 4000     # Maximum engine force (N)

# Define controller
p = n
pid_params = [torch.tensor([1.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32)]


def road_slope(x: torch.Tensor):
    """
    Define the road slope as a function of position.
    
    Parameters:
        x: float
            Position along the road (m)
    
    Returns:
        float: Slope angle (in radians)
    """
    
    # return 0.1 * np.sin(0.05 * x)  # 10% grade sinusoidal hill
    if len(x.size()) == 2:
        if x[0] >= 100 and x[0] < 300:
            return torch.tensor([torch.pi / 6])
        else: 
            return torch.zeros(1)
    elif len(x.size()) == 3:
        theta = torch.zeros_like(x[:,0])
        theta[torch.where((x[:,1] > 100) & (x[:,1] < 300))] = torch.pi / 6
        return theta

    else: raise ValueError("Unrecognized size of input")

######################################################
####### State evolution function f for Car ###########
######################################################

def f(x: torch.Tensor, u: torch.Tensor):
    """
    Vehicle dynamics with a road slope.

    Parameters:


    Returns:
    
    """
    # Compute the road slope
    theta = road_slope(x)
    
    # Forces
    F_engine = torch.multiply(u.squeeze(2), Fmax)                          # Engine force
    F_gravity = mass*g*torch.sin(theta)                   # Gravity force along the slope
    F_rolling = mass * g * Cr * torch.cos(theta)              # Rolling resistance
    F_drag = 0.5 * rho * Cd * A * torch.pow(x[:,1], 2) * torch.cos(theta)   # Aerodynamic drag
    
    # Acceleration
    F_total = F_engine - (F_gravity + F_rolling + F_drag)
    a = F_total / mass  # Newton's second law
    derivative = torch.stack((x[:,1], a), dim=1)
    return torch.add(x, torch.multiply(derivative, dt))

##################################################
### Observation function h for Lorenz Atractor ###
##################################################
def h(x: torch.Tensor):
    return x[:,1].unsqueeze(2)

if __name__ == "__main__":
    x = torch.rand(2,1)
    u = torch.ones(1)
    breakpoint()
    x1 = f(x,u)
    print(x1)
    print(x1.size())

###############################################
### process noise Q and observation noise R ###
###############################################

Q_structure = torch.eye(m)
R_structure = torch.ones(n)