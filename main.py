# Code for applied math challenge 
# This will host the main construction of the PINN
# And some visualization
import matplotlib_inline
import seaborn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.distributed as dist
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PINN(nn.Module):
    def __init__(self, hidden_layers=[50, 50, 50, 50]):
        """
        Physics-Informed Neural Network for predator-prey system
        
        Args:
            hidden_layers: List containing number of neurons in each hidden layer
        """

        super(PINN, self).__init__()
        
        # Input layer: time t
        # Output layer: predator and prey populations (x,y)
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.Tanh())
        
        # Final hidden layer to output layer (2 outputs: x and y)
        layers.append(nn.Linear(hidden_layers[-1], 2))
        
        # Sequential model
        self.model = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
        # System parameters (all positive)
        # a: prey growth rate
        # b: prey predation rate
        # c: predator death rate
        # d: predator growth rate due to predation
        # These will be trainable parameters of the PINN
        self.a = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.b = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.c = nn.Parameter(torch.tensor(1.5, requires_grad=True))
        self.d = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
    def forward(self, t):
        """
        Forward pass to compute predator (y) and prey (x) populations
        
        Args:
            t: Time points tensor of shape [batch_size, 1]
            
        Returns:
            Tensor of shape [batch_size, 2] for (x,y) populations
        """
        return self.model(t)
    
    def compute_derivatives(self, t):
        """
        Compute the derivatives dx/dt and dy/dt using automatic differentiation
        
        Args:
            t: Time points tensor of shape [batch_size, 1] with requires_grad=True
            
        Returns:
            Tuple of (x, y, dx_dt, dy_dt)
        """
        t.requires_grad_(True)
        xy = self.forward(t)
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        
        # Compute gradients dx/dt and dy/dt using automatic differentiation
        dx_dt = torch.autograd.grad(
            x, t, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        dy_dt = torch.autograd.grad(
            y, t, 
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=True
        )[0]
        
        return x, y, dx_dt, dy_dt
    
    def compute_physics_loss(self, t):
        """
        Compute physics-informed loss based on the predator-prey equations
        
        Args:
            t: Time points tensor of shape [batch_size, 1]
            
        Returns:
            Physics-informed loss tensor
        """
        # Get populations and their derivatives
        x, y, dx_dt, dy_dt = self.compute_derivatives(t)
        
        # Predator-prey system of equations:
        # dx/dt = ax - bxy
        # dy/dt = dxy - cy
        
        # Compute residuals based on the differential equations
        f_x = dx_dt - (self.a * x - self.b * x * y)
        f_y = dy_dt - (self.d * x * y - self.c * y)
        
        # Mean squared error of residuals
        mse_x = torch.mean(torch.square(f_x))
        mse_y = torch.mean(torch.square(f_y))
        
        # Combined loss
        physics_loss = mse_x + mse_y
        
        return physics_loss
    
    def compute_data_loss(self, t_data, xy_data, mask=None):
        """
        Compute data-fitting loss for known data points
        
        Args:
            t_data: Time points tensor of shape [batch_size, 1]
            xy_data: Population tensor of shape [batch_size, 2]
            mask: Optional tensor for masking out unknown values
            
        Returns:
            Data loss tensor
        """
        xy_pred = self.forward(t_data)
        
        if mask is not None:
            # Apply mask to only compute loss on known values
            squared_error = torch.square(xy_pred - xy_data) * mask
            data_loss = torch.sum(squared_error) / torch.sum(mask)
        else:
            data_loss = torch.mean(torch.square(xy_pred - xy_data))
            
        return data_loss


def train_pinn(model, t_domain, t_data=None, xy_data=None, mask=None,
              epochs=10000, learning_rate=0.001, print_freq=500,
              physics_weight=1.0, data_weight=1.0):
    """
    Train the PINN model
    
    Args:
        model: PINN model
        t_domain: Tensor of time points for physics constraints
        t_data: Tensor of time points for data constraints (optional)
        xy_data: Tensor of (x,y) data points (optional)
        mask: Optional tensor for masking out unknown values
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        print_freq: Frequency of printing loss
        physics_weight: Weight of the physics loss term
        data_weight: Weight of the data loss term
        
    Returns:
        Lists of total loss, physics loss, and data loss histories
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # If we have data constraints
    has_data = (t_data is not None) and (xy_data is not None)
    
    # For tracking losses
    loss_history = []
    physics_loss_history = []
    data_loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Physics-informed loss
        physics_loss = model.compute_physics_loss(t_domain) * physics_weight
        
        # Data loss (if data is provided)
        if has_data:
            data_loss = model.compute_data_loss(t_data, xy_data, mask) * data_weight
            total_loss = physics_loss + data_loss
        else:
            data_loss = torch.tensor(0.0)
            total_loss = physics_loss
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Force parameters to stay positive (as stated in the problem)
        with torch.no_grad():
            model.a.data.clamp_(min=0.001)
            model.b.data.clamp_(min=0.001)
            model.c.data.clamp_(min=0.001)
            model.d.data.clamp_(min=0.001)
        
        # Record losses
        loss_history.append(total_loss.item())
        physics_loss_history.append(physics_loss.item())
        data_loss_history.append(data_loss.item() if has_data else 0.0)
        
        # Print progress
        if epoch % print_freq == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, "
                  f"Physics Loss: {physics_loss.item():.6f}, "
                  f"Data Loss: {data_loss.item() if has_data else 0.0:.6f}")
            print(f"Parameters - a: {model.a.item():.4f}, b: {model.b.item():.4f}, "
                  f"c: {model.c.item():.4f}, d: {model.d.item():.4f}")
    
    return loss_history, physics_loss_history, data_loss_history


def prepare_data_from_csv():
    # Data normalization parameters
    min_day = 0
    max_day = 90
    day_range = 90
    max_rabbits = 369
    max_bobcats = 155

    # Normalized rabbit data (day, population)
    normalized_rabbit_data = [
        [0.000000, 0.379404],
        [0.011111, 0.346883],
        [0.033333, 0.406504],
        [0.055556, 0.360434],
        [0.066667, 0.363144],
        [0.088889, 0.398374],
        [0.100000, 0.387534],
        [0.111111, 0.411924],
        [0.122222, 0.390244],
        [0.133333, 0.368564],
        [0.144444, 0.409214],
        [0.155556, 0.406504],
        [0.188889, 0.414634],
        [0.200000, 0.428184],
        [0.211111, 0.420054],
        [0.233333, 0.441734],
        [0.255556, 0.479675],
        [0.266667, 0.406504],
        [0.277778, 0.433604],
        [0.288889, 0.474255],
        [0.300000, 0.504065],
        [0.311111, 0.474255],
        [0.322222, 0.501355],
        [0.333333, 0.506775],
        [0.355556, 0.517615],
        [0.388889, 0.552846],
        [0.400000, 0.563686],
        [0.455556, 0.620596],
        [0.488889, 0.653117],
        [0.500000, 0.674797],
        [0.533333, 0.704607],
        [0.555556, 0.745257],
        [0.588889, 0.750678],
        [0.600000, 0.807588],
        [0.611111, 0.769648],
        [0.622222, 0.840108],
        [0.633333, 0.815718],
        [0.655556, 0.842818],
        [0.666667, 0.826558],
        [0.677778, 0.905149],
        [0.688889, 0.886179],
        [0.711111, 0.894309],
        [0.733333, 0.910569],
        [0.755556, 0.962060],
        [0.766667, 0.956640],
        [0.777778, 0.975610],
        [0.788889, 1.000000],
        [0.811111, 0.972900],
        [0.822222, 0.991870],
        [0.855556, 0.986450],
        [0.866667, 0.991870],
        [0.877778, 0.986450],
        [0.888889, 0.972900],
        [0.900000, 0.994580],
        [0.933333, 0.970190],
        [0.944444, 0.956640],
        [0.955556, 0.926829],
        [0.966667, 0.948509],
        [0.977778, 0.913279],
        [1.000000, 0.915989]
    ]

    # Normalized bobcat data (day, population)
    normalized_bobcat_data = [
        [0.044444, 0.690323],
        [0.055556, 0.716129],
        [0.066667, 0.600000],
        [0.088889, 0.593548],
        [0.100000, 0.580645],
        [0.122222, 0.587097],
        [0.133333, 0.541935],
        [0.144444, 0.593548],
        [0.177778, 0.522581],
        [0.188889, 0.490323],
        [0.255556, 0.503226],
        [0.277778, 0.380645],
        [0.300000, 0.464516],
        [0.344444, 0.451613],
        [0.355556, 0.432258],
        [0.377778, 0.406452],
        [0.388889, 0.458065],
        [0.400000, 0.412903],
        [0.444444, 0.361290],
        [0.466667, 0.412903],
        [0.500000, 0.348387],
        [0.511111, 0.380645],
        [0.522222, 0.445161],
        [0.555556, 0.419355],
        [0.577778, 0.451613],
        [0.588889, 0.393548],
        [0.600000, 0.522581],
        [0.611111, 0.445161],
        [0.633333, 0.483871],
        [0.644444, 0.419355],
        [0.677778, 0.477419],
        [0.711111, 0.516129],
        [0.722222, 0.451613],
        [0.744444, 0.574194],
        [0.766667, 0.567742],
        [0.777778, 0.587097],
        [0.788889, 0.625806],
        [0.811111, 0.600000],
        [0.833333, 0.658065],
        [0.844444, 0.625806],
        [0.866667, 0.619355],
        [0.888889, 0.716129],
        [0.922222, 0.845161],
        [0.955556, 0.896774],
        [0.977778, 1.000000]
    ]

    # Convert to PyTorch tensors
    rabbit_data_tensor = torch.tensor(normalized_rabbit_data, dtype=torch.float32)
    bobcat_data_tensor = torch.tensor(normalized_bobcat_data, dtype=torch.float32)

    # Create time points and population values
    t_rabbit = rabbit_data_tensor[:, 0:1]
    rabbit_population = rabbit_data_tensor[:, 1:2]
    t_bobcat = bobcat_data_tensor[:, 0:1]
    bobcat_population = bobcat_data_tensor[:, 1:2]

    # Get all unique time points
    t_unique = torch.unique(torch.cat([t_rabbit, t_bobcat], dim=0))
    t_data = t_unique.reshape(-1, 1)
    
    # Initialize population data and mask tensors
    xy_data = torch.zeros((t_data.shape[0], 2), dtype=torch.float32)
    mask = torch.zeros((t_data.shape[0], 2), dtype=torch.float32)
    
    # Fill in known rabbit populations and update mask
    for i in range(len(t_rabbit)):
        t_val = t_rabbit[i].item()
        idx = (t_data == t_val).nonzero(as_tuple=True)[0][0]
        xy_data[idx, 0] = rabbit_population[i]
        mask[idx, 0] = 1.0
    
    # Fill in known bobcat populations and update mask
    for i in range(len(t_bobcat)):
        t_val = t_bobcat[i].item()
        idx = (t_data == t_val).nonzero(as_tuple=True)[0][0]
        xy_data[idx, 1] = bobcat_population[i]
        mask[idx, 1] = 1.0
    
    # Return important data for model training and evaluation
    return {
        'min_day': min_day,
        'max_day': max_day, 
        'day_range': day_range,
        'max_rabbits': max_rabbits,
        'max_bobcats': max_bobcats,
        't_data': t_data,
        'xy_data': xy_data,
        'mask': mask,
        't_rabbit': t_rabbit,
        'rabbit_population': rabbit_population,
        't_bobcat': t_bobcat,
        'bobcat_population': bobcat_population
    }


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data from CSV files
    data_dict = prepare_data_from_csv()
    
    # Create time domain for physics constraints (more points than just the data)
    t_min, t_max = 0.0, 1.0  # Normalized time range
    n_domain_points = 1000
    t_domain = torch.linspace(t_min, t_max, n_domain_points, requires_grad=True).reshape(-1, 1)
    
    # Create PINN model
    model = PINN(hidden_layers=[64, 128, 128, 64])
    
    # Train the model using both physics constraints and data
    loss_history, physics_loss_history, data_loss_history = train_pinn(
        model, 
        t_domain, 
        data_dict['t_data'], 
        data_dict['xy_data'], 
        data_dict['mask'],
        epochs=20000,
        learning_rate=0.001,
        print_freq=1000,
        physics_weight=0.5,  # Adjust these weights as needed
        data_weight=1.0
    )
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, label='Total Loss')
    plt.semilogy(physics_loss_history, label='Physics Loss')
    plt.semilogy(data_loss_history, label='Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.title('PINN Training Loss History')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_history.png')
    
    # Evaluate trained model on a fine grid
    with torch.no_grad():
        t_eval = torch.linspace(t_min, t_max, 500).reshape(-1, 1)
        xy_pred = model(t_eval)
        x_pred = xy_pred[:, 0].numpy()
        y_pred = xy_pred[:, 1].numpy()
        t_eval_numpy = t_eval.squeeze().numpy()
        
        # De-normalize for plotting
        t_days = t_eval_numpy * data_dict['day_range'] + data_dict['min_day']
        rabbit_pop = x_pred * data_dict['max_rabbits']
        bobcat_pop = y_pred * data_dict['max_bobcats']
        
        # Original data points (de-normalized)
        rabbit_days = data_dict['t_rabbit'].numpy() * data_dict['day_range'] + data_dict['min_day']
        rabbit_data = data_dict['rabbit_population'].numpy() * data_dict['max_rabbits']
        bobcat_days = data_dict['t_bobcat'].numpy() * data_dict['day_range'] + data_dict['min_day']
        bobcat_data = data_dict['bobcat_population'].numpy() * data_dict['max_bobcats']
    
    # Plot solution vs data
    plt.figure(figsize=(15, 10))
    
    # Rabbit population vs time
    plt.subplot(2, 1, 1)
    plt.plot(t_days, rabbit_pop, 'b-', linewidth=2, label='PINN Prediction (Rabbits)')
    plt.scatter(rabbit_days, rabbit_data, c='blue', marker='o', alpha=0.6, label='Rabbit Data')
    plt.xlabel('Day')
    plt.ylabel('Rabbit Population')
    plt.title('Rabbit Population: PINN Prediction vs Actual Data')
    plt.grid(True)
    plt.legend()
    
    # Bobcat population vs time
    plt.subplot(2, 1, 2)
    plt.plot(t_days, bobcat_pop, 'r-', linewidth=2, label='PINN Prediction (Bobcats)')
    plt.scatter(bobcat_days, bobcat_data, c='red', marker='o', alpha=0.6, label='Bobcat Data')
    plt.xlabel('Day')
    plt.ylabel('Bobcat Population')
    plt.title('Bobcat Population: PINN Prediction vs Actual Data')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('population_fit.png')
    
    # Phase portrait
    plt.figure(figsize=(10, 8))
    plt.plot(rabbit_pop, bobcat_pop, 'g-', linewidth=2)
    plt.scatter(rabbit_data, bobcat_data, c='purple', alpha=0.6, label='Data Points')
    plt.xlabel('Rabbit Population')
    plt.ylabel('Bobcat Population')
    plt.title('Phase Portrait of Predator-Prey Dynamics')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('phase_portrait.png')
    
    # Print learned parameters
    print("\nLearned Parameters:")
    print(f"a (prey growth rate): {model.a.item():.6f}")
    print(f"b (prey predation rate): {model.b.item():.6f}")
    print(f"c (predator death rate): {model.c.item():.6f}")
    print(f"d (predator growth from predation): {model.d.item():.6f}")
    
    # Calculate and report additional metrics
    
    # Evaluate the system equations at specific points
    a, b, c, d = model.a.item(), model.b.item(), model.c.item(), model.d.item()
    
    # Print the system of equations with learned parameters
    print("\nLearned Predator-Prey System:")
    print(f"dx/dt = {a:.4f}x - {b:.4f}xy")
    print(f"dy/dt = {d:.4f}xy - {c:.4f}y")
    
    # Calculate equilibrium points
    # x=0, y=0 is always an equilibrium
    # The non-trivial equilibrium is (c/d, a/b)
    eq_x = c / d
    eq_y = a / b
    
    print("\nEquilibrium Points:")
    print(f"Trivial: (x,y) = (0,0)")
    print(f"Non-trivial: (x,y) = ({eq_x:.4f},{eq_y:.4f})")
    print(f"De-normalized: (Rabbits,Bobcats) = ({eq_x * data_dict['max_rabbits']:.1f},{eq_y * data_dict['max_bobcats']:.1f})")


if __name__ == "__main__":
    main()