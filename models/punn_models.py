"""
PUNN Core Models

Implements Partition of Unity Neural Networks with various gate functions:
- SigmaGate: Sigmoid activation with MLP
- BumpGate: C∞ bump function with compact support
- GaussianGate: Gaussian activation

Reference: "Partition of Unity Neural Networks for Interpretable Classification"
"""

import torch
import torch.nn as nn
import numpy as np


class SigmaGate(nn.Module):
    """
    Sigma Gate: g(x) = σ(f_θ(x))
    
    A two-hidden-layer MLP with sigmoid output.
    
    Parameters:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (default: 64)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class BumpGate(nn.Module):
    """
    Bump Gate: g(x) = φ(tanh(f_θ(x)))
    
    Uses the C∞ bump function with compact support:
    φ(t) = exp(-1/(1-t²)) for |t| < 1, else 0
    
    The tanh ensures the MLP output stays within (-1, 1).
    
    Parameters:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (default: 64)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
    
    def bump_function(self, t):
        """C∞ bump function with compact support on (-1, 1)"""
        mask = torch.abs(t) < 1.0
        result = torch.zeros_like(t)
        t_safe = torch.clamp(t[mask], -0.999, 0.999)
        result[mask] = torch.exp(-1.0 / (1.0 - t_safe ** 2))
        return result
    
    def forward(self, x):
        t = self.net(x).squeeze(-1)
        return self.bump_function(t)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class GaussianGate(nn.Module):
    """
    Gaussian Gate: g(x) = exp(-f_θ(x)²)
    
    A two-hidden-layer MLP with Gaussian activation.
    
    Parameters:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (default: 64)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        t = self.net(x).squeeze(-1)
        return torch.exp(-t ** 2)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class PUNN(nn.Module):
    """
    Partition of Unity Neural Network
    
    Constructs partition functions h_1, ..., h_k satisfying:
    - sum(h_i(x)) = 1 for all x (partition of unity)
    - h_i(x) >= 0 for all x, i (non-negativity)
    
    Each h_i(x) directly represents P(class i | x).
    
    Parameters:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        gate_type: Type of gate function ('sigma', 'bump', 'gaussian')
        hidden_dim: Hidden dimension for gate networks
    """
    def __init__(self, input_dim, num_classes, gate_type='sigma', hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gate_type = gate_type
        self.hidden_dim = hidden_dim
        
        # Create k-1 gate functions
        gate_cls = {
            'sigma': SigmaGate,
            'bump': BumpGate,
            'gaussian': GaussianGate
        }[gate_type]
        
        self.gates = nn.ModuleList([
            gate_cls(input_dim, hidden_dim) for _ in range(num_classes - 1)
        ])
    
    def forward(self, x):
        """
        Compute partition functions h_1, ..., h_k
        
        h_1 = g_1
        h_i = (1-g_1)...(1-g_{i-1}) * g_i  for i = 2, ..., k-1
        h_k = (1-g_1)...(1-g_{k-1})
        
        Returns: (batch_size, num_classes) tensor of class probabilities
        """
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        # Compute gate values
        g_values = [gate(x) for gate in self.gates]
        
        # Compute partition functions recursively
        remainder = torch.ones(batch_size, device=x.device)
        
        for i in range(self.num_classes - 1):
            h[:, i] = remainder * g_values[i]
            remainder = remainder * (1 - g_values[i])
        
        # Last class gets the remainder
        h[:, -1] = remainder
        
        return h
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class MLP(nn.Module):
    """
    Standard MLP baseline with softmax output.
    
    Parameters:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions (default: [128, 64])
    """
    def __init__(self, input_dim, num_classes, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
