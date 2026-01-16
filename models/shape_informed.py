"""
Shape-Informed Gate Functions for PUNN

Implements gates with geometric priors for parameter-efficient classification:
- SphericalShellGate: Spherical regions (d+2 parameters)
- FourierShellGate: Star-shaped regions via Fourier series
- EllipsoidGate: Axis-aligned or rotated ellipsoids
- SphericalHarmonicsGate: General direction-dependent radii

Reference: "Partition of Unity Neural Networks for Interpretable Classification"
"""

import torch
import torch.nn as nn
import numpy as np


class SphericalShellGate(nn.Module):
    """
    Spherical Shell Gate with constant radii.
    
    Domain: {x : r₁ ≤ ||x - c|| ≤ r₂}
    Gate: g(x) = sigmoid(s * (r₂ - ||x-c||)) * sigmoid(s * (||x-c|| - r₁))
    
    Parameters: d + 2 (center c ∈ ℝᵈ, radii r₁, r₂)
    
    Args:
        input_dim: Input feature dimension
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.center = nn.Parameter(torch.zeros(input_dim))
        self.log_r1 = nn.Parameter(torch.tensor(0.5))
        self.log_r2 = nn.Parameter(torch.tensor(1.5))
        self.log_sharpness = nn.Parameter(torch.tensor(2.0))
    
    def get_radii(self):
        r1 = torch.nn.functional.softplus(self.log_r1)
        r2 = r1 + torch.nn.functional.softplus(self.log_r2)
        return r1, r2
    
    def forward(self, x):
        r1, r2 = self.get_radii()
        sharpness = torch.exp(self.log_sharpness)
        dist = torch.norm(x - self.center, dim=-1)
        
        # Smooth indicator for shell: high when r1 < dist < r2
        inner = torch.sigmoid(sharpness * (dist - r1))
        outer = torch.sigmoid(sharpness * (r2 - dist))
        return inner * outer
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class FourierShellGate(nn.Module):
    """
    Fourier Shell Gate for 2D star-shaped regions.
    
    The radius is direction-dependent:
    r(θ) = a₀ + Σₖ (aₖcos(kθ) + bₖsin(kθ))
    
    Parameters: 2 + 2K + 1 (center, harmonics coefficients)
    
    Args:
        num_harmonics: Number of Fourier harmonics (default: 5)
    """
    def __init__(self, num_harmonics=5):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.center = nn.Parameter(torch.zeros(2))
        
        # Fourier coefficients: a0, a1, b1, a2, b2, ...
        self.a0 = nn.Parameter(torch.tensor(1.0))
        self.a_cos = nn.Parameter(torch.zeros(num_harmonics))
        self.b_sin = nn.Parameter(torch.zeros(num_harmonics))
        self.log_sharpness = nn.Parameter(torch.tensor(2.0))
    
    def compute_radius(self, theta):
        """Compute direction-dependent radius"""
        r = self.a0.clone()
        for k in range(1, self.num_harmonics + 1):
            r = r + self.a_cos[k-1] * torch.cos(k * theta)
            r = r + self.b_sin[k-1] * torch.sin(k * theta)
        return torch.abs(r) + 0.1  # Ensure positive
    
    def forward(self, x):
        diff = x - self.center
        dist = torch.norm(diff, dim=-1)
        theta = torch.atan2(diff[:, 1], diff[:, 0])
        r = self.compute_radius(theta)
        sharpness = torch.exp(self.log_sharpness)
        return torch.sigmoid(sharpness * (r - dist))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class EllipsoidGate(nn.Module):
    """
    Ellipsoid Gate with learnable center and axis radii.
    
    Domain: {x : (x-c)ᵀ diag(1/r²) (x-c) ≤ 1}
    Gate: g(x) = sigmoid(s * (1 - ||D(x-c)||²))
    
    Parameters: 2d + 1 (center c, radii r, sharpness s)
    
    Args:
        input_dim: Input feature dimension
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.center = nn.Parameter(torch.zeros(input_dim))
        self.log_radii = nn.Parameter(torch.zeros(input_dim))
        self.log_sharpness = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, x):
        radii = torch.exp(self.log_radii) + 0.1
        sharpness = torch.exp(self.log_sharpness)
        diff = (x - self.center) / radii
        dist_sq = torch.sum(diff ** 2, dim=-1)
        return torch.sigmoid(sharpness * (1 - dist_sq))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class SphericalHarmonicsGate(nn.Module):
    """
    Spherical Harmonics Gate for general direction-dependent radii.
    
    The radius is parametrized using spherical harmonics basis:
    r(n̂) = Σₗₘ cₗₘ Yₗₘ(n̂)
    
    For computational simplicity, we use polynomial basis up to degree L.
    
    Parameters: d + n_terms + 1 (center, coefficients, sharpness)
    where n_terms depends on degree L.
    
    Args:
        input_dim: Input feature dimension
        degree: Maximum polynomial degree (0, 1, or 2)
    """
    def __init__(self, input_dim, degree=2):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.center = nn.Parameter(torch.zeros(input_dim))
        
        # Number of basis terms
        if degree == 0:
            n_terms = 1
        elif degree == 1:
            n_terms = 1 + input_dim
        else:  # degree >= 2
            n_terms = 1 + input_dim + input_dim * (input_dim + 1) // 2
        
        self.coeffs = nn.Parameter(torch.zeros(n_terms))
        self.coeffs.data[0] = 1.0  # Initialize with spherical
        self.log_sharpness = nn.Parameter(torch.tensor(2.0))
    
    def compute_radius(self, n):
        """Compute direction-dependent radius from unit direction n"""
        batch_size = n.shape[0]
        d = self.input_dim
        
        # Build polynomial basis
        basis = [torch.ones(batch_size, device=n.device)]
        
        if self.degree >= 1:
            for i in range(d):
                basis.append(n[:, i])
        
        if self.degree >= 2:
            for i in range(d):
                for j in range(i, d):
                    basis.append(n[:, i] * n[:, j])
        
        basis = torch.stack(basis[:len(self.coeffs)], dim=1)
        r = torch.sum(self.coeffs * basis, dim=1)
        return torch.abs(r) + 0.1
    
    def forward(self, x):
        diff = x - self.center
        dist = torch.norm(diff, dim=-1, keepdim=True)
        n = diff / (dist + 1e-8)
        dist = dist.squeeze(-1)
        r = self.compute_radius(n)
        sharpness = torch.exp(self.log_sharpness)
        return torch.sigmoid(sharpness * (r - dist))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class PUNN_ShapeInformed(nn.Module):
    """
    PUNN with shape-informed gates.
    
    Parameters:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        gate_type: Type of shape-informed gate
            - 'spherical': SphericalShellGate
            - 'ellipsoid': EllipsoidGate
            - 'fourier': FourierShellGate (2D only)
            - 'harmonics': SphericalHarmonicsGate
        **gate_kwargs: Additional arguments for gate constructor
    """
    def __init__(self, input_dim, num_classes, gate_type='spherical', **gate_kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gate_type = gate_type
        
        gate_cls = {
            'spherical': SphericalShellGate,
            'ellipsoid': EllipsoidGate,
            'fourier': FourierShellGate,
            'harmonics': SphericalHarmonicsGate
        }[gate_type]
        
        # Handle gate-specific arguments
        if gate_type == 'fourier':
            self.gates = nn.ModuleList([
                gate_cls(**gate_kwargs) for _ in range(num_classes - 1)
            ])
        elif gate_type == 'harmonics':
            self.gates = nn.ModuleList([
                gate_cls(input_dim, **gate_kwargs) for _ in range(num_classes - 1)
            ])
        else:
            self.gates = nn.ModuleList([
                gate_cls(input_dim) for _ in range(num_classes - 1)
            ])
    
    def forward(self, x):
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.num_classes, device=x.device)
        remainder = torch.ones(batch_size, device=x.device)
        
        for i, gate in enumerate(self.gates):
            g = gate(x)
            h[:, i] = remainder * g
            remainder = remainder * (1 - g)
        
        h[:, -1] = remainder
        return h
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
