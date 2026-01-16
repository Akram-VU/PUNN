"""
PUNN - Partition of Unity Neural Networks

Core model implementations for PUNN with various gate types.
"""

from .punn_models import (
    SigmaGate,
    BumpGate,
    GaussianGate,
    PUNN,
    MLP
)

from .shape_informed import (
    SphericalShellGate,
    FourierShellGate,
    EllipsoidGate,
    SphericalHarmonicsGate,
    PUNN_ShapeInformed
)

__all__ = [
    'SigmaGate',
    'BumpGate', 
    'GaussianGate',
    'PUNN',
    'MLP',
    'SphericalShellGate',
    'FourierShellGate',
    'EllipsoidGate',
    'SphericalHarmonicsGate',
    'PUNN_ShapeInformed'
]
