# PUNN: Partition of Unity Neural Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akram-VU/PUNN/blob/main/PUNN_Demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of **"Partition of Unity Neural Networks for Interpretable Classification with Explicit Class Regions"**.

## Overview

PUNN is a neural network architecture where class probabilities arise directly from a **partition of unity** construction, eliminating the need for softmax normalization. Each partition function $h_i(x)$ directly represents $P(\text{class } i \mid x)$.

### Key Features

- **Interpretable by design**: Hierarchical gate structure shows explicit accept/reject decisions
- **No softmax needed**: Valid probabilities guaranteed by construction ($\sum_i h_i(x) = 1$)
- **Flexible gates**: Support for sigmoid, bump, Gaussian, and shape-informed parameterizations
- **Parameter efficient**: Shape-informed gates achieve up to 300× parameter reduction when geometric priors match data

## Installation

```bash
git clone https://github.com/Akram-VU/punn.git
cd punn
pip install -r requirements.txt
```

## Quick Start

```python
from models import PUNN, MLP

# Create a PUNN classifier with sigmoid gates
model = PUNN(
    input_dim=784,      # Input feature dimension
    num_classes=10,     # Number of classes
    gate_type='sigma',  # Gate type: 'sigma', 'bump', or 'gaussian'
    hidden_dim=64       # Hidden dimension for gate networks
)

# Forward pass returns class probabilities (no softmax needed!)
probs = model(x)  # Shape: (batch_size, num_classes)
# probs.sum(dim=1) == 1 guaranteed by construction

# Training with cross-entropy loss
loss = -torch.log(probs[range(len(y)), y] + 1e-10).mean()
```

### Shape-Informed Gates

When geometric priors are available, use parameter-efficient shape-informed gates:

```python
from models import PUNN_ShapeInformed

# Spherical shell gates for radially structured data
model = PUNN_ShapeInformed(
    input_dim=2,
    num_classes=2,
    gate_type='spherical'  # Only 4 parameters!
)

# Other options: 'ellipsoid', 'fourier', 'harmonics'
```

## Architecture

PUNN constructs partition functions through a recursive product of gate functions:

```
h_1(x) = g_1(x)
h_i(x) = (1-g_1(x))...(1-g_{i-1}(x)) · g_i(x)    for i = 2,...,k-1
h_k(x) = (1-g_1(x))...(1-g_{k-1}(x))
```

This guarantees $\sum_{i=1}^k h_i(x) = 1$ and $h_i(x) \geq 0$ for all $x$.

## Reproducing Paper Results

### Synthetic Experiments (Table 1, Figures 1-2)

```bash
cd experiments
python synthetic_experiments.py
```

Datasets: Moons, Circles, XOR, Helix

### UCI Benchmarks (Table 2)

```bash
python uci_experiments.py
```

Datasets: Iris, Wine, Breast Cancer, Digits, Pendigits, Satimage, Optdigits

### MNIST (Table 3)

```bash
python mnist_experiments.py
```

### Shape-Informed Experiments (Table 4, Figures 3-4)

```bash
python shape_informed_experiments.py
```

## Results Summary

### UCI Benchmarks

| Dataset | PUNN-Sigma | PUNN-Bump | MLP |
|---------|------------|-----------|-----|
| Iris | 94.7% | 94.0% | 94.7% |
| Wine | 97.2% | 95.6% | 96.7% |
| Breast Cancer | 95.4% | 96.3% | 95.8% |
| Digits | 97.8% | 96.4% | 98.0% |

### MNIST

| Model | Accuracy | Parameters |
|-------|----------|------------|
| MLP | 98.19% | 269,322 |
| PUNN-Sigma | 97.85% | 2,403,081 |
| PUNN-Bump | 97.0% | 2,403,090 |

### Shape-Informed (Parameter Efficiency)

| Dataset | Gate Type | Accuracy | Params | Reduction |
|---------|-----------|----------|--------|-----------|
| Circles | Spherical Shell | 98.9% | 4 | **304×** |
| Circles | MLP | 98.6% | 1,218 | - |
| Conc. Rings | Spherical Shell | 100% | 10 | **125×** |

## Project Structure

```
punn/
├── models/
│   ├── __init__.py
│   ├── punn_models.py      # Core PUNN, gate functions, MLP baseline
│   └── shape_informed.py   # Shape-informed gate implementations
├── experiments/
│   ├── synthetic_experiments.py
│   ├── uci_experiments.py
│   ├── mnist_experiments.py
│   └── shape_informed_experiments.py
├── requirements.txt
└── README.md
```

## Gate Types

### MLP-Based Gates

- **SigmaGate**: $g(x) = \sigma(f_\theta(x))$ — smooth transition over entire real line
- **BumpGate**: $g(x) = \phi(\tanh(f_\theta(x)))$ — $C^\infty$ with compact support
- **GaussianGate**: $g(x) = \exp(-f_\theta(x)^2)$ — localized activation

### Shape-Informed Gates

- **SphericalShellGate**: Spherical regions ($d+2$ parameters)
- **EllipsoidGate**: Axis-aligned ellipsoids ($2d+1$ parameters)
- **FourierShellGate**: Star-shaped regions via Fourier series (2D)
- **SphericalHarmonicsGate**: General direction-dependent radii

## Citation

```bibtex
@article{aldroubi2025punn,
  title={Partition of Unity Neural Networks for Interpretable Classification with Explicit Class Regions},
  author={Aldroubi, Akram},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Windsurf AI (Cascade) was used solely as a coding assistant for experiment implementation.
