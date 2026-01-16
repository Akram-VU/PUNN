"""
PUNN Synthetic Experiments

Reproduces Table 1 and Figures 1-2 from the paper.

Datasets: Moons, Circles, XOR, Helix
Models: PUNN-Sigma, PUNN-Bump, PUNN-Gaussian, MLP

Usage:
    python synthetic_experiments.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

from models import PUNN, MLP

# Configuration
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_xor(n_samples=1000, noise=0.1):
    """Generate XOR dataset with 4 clusters."""
    n_per_cluster = n_samples // 4
    centers = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    labels = [0, 1, 1, 0]
    
    X, y = [], []
    for center, label in zip(centers, labels):
        cluster = np.random.randn(n_per_cluster, 2) * noise + np.array(center)
        X.append(cluster)
        y.extend([label] * n_per_cluster)
    
    X = np.vstack(X)
    y = np.array(y)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def make_helix(n_samples=1000, noise=0.1, n_rotations=2):
    """Generate two interleaved spirals dataset."""
    n_per_class = n_samples // 2
    
    t1 = np.linspace(0, n_rotations * 2 * np.pi, n_per_class)
    x1 = t1 * np.cos(t1) + np.random.randn(n_per_class) * noise
    y1 = t1 * np.sin(t1) + np.random.randn(n_per_class) * noise
    
    t2 = np.linspace(0, n_rotations * 2 * np.pi, n_per_class)
    x2 = t2 * np.cos(t2 + np.pi) + np.random.randn(n_per_class) * noise
    y2 = t2 * np.sin(t2 + np.pi) + np.random.randn(n_per_class) * noise
    
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def get_dataset(name, n_samples=1000, noise=0.1):
    """Get dataset by name."""
    if name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=SEED)
    elif name == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=SEED)
    elif name == 'xor':
        X, y = make_xor(n_samples=n_samples, noise=noise)
    elif name == 'helix':
        X, y = make_helix(n_samples=n_samples, noise=noise)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y


def train_model(model, train_loader, epochs=200, lr=0.01):
    """Train a model using cross-entropy loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            probs = model(X_batch)
            loss = -torch.log(probs[range(len(y_batch)), y_batch] + 1e-10).mean()
            loss.backward()
            optimizer.step()
    
    return model


def evaluate_model(model, test_loader):
    """Evaluate model accuracy."""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            probs = model(X_batch)
            preds = probs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    
    return 100.0 * correct / total


def run_experiments():
    """Run all synthetic experiments."""
    datasets = ['moons', 'circles', 'xor', 'helix']
    gate_types = ['sigma', 'bump', 'gaussian']
    
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name.upper()}")
        print('='*60)
        
        # Load and preprocess data
        X, y = get_dataset(dataset_name, n_samples=1000, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        results[dataset_name] = {}
        
        # Test each gate type
        for gate_type in gate_types:
            print(f"\nPUNN-{gate_type.capitalize()}:")
            model = PUNN(input_dim=2, num_classes=2, gate_type=gate_type, hidden_dim=32)
            model = model.to(DEVICE)
            model = train_model(model, train_loader, epochs=200, lr=0.01)
            acc = evaluate_model(model, test_loader)
            params = model.count_parameters()
            print(f"  Accuracy: {acc:.1f}%")
            print(f"  Parameters: {params:,}")
            results[dataset_name][f'punn_{gate_type}'] = {'accuracy': acc, 'parameters': params}
        
        # MLP baseline
        print(f"\nMLP Baseline:")
        mlp = MLP(input_dim=2, num_classes=2, hidden_dims=[32, 32])
        mlp = mlp.to(DEVICE)
        mlp = train_model(mlp, train_loader, epochs=200, lr=0.01)
        acc = evaluate_model(mlp, test_loader)
        params = mlp.count_parameters()
        print(f"  Accuracy: {acc:.1f}%")
        print(f"  Parameters: {params:,}")
        results[dataset_name]['mlp'] = {'accuracy': acc, 'parameters': params}
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Test Accuracy (%)")
    print("="*70)
    print(f"{'Dataset':<12} {'PUNN-Sigma':<12} {'PUNN-Bump':<12} {'PUNN-Gaussian':<14} {'MLP':<10}")
    print("-"*70)
    for ds in datasets:
        row = f"{ds.capitalize():<12}"
        for model in ['punn_sigma', 'punn_bump', 'punn_gaussian', 'mlp']:
            row += f"{results[ds][model]['accuracy']:<12.1f}"
        print(row)
    
    # Save results
    with open('synthetic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to synthetic_results.json")
    
    return results


def plot_decision_boundaries(save_path='fig_decision_boundaries.pdf'):
    """Generate decision boundary visualization (Figure 1 in paper)."""
    datasets = ['moons', 'circles', 'xor', 'helix']
    gate_types = ['sigma', 'bump', 'gaussian']
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    for i, dataset_name in enumerate(datasets):
        X, y = get_dataset(dataset_name, n_samples=1000, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        for j, gate_type in enumerate(gate_types):
            ax = axes[i, j]
            
            model = PUNN(input_dim=2, num_classes=2, gate_type=gate_type, hidden_dim=32).to(DEVICE)
            model = train_model(model, train_loader, epochs=200, lr=0.01)
            
            # Create grid for decision boundary
            x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
            y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            
            grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(DEVICE)
            with torch.no_grad():
                Z = model(grid)[:, 1].cpu().numpy().reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
            ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
            ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='RdBu', 
                      edgecolors='white', s=20, alpha=0.7)
            
            if i == 0:
                ax.set_title(f'{gate_type.capitalize()} Gate', fontsize=12)
            if j == 0:
                ax.set_ylabel(dataset_name.capitalize(), fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved decision boundary plot to {save_path}")
    plt.close()


if __name__ == '__main__':
    print("PUNN Synthetic Experiments")
    print("="*60)
    
    results = run_experiments()
    
    print("\nGenerating decision boundary plots...")
    plot_decision_boundaries()
    
    print("\nDone!")
