"""
PUNN Shape-Informed Gate Experiments

Reproduces Table 4 and Figures 3-4 from the paper.

Demonstrates dramatic parameter reductions when geometric priors match data.

Usage:
    python shape_informed_experiments.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

from models import PUNN, MLP, PUNN_ShapeInformed
from models.shape_informed import SphericalShellGate, FourierShellGate, EllipsoidGate, SphericalHarmonicsGate

# Configuration
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_concentric_rings(n_samples=1000, noise=0.05):
    """Generate 3-class concentric rings dataset."""
    n_per_class = n_samples // 3
    
    # Inner ring (class 0)
    theta = np.random.uniform(0, 2*np.pi, n_per_class)
    r = 0.5 + np.random.randn(n_per_class) * noise
    x0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Middle ring (class 1)
    theta = np.random.uniform(0, 2*np.pi, n_per_class)
    r = 1.5 + np.random.randn(n_per_class) * noise
    x1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Outer ring (class 2)
    theta = np.random.uniform(0, 2*np.pi, n_per_class)
    r = 2.5 + np.random.randn(n_per_class) * noise
    x2 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    X = np.vstack([x0, x1, x2])
    y = np.array([0]*n_per_class + [1]*n_per_class + [2]*n_per_class)
    
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=500):
    """Train model and return test accuracy."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            probs = model(X_batch)
            loss = -torch.log(probs[range(len(y_batch)), y_batch] + 1e-10).mean()
            loss.backward()
            optimizer.step()
    
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


def run_circles_experiment():
    """Compare spherical shell vs MLP on Circles dataset."""
    print("\n" + "="*60)
    print("CIRCLES DATASET")
    print("="*60)
    
    X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    results = {}
    
    # Spherical Shell
    model = PUNN_ShapeInformed(input_dim=2, num_classes=2, gate_type='spherical').to(DEVICE)
    acc = train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=500)
    params = model.count_parameters()
    print(f"Spherical Shell: {acc:.1f}% ({params} params)")
    results['spherical'] = {'accuracy': acc, 'parameters': params}
    
    # MLP baseline
    model = PUNN(input_dim=2, num_classes=2, gate_type='sigma', hidden_dim=32).to(DEVICE)
    acc = train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=200)
    params = model.count_parameters()
    print(f"MLP: {acc:.1f}% ({params} params)")
    results['mlp'] = {'accuracy': acc, 'parameters': params}
    
    reduction = results['mlp']['parameters'] / results['spherical']['parameters']
    print(f"Parameter reduction: {reduction:.0f}x")
    
    return results


def run_moons_experiment():
    """Compare Fourier shell vs MLP on Moons dataset."""
    print("\n" + "="*60)
    print("MOONS DATASET")
    print("="*60)
    
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    results = {}
    
    # Fourier Shell
    model = PUNN_ShapeInformed(input_dim=2, num_classes=2, gate_type='fourier', num_harmonics=5).to(DEVICE)
    acc = train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=500)
    params = model.count_parameters()
    print(f"Fourier Shell: {acc:.1f}% ({params} params)")
    results['fourier'] = {'accuracy': acc, 'parameters': params}
    
    # MLP baseline
    model = PUNN(input_dim=2, num_classes=2, gate_type='sigma', hidden_dim=32).to(DEVICE)
    acc = train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=200)
    params = model.count_parameters()
    print(f"MLP: {acc:.1f}% ({params} params)")
    results['mlp'] = {'accuracy': acc, 'parameters': params}
    
    reduction = results['mlp']['parameters'] / results['fourier']['parameters']
    print(f"Parameter reduction: {reduction:.0f}x")
    
    return results


def run_rings_experiment():
    """Compare spherical shell vs MLP on 3-class concentric rings."""
    print("\n" + "="*60)
    print("CONCENTRIC RINGS (3-CLASS)")
    print("="*60)
    
    X, y = make_concentric_rings(n_samples=1500, noise=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    results = {}
    
    # Spherical Shell
    model = PUNN_ShapeInformed(input_dim=2, num_classes=3, gate_type='spherical').to(DEVICE)
    acc = train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=500)
    params = model.count_parameters()
    print(f"Spherical Shell: {acc:.1f}% ({params} params)")
    results['spherical'] = {'accuracy': acc, 'parameters': params}
    
    # MLP baseline
    model = PUNN(input_dim=2, num_classes=3, gate_type='sigma', hidden_dim=32).to(DEVICE)
    acc = train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=200)
    params = model.count_parameters()
    print(f"MLP: {acc:.1f}% ({params} params)")
    results['mlp'] = {'accuracy': acc, 'parameters': params}
    
    reduction = results['mlp']['parameters'] / results['spherical']['parameters']
    print(f"Parameter reduction: {reduction:.0f}x")
    
    return results


def run_iris_experiment():
    """Compare spherical harmonics vs MLP on Iris dataset."""
    print("\n" + "="*60)
    print("IRIS DATASET")
    print("="*60)
    
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    results = {}
    
    # Spherical Harmonics (degree 2)
    model = PUNN_ShapeInformed(input_dim=4, num_classes=3, gate_type='harmonics', degree=2).to(DEVICE)
    acc = train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=500)
    params = model.count_parameters()
    print(f"Harmonics (deg=2): {acc:.1f}% ({params} params)")
    results['harmonics'] = {'accuracy': acc, 'parameters': params}
    
    # Ellipsoid
    model = PUNN_ShapeInformed(input_dim=4, num_classes=3, gate_type='ellipsoid').to(DEVICE)
    acc = train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=500)
    params = model.count_parameters()
    print(f"Ellipsoid: {acc:.1f}% ({params} params)")
    results['ellipsoid'] = {'accuracy': acc, 'parameters': params}
    
    # MLP baseline
    model = PUNN(input_dim=4, num_classes=3, gate_type='sigma', hidden_dim=32).to(DEVICE)
    acc = train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=200)
    params = model.count_parameters()
    print(f"MLP: {acc:.1f}% ({params} params)")
    results['mlp'] = {'accuracy': acc, 'parameters': params}
    
    return results


def run_ablation_harmonics_degree():
    """Ablation study: effect of harmonics degree on Iris."""
    print("\n" + "="*60)
    print("ABLATION: Harmonics Degree on Iris")
    print("="*60)
    
    data = load_iris()
    X, y = data.data, data.target
    
    results = {}
    
    for degree in [0, 1, 2]:
        accs = []
        for run in range(5):
            seed = SEED + run
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            model = PUNN_ShapeInformed(input_dim=4, num_classes=3, gate_type='harmonics', degree=degree).to(DEVICE)
            acc = train_and_evaluate(model, train_loader, test_loader, lr=0.01, epochs=500)
            accs.append(acc)
        
        params = model.count_parameters()
        mean, std = np.mean(accs), np.std(accs)
        print(f"Degree {degree}: {mean:.1f} ± {std:.1f}% ({params} params)")
        results[f'degree_{degree}'] = {'mean': mean, 'std': std, 'parameters': params}
    
    return results


def run_experiments():
    """Run all shape-informed experiments."""
    all_results = {}
    
    all_results['circles'] = run_circles_experiment()
    all_results['moons'] = run_moons_experiment()
    all_results['rings'] = run_rings_experiment()
    all_results['iris'] = run_iris_experiment()
    all_results['ablation_degree'] = run_ablation_harmonics_degree()
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Shape-Informed vs MLP")
    print("="*70)
    print(f"{'Dataset':<20} {'Gate Type':<15} {'Accuracy':<12} {'Params':<10} {'Reduction':<10}")
    print("-"*70)
    
    for dataset in ['circles', 'moons', 'rings']:
        res = all_results[dataset]
        shape_key = list(res.keys())[0]  # First key is shape-informed
        mlp_params = res['mlp']['parameters']
        shape_params = res[shape_key]['parameters']
        reduction = mlp_params / shape_params
        
        print(f"{dataset.capitalize():<20} {shape_key:<15} {res[shape_key]['accuracy']:<12.1f} {shape_params:<10} {reduction:.0f}x")
        print(f"{'':<20} {'MLP':<15} {res['mlp']['accuracy']:<12.1f} {mlp_params:<10}")
    
    # Save results
    with open('shape_informed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print("\nResults saved to shape_informed_results.json")
    
    return all_results


if __name__ == '__main__':
    print("PUNN Shape-Informed Experiments")
    print("="*60)
    run_experiments()
    print("\nDone!")
