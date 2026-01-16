"""
PUNN MNIST Experiments

Reproduces Table 3 from the paper.

Models: PUNN-Sigma, PUNN-Bump, MLP

Usage:
    python mnist_experiments.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json

from models import PUNN, MLP

# Configuration
CONFIG = {
    'seed': 42,
    'input_dim': 784,
    'num_classes': 10,
    'hidden_dim': 256,
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 0.001,
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_mnist_loaders():
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        probs = model(X_batch)
        loss = -torch.log(probs[range(len(y_batch)), y_batch] + 1e-10).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader):
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
    """Run MNIST experiments."""
    print(f"Device: {DEVICE}")
    print(f"Configuration: {CONFIG}")
    
    train_loader, test_loader = get_mnist_loaders()
    print(f"\nDataset loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")
    
    results = {}
    
    # PUNN-Sigma
    print("\n" + "="*60)
    print("Training PUNN-Sigma")
    print("="*60)
    
    model = PUNN(
        input_dim=CONFIG['input_dim'],
        num_classes=CONFIG['num_classes'],
        gate_type='sigma',
        hidden_dim=CONFIG['hidden_dim']
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    params = model.count_parameters()
    print(f"Parameters: {params:,}")
    
    start_time = time.time()
    for epoch in range(CONFIG['epochs']):
        loss = train_epoch(model, train_loader, optimizer)
        if (epoch + 1) % 5 == 0:
            acc = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Test Acc={acc:.2f}%")
    
    train_time = time.time() - start_time
    final_acc = evaluate(model, test_loader)
    print(f"Final: {final_acc:.2f}% in {train_time:.1f}s")
    results['punn_sigma'] = {'accuracy': final_acc, 'parameters': params, 'time': train_time}
    
    # PUNN-Bump
    print("\n" + "="*60)
    print("Training PUNN-Bump")
    print("="*60)
    
    model = PUNN(
        input_dim=CONFIG['input_dim'],
        num_classes=CONFIG['num_classes'],
        gate_type='bump',
        hidden_dim=CONFIG['hidden_dim']
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    params = model.count_parameters()
    print(f"Parameters: {params:,}")
    
    start_time = time.time()
    for epoch in range(CONFIG['epochs']):
        loss = train_epoch(model, train_loader, optimizer)
        if (epoch + 1) % 5 == 0:
            acc = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Test Acc={acc:.2f}%")
    
    train_time = time.time() - start_time
    final_acc = evaluate(model, test_loader)
    print(f"Final: {final_acc:.2f}% in {train_time:.1f}s")
    results['punn_bump'] = {'accuracy': final_acc, 'parameters': params, 'time': train_time}
    
    # MLP Baseline
    print("\n" + "="*60)
    print("Training MLP Baseline")
    print("="*60)
    
    model = MLP(
        input_dim=CONFIG['input_dim'],
        num_classes=CONFIG['num_classes'],
        hidden_dims=[CONFIG['hidden_dim'], CONFIG['hidden_dim']]
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    params = model.count_parameters()
    print(f"Parameters: {params:,}")
    
    start_time = time.time()
    for epoch in range(CONFIG['epochs']):
        loss = train_epoch(model, train_loader, optimizer)
        if (epoch + 1) % 5 == 0:
            acc = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Test Acc={acc:.2f}%")
    
    train_time = time.time() - start_time
    final_acc = evaluate(model, test_loader)
    print(f"Final: {final_acc:.2f}% in {train_time:.1f}s")
    results['mlp'] = {'accuracy': final_acc, 'parameters': params, 'time': train_time}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Accuracy':<12} {'Parameters':<15} {'Time':<10}")
    print("-"*60)
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['accuracy']:<12.2f} {data['parameters']:<15,} {data['time']:<10.1f}s")
    
    # Save results
    with open('mnist_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to mnist_results.json")
    
    return results


if __name__ == '__main__':
    print("PUNN MNIST Experiments")
    print("="*60)
    run_experiments()
    print("\nDone!")
