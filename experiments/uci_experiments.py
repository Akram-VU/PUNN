"""
PUNN UCI Benchmark Experiments

Reproduces Table 2 from the paper.

Datasets: Iris, Wine, Breast Cancer, Digits, Pendigits, Satimage, Optdigits
Models: PUNN-Sigma, PUNN-Bump, MLP

Usage:
    python uci_experiments.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import urllib.request
import json

from models import PUNN, MLP

# Configuration
CONFIG = {
    'seed': 42,
    'hidden_dim': 64,
    'batch_size': 32,
    'epochs': 100,
    'lr_sigma': 0.001,
    'lr_bump': 0.001,
    'lr_mlp': 0.001,
    'n_runs': 5,
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pendigits():
    """Load Pendigits dataset from UCI."""
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes"
    
    os.makedirs("./data", exist_ok=True)
    train_file, test_file = "./data/pendigits.tra", "./data/pendigits.tes"
    
    if not os.path.exists(train_file):
        print("  Downloading Pendigits dataset...")
        urllib.request.urlretrieve(train_url, train_file)
        urllib.request.urlretrieve(test_url, test_file)
    
    train_data = np.genfromtxt(train_file, delimiter=',')
    test_data = np.genfromtxt(test_file, delimiter=',')
    data = np.vstack([train_data, test_data])
    return data[:, :-1], data[:, -1].astype(int)


def load_satimage():
    """Load Satimage dataset from UCI."""
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst"
    
    os.makedirs("./data", exist_ok=True)
    train_file, test_file = "./data/sat.trn", "./data/sat.tst"
    
    if not os.path.exists(train_file):
        print("  Downloading Satimage dataset...")
        urllib.request.urlretrieve(train_url, train_file)
        urllib.request.urlretrieve(test_url, test_file)
    
    train_data = np.genfromtxt(train_file)
    test_data = np.genfromtxt(test_file)
    data = np.vstack([train_data, test_data])
    X, y = data[:, :-1], data[:, -1].astype(int)
    # Remap labels to 0-5
    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    return X, y


def load_optdigits():
    """Load Optdigits dataset from UCI."""
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
    
    os.makedirs("./data", exist_ok=True)
    train_file, test_file = "./data/optdigits.tra", "./data/optdigits.tes"
    
    if not os.path.exists(train_file):
        print("  Downloading Optdigits dataset...")
        urllib.request.urlretrieve(train_url, train_file)
        urllib.request.urlretrieve(test_url, test_file)
    
    train_data = np.genfromtxt(train_file, delimiter=',')
    test_data = np.genfromtxt(test_file, delimiter=',')
    data = np.vstack([train_data, test_data])
    return data[:, :-1], data[:, -1].astype(int)


def get_dataset(name):
    """Load dataset by name."""
    loaders = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
        'digits': load_digits,
        'pendigits': load_pendigits,
        'satimage': load_satimage,
        'optdigits': load_optdigits,
    }
    
    if name in ['pendigits', 'satimage', 'optdigits']:
        X, y = loaders[name]()
    else:
        data = loaders[name]()
        X, y = data.data, data.target
    
    return X, y


def train_and_evaluate(model, train_loader, test_loader, lr, epochs):
    """Train model and return test accuracy."""
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
    
    # Evaluate
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
    """Run UCI benchmark experiments."""
    datasets = ['iris', 'wine', 'breast_cancer', 'digits', 'pendigits', 'satimage', 'optdigits']
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name.upper()}")
        print('='*60)
        
        X, y = get_dataset(dataset_name)
        n_classes = len(np.unique(y))
        input_dim = X.shape[1]
        
        print(f"  Samples: {len(y)}, Features: {input_dim}, Classes: {n_classes}")
        
        results[dataset_name] = {'punn_sigma': [], 'punn_bump': [], 'mlp': []}
        
        for run in range(CONFIG['n_runs']):
            seed = CONFIG['seed'] + run
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
            
            # PUNN-Sigma
            model = PUNN(input_dim, n_classes, gate_type='sigma', hidden_dim=CONFIG['hidden_dim']).to(DEVICE)
            acc = train_and_evaluate(model, train_loader, test_loader, CONFIG['lr_sigma'], CONFIG['epochs'])
            results[dataset_name]['punn_sigma'].append(acc)
            
            # PUNN-Bump
            model = PUNN(input_dim, n_classes, gate_type='bump', hidden_dim=CONFIG['hidden_dim']).to(DEVICE)
            acc = train_and_evaluate(model, train_loader, test_loader, CONFIG['lr_bump'], CONFIG['epochs'])
            results[dataset_name]['punn_bump'].append(acc)
            
            # MLP
            model = MLP(input_dim, n_classes, hidden_dims=[128, 64]).to(DEVICE)
            acc = train_and_evaluate(model, train_loader, test_loader, CONFIG['lr_mlp'], CONFIG['epochs'])
            results[dataset_name]['mlp'].append(acc)
        
        # Print results for this dataset
        for model_name in ['punn_sigma', 'punn_bump', 'mlp']:
            accs = results[dataset_name][model_name]
            mean, std = np.mean(accs), np.std(accs)
            print(f"  {model_name}: {mean:.1f} ± {std:.1f}%")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Test Accuracy (%) - Mean over 5 runs")
    print("="*70)
    print(f"{'Dataset':<15} {'Classes':<8} {'PUNN-Sigma':<12} {'PUNN-Bump':<12} {'MLP':<10}")
    print("-"*70)
    
    for ds in datasets:
        X, y = get_dataset(ds)
        n_classes = len(np.unique(y))
        row = f"{ds:<15} {n_classes:<8}"
        for model in ['punn_sigma', 'punn_bump', 'mlp']:
            mean = np.mean(results[ds][model])
            row += f"{mean:<12.1f}"
        print(row)
    
    # Save results
    with open('uci_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to uci_results.json")
    
    return results


if __name__ == '__main__':
    print("PUNN UCI Benchmark Experiments")
    print("="*60)
    run_experiments()
    print("\nDone!")
