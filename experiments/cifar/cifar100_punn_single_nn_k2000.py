"""
CIFAR-100: CNN + PUNN with Single-NN and FIXED readout to classes.

The partition-of-unity vector h is fixed-sum over K partitions.  Each class
gets an equal number of partitions, and the class probability is the sum of
its assigned partition components.  No learned V matrix.

For C=100, choose K a multiple of 100 (e.g., K=2000, 20 partitions/class).

This preserves the PUNN semantics: class probabilities are still constructed
from partition-of-unity components.
"""

import glob
import json
import os
import random
import shutil
import tarfile
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =============================================================================
# CONFIG
# =============================================================================

SMOKE = os.environ.get('SMOKE', '0') == '1'
NUM_PARTITIONS = int(os.environ.get('NUM_PARTITIONS', '2000'))
PUNN_HIDDEN_DIM = int(os.environ.get('PUNN_HIDDEN_DIM', '796'))

if NUM_PARTITIONS % 100 != 0:
    raise ValueError(f'NUM_PARTITIONS must be a multiple of 100, got {NUM_PARTITIONS}')

CONFIG = {
    'seeds': [2],
    'subsample_train': 50000,
    'subsample_test': 10000,
    'num_classes': 100,
    'num_partitions': NUM_PARTITIONS,
    'punn_hidden_dim': PUNN_HIDDEN_DIM,
    'partitions_per_class': NUM_PARTITIONS // 100,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': int(os.environ.get('EPOCHS', '5' if SMOKE else '60')),
    'dropout': 0.5,
    'smoke': SMOKE,
}

OUT_DIR = '.'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# =============================================================================
# DATA
# =============================================================================

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])
tf_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

# Try to use a pre-uploaded CIFAR-100 tarball to save time.
DATA_ROOT = './data'
EXTRACTED_DIR = os.path.join(DATA_ROOT, 'cifar-100-python')
if not os.path.isdir(EXTRACTED_DIR):
    os.makedirs(DATA_ROOT, exist_ok=True)
    candidate_patterns = [
        '/content/cifar-100-python.tar.gz',
        '/content/drive/MyDrive/cifar-100-python.tar.gz',
        '/content/drive/MyDrive/**/cifar-100-python.tar.gz',
        '/tmp/cifar-100-python.tar.gz',
        './cifar-100-python.tar.gz',
        os.path.join(DATA_ROOT, 'cifar-100-python.tar.gz'),
    ]
    candidates = []
    for pat in candidate_patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    if candidates:
        src = candidates[0]
        print(f'Using pre-downloaded tarball: {src}')
        dst = os.path.join(DATA_ROOT, 'cifar-100-python.tar.gz')
        if os.path.abspath(src) != os.path.abspath(dst) and not os.path.exists(dst):
            shutil.copy(src, dst)
        with tarfile.open(dst, 'r:gz') as tf:
            tf.extractall(DATA_ROOT)
        print(f'Extracted to {EXTRACTED_DIR}')
    else:
        print('No pre-downloaded tarball found; torchvision will download.')

_full_train = datasets.CIFAR100(root=DATA_ROOT, train=True,  download=True, transform=tf_train)
_full_test  = datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=tf_test)

rng = np.random.RandomState(12345)
train_idx = rng.choice(len(_full_train), CONFIG['subsample_train'], replace=False)
test_idx  = np.arange(min(CONFIG['subsample_test'], len(_full_test)))
train_ds = torch.utils.data.Subset(_full_train, train_idx.tolist())
test_ds  = torch.utils.data.Subset(_full_test,  test_idx.tolist())
print(f'Train: {len(train_ds)}   Test: {len(test_ds)}')


def make_loaders(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    tr = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                    num_workers=2, generator=g, drop_last=True)
    te = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)
    return tr, te


# =============================================================================
# MODEL
# =============================================================================

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.feature_dim = 256 * 4 * 4

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)


class PUNNSingleNNHeadFixed(nn.Module):
    """Single shared MLP computes all K-1 gates; class probs are fixed sums."""
    def __init__(self, input_dim, k, h, C):
        super().__init__()
        self.k = k
        self.C = C

        self.theta_net = nn.Sequential(
            nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(h, k - 1),
        )

        # Initialise final bias so E[h_i] = 1/K at t=0.
        with torch.no_grad():
            for i in range(k - 1):
                target = 1.0 / (k - i)
                init_bias = float(np.log(target / (1.0 - target)))
                self.theta_net[-1].bias[i] = init_bias

        # Fixed interleaved assignment: partition i assigned to class i % C.
        # class_mask[c, k] = 1 iff partition k belongs to class c.
        assignments = torch.arange(k) % C
        class_mask = F.one_hot(assignments, num_classes=C).float().t()  # [C, K]
        # We use it as a log mask: 0 for include, -inf for exclude.
        self.register_buffer('log_class_mask', torch.log(class_mask))

    def forward(self, x):
        # [B, K-1] gate pre-sigmoid logits.
        pre = self.theta_net(x)
        # log-space stick-breaking.
        log_g   = F.logsigmoid(pre)
        log_1mg = F.logsigmoid(-pre)
        cum_1mg = F.pad(torch.cumsum(log_1mg, dim=1), (1, 0))
        log_h = torch.empty(pre.size(0), self.k, device=x.device)
        log_h[:, :-1] = log_g + cum_1mg[:, :-1]
        log_h[:,  -1] = cum_1mg[:, -1]
        # log_p[b, c] = log( sum_{k in class c} h_k )
        log_p = torch.logsumexp(log_h.unsqueeze(1) + self.log_class_mask, dim=2)
        return log_p


class Model(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# TRAIN / EVAL
# =============================================================================

def train_one(seed):
    print(f'\n===== Seed {seed} =====')
    set_seed(seed)
    tr, te = make_loaders(seed)

    set_seed(seed)
    m = Model(PUNNSingleNNHeadFixed(4096, CONFIG['num_partitions'],
                                     CONFIG['punn_hidden_dim'], CONFIG['num_classes']))
    m = m.to(device)
    print(f'PUNN-SingleNN (fixed readout) params: {count_parameters(m):,}')

    opt = torch.optim.Adam(m.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG['epochs'])
    crit = nn.NLLLoss()

    best = 0.0
    history = []
    t0 = time.time()
    for ep in range(CONFIG['epochs']):
        m.train()
        for X, y in tr:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            log_p = m(X)
            loss = crit(log_p, y)
            loss.backward()
            opt.step()
        sch.step()

        m.eval()
        corr = tot = 0
        with torch.no_grad():
            for X, y in te:
                X = X.to(device)
                y = y.to(device)
                pred = m(X).argmax(1)
                corr += (pred == y).sum().item()
                tot += y.numel()
        acc = 100.0 * corr / tot
        if acc > best:
            best = acc
        history.append({'epoch': ep + 1, 'test_acc': acc})
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  ep {ep+1:2d}/{CONFIG["epochs"]}  test={acc:.2f}%  best={best:.2f}%')

    return {'seed': seed, 'best_acc': best, 'wall_s': time.time() - t0, 'history': history}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('CONFIG:', json.dumps(CONFIG, indent=2))
    _ref = Model(PUNNSingleNNHeadFixed(4096, CONFIG['num_partitions'],
                                        CONFIG['punn_hidden_dim'], CONFIG['num_classes']))
    total_params = count_parameters(_ref)
    print(f'PUNN-SingleNN (fixed readout) total params: {total_params:,}')
    del _ref
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    results = {'config': CONFIG, 'params': total_params,
               'date': datetime.now().isoformat(), 'device': str(device), 'runs': []}

    for seed in CONFIG['seeds']:
        r = train_one(seed)
        results['runs'].append(dict(**r))

        _partial = os.path.join(OUT_DIR, f'cifar100_punn_fixed_partial_after_seed{seed}.json')
        with open(_partial, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  [checkpoint] saved {_partial}')
        try:
            from google.colab import files as _colab_files
            _colab_files.download(_partial)
            print(f'  [checkpoint] auto-downloaded {_partial}')
        except Exception as _e:
            print(f'  [checkpoint] auto-download skipped ({_e})')

    accs = [r['best_acc'] for r in results['runs']]
    summary = {
        'mean': float(np.mean(accs)),
        'std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        'per_seed': accs,
    }
    print('\n' + '=' * 60)
    print(f'PUNN-SingleNN (fixed readout, CIFAR-100): {summary["mean"]:.2f}% +/- {summary["std"]:.2f}')
    print(f'seeds={accs}')
    print('=' * 60)

    results['summary'] = summary
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_json = os.path.join(OUT_DIR, f'cifar100_cnn_punn_fixed_{stamp}.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved:', out_json)

    try:
        from google.colab import files
        files.download(out_json)
    except Exception:
        pass


if __name__ == '__main__':
    main()
