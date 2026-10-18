"""
MNIST PUNN Ordering Study
=========================

For each of M random permutations pi of the class labels, train one PUNN
and record everything needed to answer the reviewer's ordering questions.

For every test sample x whose true class y is at position r = pi[y], we
record:
  * theta_j(x)  for j = 0, ..., K-2                (gate outputs)
  * R_j(x) = prod_{i<j} (1 - theta_i(x))            (remaining mass)
  * h_j(x)                                          (partition prob at pos j)
  * p_y(x) = h_{pi[y]}(x)                           (predicted prob of true class)
  * whether classified correctly

Analyses (all from the same trained models):
  (1) Overall order sensitivity : mean/std/min/max/range of Acc(pi)
  (2) Accuracy vs position      : A_r = mean acc of classes placed at position r
  (3) Remaining-mass analysis   : E[R_r(x) | correct] vs E[R_r(x) | wrong]
  (4) Controlled per-class move : Acc_c vs pos_pi(c)

Outputs (in this script's directory):
  mnist_ordering_study.json         (summary tables)
  mnist_ordering_study_raw/         (per-permutation .npz with gate traces)
  mnist_ordering_study_positional.pdf
  mnist_ordering_study_remaining_mass.pdf
  mnist_ordering_study_per_class.pdf
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_experiment import PUNN, DEVICE


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    'M': 20,                        # number of random permutations
    'num_classes': 10,
    'input_dim': 784,
    'hidden_dim': 128,
    'batch_size': 128,
    'epochs': 10,                   # MNIST + PUNN converges fast
    'lr': 1e-3,
    'gate_type': 'sigma',
    'seed_base': 42,                # each permutation uses seed_base + perm_idx
    'permutation_master_seed': 0,   # for reproducibility of the permutation set
    # -- Controlled-move sub-experiment --
    'controlled_target_classes': [0, 5, 9],   # move each of these to early/mid/late
    'controlled_positions':      [0, 4, 9],   # early, middle, late partition indices
    'controlled_seed_base': 1000,             # different seed range from main sweep
}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, 'mnist_ordering_study.json')
OUT_RAW_DIR = os.path.join(HERE, 'mnist_ordering_study_raw')
os.makedirs(OUT_RAW_DIR, exist_ok=True)

K = CONFIG['num_classes']
NUM_GATES = K - 1


# =============================================================================
# DATA
# =============================================================================

def get_loaders():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(os.path.join(HERE, 'data'), train=True,
                              download=True, transform=tf)
    test_ds = datasets.MNIST(os.path.join(HERE, 'data'), train=False,
                             download=True, transform=tf)
    return train_ds, test_ds


# =============================================================================
# TRAINING (permutation-aware)
# =============================================================================

def train_one_permutation(pi, train_ds, test_ds, seed):
    """
    Train a PUNN with class labels remapped by pi.
    pi[c] = position (partition index) assigned to original class c.

    Returns a dict with per-sample diagnostics on the test set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pi_t = torch.tensor(pi, dtype=torch.long, device=DEVICE)         # [K]
    inv_pi = torch.argsort(pi_t)                                     # position -> class

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=0)

    model = PUNN(CONFIG['input_dim'], K, CONFIG['gate_type'],
                 CONFIG['hidden_dim']).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    for ep in range(CONFIG['epochs']):
        model.train()
        for X, y in train_loader:
            X = X.view(X.size(0), -1).to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            y_pos = pi_t[y]                     # position targets

            opt.zero_grad()
            probs = model(X)                    # [B, K] on simplex
            loss = -torch.log(probs[torch.arange(len(y_pos), device=DEVICE),
                                    y_pos] + 1e-10).mean()
            loss.backward()
            opt.step()

    # ---- Collect full test diagnostics ------------------------------------
    model.eval()
    all_theta, all_R, all_H, all_y = [], [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.view(X.size(0), -1).to(DEVICE)
            B = X.size(0)
            theta = torch.stack([g(X) for g in model.gates], dim=1)   # [B, K-1]
            R = torch.ones(B, K, device=DEVICE)
            for j in range(NUM_GATES):
                R[:, j + 1] = R[:, j] * (1.0 - theta[:, j])
            H = torch.zeros(B, K, device=DEVICE)
            H[:, :NUM_GATES] = R[:, :NUM_GATES] * theta
            H[:, NUM_GATES] = R[:, NUM_GATES]

            all_theta.append(theta.cpu().numpy())
            all_R.append(R.cpu().numpy())
            all_H.append(H.cpu().numpy())
            all_y.append(y.numpy())

    theta = np.concatenate(all_theta)            # [N, K-1]
    R = np.concatenate(all_R)                    # [N, K]
    H = np.concatenate(all_H)                    # [N, K]
    y_true = np.concatenate(all_y)               # [N]

    # Predicted position = argmax over partition index
    pred_pos = H.argmax(axis=1)                  # [N]
    inv_pi_np = np.asarray(inv_pi.cpu().numpy())
    y_pred_class = inv_pi_np[pred_pos]           # map position -> original class

    correct = (y_pred_class == y_true)
    overall_acc = 100.0 * correct.mean()

    # Per-class accuracy (original class labels)
    per_class_acc = np.zeros(K)
    per_class_n = np.zeros(K, dtype=int)
    for c in range(K):
        m = (y_true == c)
        per_class_n[c] = int(m.sum())
        per_class_acc[c] = 100.0 * correct[m].mean() if m.any() else np.nan

    return dict(
        pi=np.asarray(pi, dtype=int),
        seed=int(seed),
        theta=theta.astype(np.float32),
        R=R.astype(np.float32),
        H=H.astype(np.float32),
        y_true=y_true.astype(np.int64),
        y_pred=y_pred_class.astype(np.int64),
        correct=correct.astype(bool),
        overall_acc=float(overall_acc),
        per_class_acc=per_class_acc.astype(np.float32),
        per_class_n=per_class_n.astype(np.int64),
    )


# =============================================================================
# AGGREGATE ANALYSES
# =============================================================================

def analyze_all(runs):
    """runs: list of dicts from train_one_permutation."""
    M = len(runs)

    # ---- 1. overall order sensitivity ----
    accs = np.array([r['overall_acc'] for r in runs])
    order_sens = dict(
        M=M,
        mean=float(accs.mean()),
        std=float(accs.std(ddof=1)) if M > 1 else 0.0,
        min=float(accs.min()),
        max=float(accs.max()),
        range=float(accs.max() - accs.min()),
        per_permutation=accs.tolist(),
    )

    # ---- 2. accuracy vs position ----
    #   for each (permutation, class) pair, we have (position, class-accuracy)
    pos_acc_pairs = []
    for r in runs:
        pi = r['pi']
        pca = r['per_class_acc']
        for c in range(K):
            pos_acc_pairs.append((int(pi[c]), float(pca[c])))
    pos_acc_arr = np.array(pos_acc_pairs)                          # [M*K, 2]
    A_r_mean = np.zeros(K); A_r_std = np.zeros(K); A_r_n = np.zeros(K, dtype=int)
    for r_idx in range(K):
        sel = pos_acc_arr[pos_acc_arr[:, 0] == r_idx, 1]
        A_r_mean[r_idx] = float(sel.mean()) if len(sel) else np.nan
        A_r_std[r_idx] = float(sel.std(ddof=1)) if len(sel) > 1 else 0.0
        A_r_n[r_idx] = len(sel)
    acc_vs_pos = dict(
        A_r_mean=A_r_mean.tolist(),
        A_r_std=A_r_std.tolist(),
        A_r_n=A_r_n.tolist(),
    )

    # ---- 3. remaining-mass analysis ----
    #   For a sample with true class y and this permutation's position r = pi[y],
    #   record R_r(x). Split by correct/incorrect and aggregate over r.
    Rr_all = {r: [] for r in range(K)}
    Rr_correct = {r: [] for r in range(K)}
    Rr_wrong = {r: [] for r in range(K)}
    p_y_correct = {r: [] for r in range(K)}
    p_y_wrong = {r: [] for r in range(K)}

    for run in runs:
        pi = run['pi']
        r_of_sample = pi[run['y_true']]                              # [N]
        Rmat = run['R']                                              # [N, K]
        Hmat = run['H']
        for r_idx in range(K):
            m = (r_of_sample == r_idx)
            if not m.any():
                continue
            rmasses = Rmat[m, r_idx]
            hvals = Hmat[m, r_idx]
            corr_m = run['correct'][m]
            Rr_all[r_idx].extend(rmasses.tolist())
            Rr_correct[r_idx].extend(rmasses[corr_m].tolist())
            Rr_wrong[r_idx].extend(rmasses[~corr_m].tolist())
            p_y_correct[r_idx].extend(hvals[corr_m].tolist())
            p_y_wrong[r_idx].extend(hvals[~corr_m].tolist())

    def _stats(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return dict(n=0, mean=None, std=None, median=None)
        return dict(
            n=int(arr.size),
            mean=float(arr.mean()),
            std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            median=float(np.median(arr)),
        )

    remaining_mass = dict(
        by_position={
            r_idx: dict(
                all=_stats(Rr_all[r_idx]),
                correct=_stats(Rr_correct[r_idx]),
                wrong=_stats(Rr_wrong[r_idx]),
                p_y_correct=_stats(p_y_correct[r_idx]),
                p_y_wrong=_stats(p_y_wrong[r_idx]),
            )
            for r_idx in range(K)
        }
    )

    # ---- 4. per-class accuracy vs position ----
    #   For each original class c: list of (position, acc) across permutations.
    per_class_curve = {c: {'positions': [], 'accs': []} for c in range(K)}
    for run in runs:
        pi = run['pi']
        pca = run['per_class_acc']
        for c in range(K):
            per_class_curve[c]['positions'].append(int(pi[c]))
            per_class_curve[c]['accs'].append(float(pca[c]))

    #   Simple linear fit acc_c = alpha_c + beta_c * pos, over permutations
    per_class_fit = {}
    for c in range(K):
        pos = np.array(per_class_curve[c]['positions'], dtype=float)
        acc = np.array(per_class_curve[c]['accs'], dtype=float)
        if len(pos) >= 2:
            beta, alpha = np.polyfit(pos, acc, 1)
            per_class_fit[c] = dict(alpha=float(alpha), beta=float(beta))
        else:
            per_class_fit[c] = dict(alpha=None, beta=None)

    return dict(
        order_sensitivity=order_sens,
        accuracy_vs_position=acc_vs_pos,
        remaining_mass=remaining_mass,
        per_class_curve={str(c): per_class_curve[c] for c in range(K)},
        per_class_fit={str(c): per_class_fit[c] for c in range(K)},
    )


# =============================================================================
# FIGURES
# =============================================================================

def plot_accuracy_vs_position(analysis, out_pdf):
    A = analysis['accuracy_vs_position']
    mean = np.asarray(A['A_r_mean'])
    std = np.asarray(A['A_r_std'])
    r_vals = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(r_vals, mean, yerr=std, fmt='o-', color='#3870b8',
                capsize=3, linewidth=1.4, markersize=6)
    ax.set_xlabel('Position in ordering (partition index $r$)')
    ax.set_ylabel(r'$A_r$ : mean class accuracy at position $r$ (%)')
    ax.set_title('Accuracy vs partition position across random orderings')
    ax.set_xticks(r_vals)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_pdf}')


def plot_remaining_mass(analysis, out_pdf):
    RM = analysis['remaining_mass']['by_position']
    r_vals = np.arange(K)
    mean_all = np.array([RM[r]['all']['mean'] if RM[r]['all']['mean'] is not None
                         else np.nan for r in range(K)])
    mean_corr = np.array([RM[r]['correct']['mean'] if RM[r]['correct']['mean']
                          is not None else np.nan for r in range(K)])
    mean_wrong = np.array([RM[r]['wrong']['mean'] if RM[r]['wrong']['mean']
                           is not None else np.nan for r in range(K)])

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(r_vals, mean_all, 'o-', label='all', color='black', linewidth=1.3)
    ax.plot(r_vals, mean_corr, 's-', label='correct', color='#2b8a3e')
    ax.plot(r_vals, mean_wrong, 'x-', label='incorrect', color='#c92a2a')
    ax.set_xlabel('Position of true class $r = \\pi(y)$')
    ax.set_ylabel(r'$E[\, R_r(x)\,]$ : mean remaining mass at true-class gate')
    ax.set_title(r'Remaining mass at the true-class gate vs its position')
    ax.set_xticks(r_vals)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_pdf}')


def plot_classwise_overlay(analysis, out_pdf):
    """All 10 classes on one axis: accuracy vs position, one line per class."""
    curves = analysis['per_class_curve']
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmap = plt.get_cmap('tab10')
    for c in range(K):
        pos = np.array(curves[str(c)]['positions'])
        acc = np.array(curves[str(c)]['accs'])
        # Aggregate by position (in case multiple perms placed c at same slot)
        order = np.argsort(pos)
        pos_s, acc_s = pos[order], acc[order]
        # per-position mean (in case of ties)
        agg = {}
        for p_, a_ in zip(pos_s, acc_s):
            agg.setdefault(int(p_), []).append(float(a_))
        xs = sorted(agg)
        ys = [np.mean(agg[p]) for p in xs]
        ax.plot(xs, ys, 'o-', color=cmap(c), label=f'class {c}',
                linewidth=1.2, markersize=5, alpha=0.85)
    ax.set_xlabel('Position in ordering (partition index)')
    ax.set_ylabel('per-class accuracy (%)')
    ax.set_title('Classwise accuracy vs sequence position (all classes)')
    ax.set_xticks(range(K))
    ax.grid(alpha=0.3)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc='lower left',
              bbox_to_anchor=(0.0, -0.28))
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_pdf}')


def write_remaining_mass_table(analysis, out_txt, out_tex):
    """Text + LaTeX table of R_r vs position, split by correct/incorrect."""
    RM = analysis['remaining_mass']['by_position']

    # --- plain text ---
    lines = []
    lines.append('Remaining mass at true-class gate  R_r(x) = prod_{i<r} (1 - theta_i(x))')
    lines.append('Aggregated over all permutations placing the true class at position r.')
    lines.append('')
    header = f'{"pos r":>6}{"n_all":>10}{"E[R_r]":>12}' \
             f'{"n_correct":>12}{"E[R_r|OK]":>14}' \
             f'{"n_wrong":>10}{"E[R_r|err]":>14}' \
             f'{"E[p_y|OK]":>14}{"E[p_y|err]":>14}'
    lines.append(header)
    lines.append('-' * len(header))
    for r_idx in range(K):
        s = RM[r_idx]
        def _fmt(x):
            return f'{x:.4f}' if isinstance(x, (int, float)) and x is not None else '  n/a '
        lines.append(
            f'{r_idx:>6}'
            f'{s["all"]["n"]:>10}'
            f'{_fmt(s["all"]["mean"]):>12}'
            f'{s["correct"]["n"]:>12}'
            f'{_fmt(s["correct"]["mean"]):>14}'
            f'{s["wrong"]["n"]:>10}'
            f'{_fmt(s["wrong"]["mean"]):>14}'
            f'{_fmt(s["p_y_correct"]["mean"]):>14}'
            f'{_fmt(s["p_y_wrong"]["mean"]):>14}'
        )
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved {out_txt}')

    # --- LaTeX ---
    tex = []
    tex.append(r'\begin{table}[t]')
    tex.append(r'\centering')
    tex.append(r'\caption{Remaining mass $R_r(x)=\prod_{i<r}(1-\theta_i(x))$ at the '
               r'true-class gate, and predicted true-class probability '
               r'$p_y(x)=h_{\pi(y)}(x)$, aggregated over all random permutations that '
               r'placed the true class at position $r$. If errors were caused by '
               r'earlier gates absorbing too much mass, $\mathbb{E}[R_r\mid\text{err}]$ '
               r'would be systematically smaller than $\mathbb{E}[R_r\mid\text{OK}]$.}')
    tex.append(r'\label{tab:mnist-remaining-mass}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{rrrrrrr}')
    tex.append(r'\toprule')
    tex.append(r'$r$ & $\mathbb{E}[R_r]$ & $\mathbb{E}[R_r\mid\text{OK}]$ & '
               r'$\mathbb{E}[R_r\mid\text{err}]$ & $\mathbb{E}[p_y\mid\text{OK}]$ & '
               r'$\mathbb{E}[p_y\mid\text{err}]$ & $n$ \\')
    tex.append(r'\midrule')
    for r_idx in range(K):
        s = RM[r_idx]
        def _tf(x):
            return f'{x:.3f}' if isinstance(x, (int, float)) and x is not None else '--'
        tex.append(
            f'{r_idx} & '
            f'{_tf(s["all"]["mean"])} & '
            f'{_tf(s["correct"]["mean"])} & '
            f'{_tf(s["wrong"]["mean"])} & '
            f'{_tf(s["p_y_correct"]["mean"])} & '
            f'{_tf(s["p_y_wrong"]["mean"])} & '
            f'{s["all"]["n"]} \\\\'
        )
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\end{table}')
    with open(out_tex, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f'Saved {out_tex}')


# =============================================================================
# CONTROLLED-MOVE EXPERIMENT
# =============================================================================

def build_controlled_permutations(target_c, positions, base_seed):
    """Return len(positions) permutations, each placing class target_c at
    the corresponding position, with the remaining K-1 classes in the SAME
    relative order across the three conditions (eliminating confounds)."""
    rng = np.random.RandomState(base_seed)
    others = [c for c in range(K) if c != target_c]
    rng.shuffle(others)                # fixed random order of the other classes
    perms = []
    for r in positions:
        pi_pos = list(others[:r]) + [target_c] + list(others[r:])
        # convert to class->position mapping
        pi = np.zeros(K, dtype=int)
        for pos_i, cls in enumerate(pi_pos):
            pi[cls] = pos_i
        perms.append(pi)
    return perms


def run_controlled_move(train_ds, test_ds):
    """For each target class, train the model at early/mid/late positions."""
    positions = CONFIG['controlled_positions']
    seed_base = CONFIG['controlled_seed_base']
    all_results = []

    for tc_i, target_c in enumerate(CONFIG['controlled_target_classes']):
        perms = build_controlled_permutations(target_c, positions,
                                              base_seed=seed_base + 7 * tc_i)
        print(f'\n--- controlled move: class {target_c} '
              f'at positions {positions} ---')
        for pi, r_pos in zip(perms, positions):
            seed = seed_base + 100 * tc_i + r_pos
            print(f'  class {target_c} @ pos {r_pos}   pi={pi.tolist()}  seed={seed}')
            t0 = time.time()
            r = train_one_permutation(pi, train_ds, test_ds, seed)
            print(f'    overall acc = {r["overall_acc"]:.2f}%  '
                  f'class-{target_c} acc = {r["per_class_acc"][target_c]:.2f}%  '
                  f'({time.time() - t0:.1f}s)')
            all_results.append(dict(
                target_class=int(target_c),
                position=int(r_pos),
                pi=r['pi'].tolist(),
                seed=int(seed),
                overall_acc=float(r['overall_acc']),
                target_class_acc=float(r['per_class_acc'][target_c]),
                per_class_acc=r['per_class_acc'].tolist(),
                per_class_n=r['per_class_n'].tolist(),
            ))
    return all_results


def plot_controlled_move(controlled_results, out_pdf):
    """Bar plot: for each target class, one bar per position (early/mid/late)."""
    positions = CONFIG['controlled_positions']
    target_classes = CONFIG['controlled_target_classes']
    labels = ['early', 'middle', 'late']

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.8 / len(positions)
    x_base = np.arange(len(target_classes))
    cmap = plt.get_cmap('viridis')

    for i, r_pos in enumerate(positions):
        vals = []
        for tc in target_classes:
            hits = [h for h in controlled_results
                    if h['target_class'] == tc and h['position'] == r_pos]
            vals.append(hits[0]['target_class_acc'] if hits else np.nan)
        offset = (i - (len(positions) - 1) / 2) * width
        ax.bar(x_base + offset, vals, width=width * 0.9,
               color=cmap(i / max(1, len(positions) - 1)),
               label=f'position {r_pos} ({labels[i]})',
               edgecolor='black', linewidth=0.5)
        for x_i, v in zip(x_base + offset, vals):
            if not np.isnan(v):
                ax.text(x_i, v + 0.3, f'{v:.1f}', ha='center', va='bottom',
                        fontsize=8)

    ax.set_xticks(x_base)
    ax.set_xticklabels([f'class {c}' for c in target_classes])
    ax.set_ylabel('per-class accuracy (%)')
    ax.set_title('Controlled move: same class at early / middle / late positions')
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([min(80, ax.get_ylim()[0]), 101])
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_pdf}')


def plot_per_class(analysis, out_pdf):
    curves = analysis['per_class_curve']
    fits = analysis['per_class_fit']

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
    for c in range(K):
        ax = axes[c // 5][c % 5]
        pos = np.array(curves[str(c)]['positions'])
        acc = np.array(curves[str(c)]['accs'])
        ax.scatter(pos, acc, s=22, alpha=0.7, color='#3870b8')
        if fits[str(c)]['alpha'] is not None:
            xs = np.linspace(0, K - 1, 20)
            ys = fits[str(c)]['alpha'] + fits[str(c)]['beta'] * xs
            ax.plot(xs, ys, '-', color='#c92a2a', linewidth=1.2,
                    label=fr'$\beta$={fits[str(c)]["beta"]:+.2f}')
            ax.legend(loc='lower left', fontsize=8, frameon=False)
        ax.set_title(f'class {c}', fontsize=10)
        ax.set_xticks(range(K))
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('position in ordering')
    for ax in axes[:, 0]:
        ax.set_ylabel('accuracy (%)')
    fig.suptitle('Per-class accuracy vs ordering position (across random permutations)',
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_pdf}')


# =============================================================================
# MAIN
# =============================================================================

def sample_permutations(M, K, master_seed):
    rng = np.random.RandomState(master_seed)
    perms = []
    seen = set()
    # Always include the identity as permutation 0 for a reference
    identity = tuple(range(K))
    perms.append(np.array(identity, dtype=int))
    seen.add(identity)
    while len(perms) < M:
        p = tuple(rng.permutation(K).tolist())
        if p in seen:
            continue
        seen.add(p)
        perms.append(np.array(p, dtype=int))
    return perms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-sweep', action='store_true',
                   help='Skip the M-permutation main sweep; load existing '
                        'mnist_ordering_study.json and go straight to the '
                        'controlled-move sub-experiment.')
    p.add_argument('--skip-controlled', action='store_true',
                   help='Only run the main sweep; skip the controlled move.')
    return p.parse_args()


def main():
    args = parse_args()

    print('=' * 72)
    print('MNIST PUNN ORDERING STUDY')
    print(f'Date: {datetime.now().isoformat(timespec="seconds")}    Device: {DEVICE}')
    print(f'Config: {CONFIG}')
    if args.skip_sweep:
        print('MODE: --skip-sweep  (reusing existing main-sweep JSON)')
    if args.skip_controlled:
        print('MODE: --skip-controlled  (main sweep only)')
    print('=' * 72)

    train_ds, test_ds = get_loaders()

    if args.skip_sweep:
        if not os.path.exists(OUT_JSON):
            raise SystemExit(
                f'--skip-sweep set but {OUT_JSON} does not exist; '
                f'run once without --skip-sweep first.'
            )
        with open(OUT_JSON) as f:
            summary = json.load(f)
        print(f'\nLoaded existing summary from {OUT_JSON}')
        print(f'  Main sweep had {len(summary["per_permutation"])} permutations, '
              f'mean acc = {summary["analysis"]["order_sensitivity"]["mean"]:.2f}%')
        # Skip straight to controlled move
        _run_controlled_and_save(summary, train_ds, test_ds)
        return

    perms = sample_permutations(CONFIG['M'], K, CONFIG['permutation_master_seed'])
    print(f'\nSampled {len(perms)} permutations (perm 0 is identity).')

    runs = []
    t0 = time.time()
    for i, pi in enumerate(perms):
        seed = CONFIG['seed_base'] + i
        print(f'\n[perm {i:02d}/{len(perms)-1}] pi = {pi.tolist()}  seed={seed}')
        t_i = time.time()
        r = train_one_permutation(pi, train_ds, test_ds, seed)
        r['perm_idx'] = i
        runs.append(r)
        # per-permutation .npz (arrays too big for JSON)
        np.savez_compressed(
            os.path.join(OUT_RAW_DIR, f'perm_{i:02d}.npz'),
            pi=r['pi'], seed=r['seed'], theta=r['theta'], R=r['R'], H=r['H'],
            y_true=r['y_true'], y_pred=r['y_pred'], correct=r['correct'],
            overall_acc=np.float32(r['overall_acc']),
            per_class_acc=r['per_class_acc'], per_class_n=r['per_class_n'],
        )
        print(f'  acc = {r["overall_acc"]:.2f}%  ({time.time() - t_i:.1f}s)')

    print(f'\nAll {len(runs)} permutations done in {time.time() - t0:.1f}s.')

    print('\nAggregating analyses ...')
    analysis = analyze_all(runs)

    print('\n== Order sensitivity ==')
    os_ = analysis['order_sensitivity']
    print(f'  mean = {os_["mean"]:.2f}%   std = {os_["std"]:.2f}%'
          f'   min = {os_["min"]:.2f}%   max = {os_["max"]:.2f}%'
          f'   range = {os_["range"]:.2f}%')

    print('\n== Accuracy vs position (A_r) ==')
    for r_idx, a in enumerate(analysis['accuracy_vs_position']['A_r_mean']):
        print(f'  pos {r_idx}:  A_r = {a:.2f}%  '
              f'(n={analysis["accuracy_vs_position"]["A_r_n"][r_idx]})')

    # JSON-safe: include per-permutation summaries but not the giant arrays
    summary = dict(
        config=CONFIG,
        device=str(DEVICE),
        date=datetime.now().isoformat(),
        permutations=[r['pi'].tolist() for r in runs],
        seeds=[r['seed'] for r in runs],
        per_permutation=[
            dict(
                perm_idx=r['perm_idx'],
                pi=r['pi'].tolist(),
                seed=r['seed'],
                overall_acc=r['overall_acc'],
                per_class_acc=r['per_class_acc'].tolist(),
                per_class_n=r['per_class_n'].tolist(),
            )
            for r in runs
        ],
        analysis=analysis,
    )
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved JSON summary to {OUT_JSON}')
    print(f'Per-permutation raw arrays in {OUT_RAW_DIR}/')

    plot_accuracy_vs_position(analysis, os.path.join(HERE, 'mnist_ordering_study_positional.pdf'))
    plot_remaining_mass(analysis, os.path.join(HERE, 'mnist_ordering_study_remaining_mass.pdf'))
    plot_per_class(analysis, os.path.join(HERE, 'mnist_ordering_study_per_class.pdf'))
    plot_classwise_overlay(analysis, os.path.join(HERE, 'mnist_ordering_study_classwise_overlay.pdf'))
    write_remaining_mass_table(
        analysis,
        os.path.join(HERE, 'mnist_ordering_study_remaining_mass.txt'),
        os.path.join(HERE, 'mnist_ordering_study_remaining_mass.tex'),
    )

    if args.skip_controlled:
        print('\n--skip-controlled set: main sweep done, exiting.')
        return
    _run_controlled_and_save(summary, train_ds, test_ds)


def _run_controlled_and_save(summary, train_ds, test_ds):
    """Run the controlled-move sub-experiment, save results into the existing
    summary JSON in place, and produce the bar plot."""
    print('\n' + '=' * 72)
    print('CONTROLLED-MOVE SUB-EXPERIMENT: same class at early/mid/late positions')
    print('=' * 72)
    controlled = run_controlled_move(train_ds, test_ds)
    summary['controlled_move'] = dict(
        target_classes=CONFIG['controlled_target_classes'],
        positions=CONFIG['controlled_positions'],
        results=controlled,
    )
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    plot_controlled_move(controlled, os.path.join(HERE, 'mnist_ordering_study_controlled_move.pdf'))

    print('\n== Controlled move summary ==')
    print(f'{"class":>7}{"pos":>6}{"acc(class)":>14}{"acc(overall)":>16}')
    for row in controlled:
        print(f'{row["target_class"]:>7}{row["position"]:>6}'
              f'{row["target_class_acc"]:>14.2f}{row["overall_acc"]:>16.2f}')


if __name__ == '__main__':
    main()
