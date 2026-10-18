"""
MNIST Interpretability Trace
============================

Concrete demonstration of PUNN's gate-rejection chain.

Trains a PUNN-Sigma model on MNIST, then picks two test examples:
  (1) A correctly classified "7" with high confidence.
  (2) A misclassified example (true label != predicted label).

For each example, we report:
  - The 9 gate values g_0(x), ..., g_8(x).
  - The 10 partition values h_0(x), ..., h_9(x).
  - The "remaining mass" left at each step of the recursion.

Outputs:
  - mnist_trace_results.json  (raw numbers)
  - mnist_trace.txt           (human-readable trace)
  - mnist_trace.pdf           (figure: digit + gate bar chart, 2x2 grid)

Reuses the PUNN class and training loop from mnist_experiment.py.
"""

import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_experiment import (
    PUNN,
    train_model,
    evaluate,
    DEVICE,
)

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    'seed': 42,
    'input_dim': 784,
    'num_classes': 10,
    'hidden_dim': 128,    # gate hidden dim; ~1.05M params, 97.43% from sweep
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 0.001,
    'target_correct_class': 7,   # the "7" the user requested
    'min_correct_confidence': 0.90,
}

OUT_JSON = '/Users/aldroua/Dropbox/Projects/PUNN PROJECT/ProjectPUNN/Real data experiments/mnist_trace_results.json'
OUT_TXT  = '/Users/aldroua/Dropbox/Projects/PUNN PROJECT/ProjectPUNN/Real data experiments/mnist_trace.txt'
OUT_PDF  = '/Users/aldroua/Dropbox/Projects/PUNN PROJECT/ProjectPUNN/Real data experiments/mnist_trace.pdf'

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])


# =============================================================================
# UTILITIES
# =============================================================================

def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=CONFIG['batch_size'], shuffle=False)
    return train_loader, test_loader, test_data


@torch.no_grad()
def gate_values(model, x_flat):
    """Return gate outputs g_0..g_{K-2} as a 1D numpy array for a single sample."""
    model.eval()
    x = x_flat.unsqueeze(0).to(DEVICE)
    gs = [g(x).item() for g in model.gates]
    return np.array(gs)


def chain_from_gates(gs):
    """Compute h_0..h_{K-1} and the remaining mass at each step from gate values gs."""
    K = len(gs) + 1
    h = np.zeros(K)
    remaining = np.zeros(K + 1)  # remaining[i] = mass left BEFORE gate i
    rem = 1.0
    remaining[0] = rem
    for i, g in enumerate(gs):
        h[i] = rem * g
        rem = rem * (1.0 - g)
        remaining[i + 1] = rem
    h[K - 1] = rem
    return h, remaining


def find_examples(model, test_data):
    """Find a correctly classified '7' with the *lowest* confidence (most
    pedagogically informative gate trace), and the first misclassified example.
    """
    model.eval()
    target_correct_class = CONFIG['target_correct_class']

    correct_candidates = []   # list of (conf, idx, x_img, y, pred, probs)
    wrong_pick = None
    with torch.no_grad():
        for idx in range(len(test_data)):
            x_img, y = test_data[idx]
            x_flat = x_img.view(-1).to(DEVICE)
            probs = model(x_flat.unsqueeze(0)).squeeze(0).cpu().numpy()
            pred = int(np.argmax(probs))
            conf = float(probs[pred])

            if y == target_correct_class and pred == y:
                correct_candidates.append((conf, idx, x_img, int(y), pred, probs))

            if wrong_pick is None and pred != y:
                wrong_pick = (idx, x_img, int(y), pred, conf, probs)

    # Pick the correct '7' with the LOWEST confidence (most interesting trace).
    correct_candidates.sort(key=lambda t: t[0])
    if correct_candidates:
        conf, idx, x_img, y_, pred, probs = correct_candidates[0]
        correct_pick = (idx, x_img, y_, pred, conf, probs)
    else:
        correct_pick = None
    return correct_pick, wrong_pick


# =============================================================================
# TRACE FORMATTING
# =============================================================================

def format_trace(label, idx, true_label, pred, conf, probs, gs, h, remaining):
    lines = []
    lines.append("=" * 72)
    lines.append(f"{label}  (test index = {idx})")
    lines.append(f"  true label   : {true_label}")
    lines.append(f"  predicted    : {pred}    (conf = {conf:.4f})")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  {'i':>2}  {'g_i(x)':>10}  {'remaining':>12}  {'h_i(x)':>10}  action")
    lines.append("  " + "-" * 60)
    for i, g in enumerate(gs):
        accept = h[i] > 0.5
        action = "ACCEPT (predict class i)" if accept else "reject"
        lines.append(f"  {i:>2}  {g:>10.4f}  {remaining[i]:>12.4f}  {h[i]:>10.4f}  {action}")
    lines.append(f"  {len(gs):>2}  {'(leftover)':>10}  {remaining[-1]:>12.4f}  {h[-1]:>10.4f}  "
                 + ("ACCEPT (predict class K-1)" if h[-1] > 0.5 else "leftover"))
    lines.append("")
    lines.append(f"  full softmax-equivalent probs h_0..h_{len(h)-1}:")
    lines.append("    " + "  ".join(f"{p:.3f}" for p in h))
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_trace(correct, wrong, gs_c, h_c, gs_w, h_w, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5),
                             gridspec_kw={'width_ratios': [1, 2.4]})

    def render(ax_img, ax_bar, pick, gs, h, title_prefix):
        idx, x_img, y_true, pred, conf, probs = pick
        # left: digit
        img = x_img.squeeze().numpy()
        ax_img.imshow(img, cmap='gray')
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.set_title(f"{title_prefix}\ntrue={y_true},  pred={pred} ({conf:.2f})",
                         fontsize=11)
        # right: gate values + h_i
        K = len(h)
        x_pos = np.arange(K)
        bars = ax_bar.bar(x_pos, h, color='#3870b8', alpha=0.85,
                          edgecolor='black', linewidth=0.5, label=r'$h_i(x)$')
        # overlay g_i (one fewer than h)
        ax_bar.plot(np.arange(len(gs)), gs, marker='o', color='#d44',
                    linewidth=1.5, label=r'$g_i(x)$')
        ax_bar.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xlabel('class / gate index $i$')
        ax_bar.set_ylabel('value')
        ax_bar.set_ylim(-0.02, 1.05)
        ax_bar.set_title(f'gate trace: $g_i$ (red) and partition $h_i$ (blue)',
                         fontsize=11)
        ax_bar.legend(loc='upper left', fontsize=9, frameon=False)
        # mark predicted class
        ax_bar.bar(pred, h[pred], color='#3870b8', edgecolor='black',
                   linewidth=1.6, hatch='//', alpha=0.95)

    render(axes[0, 0], axes[0, 1], correct, gs_c, h_c,
           "(a) correctly classified")
    render(axes[1, 0], axes[1, 1], wrong, gs_w, h_w,
           "(b) misclassified")

    fig.suptitle('PUNN gate-rejection chain on MNIST test examples',
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved figure to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("MNIST INTERPRETABILITY TRACE")
    print(f"Device: {DEVICE}    Date: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 70)

    train_loader, test_loader, test_data = get_loaders()

    print(f"\nTraining PUNN-Sigma  (gate hidden dim={CONFIG['hidden_dim']}) ...")
    model = PUNN(CONFIG['input_dim'], CONFIG['num_classes'],
                 'sigma', CONFIG['hidden_dim']).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")
    t0 = time.time()
    train_model(model, train_loader, test_loader,
                CONFIG['epochs'], CONFIG['learning_rate'])
    print(f"  training time: {time.time() - t0:.0f}s")

    test_acc = evaluate(model, test_loader)
    print(f"  final test accuracy: {test_acc:.2f}%")

    # ---- find examples ----
    print("\nSearching for examples ...")
    correct_pick, wrong_pick = find_examples(model, test_data)
    if correct_pick is None or wrong_pick is None:
        print("ERROR: failed to find both example types.")
        return

    # ---- compute traces ----
    def trace_one(pick):
        idx, x_img, y_true, pred, conf, probs = pick
        gs = gate_values(model, x_img.view(-1))
        h, remaining = chain_from_gates(gs)
        return gs, h, remaining

    gs_c, h_c, rem_c = trace_one(correct_pick)
    gs_w, h_w, rem_w = trace_one(wrong_pick)

    # ---- text output ----
    txt_correct = format_trace("CORRECTLY CLASSIFIED",
                               correct_pick[0], correct_pick[2], correct_pick[3],
                               correct_pick[4], correct_pick[5],
                               gs_c, h_c, rem_c)
    txt_wrong = format_trace("MISCLASSIFIED",
                             wrong_pick[0], wrong_pick[2], wrong_pick[3],
                             wrong_pick[4], wrong_pick[5],
                             gs_w, h_w, rem_w)
    txt_all = txt_correct + "\n" + txt_wrong
    print("\n" + txt_all)
    with open(OUT_TXT, 'w') as f:
        f.write(txt_all)
    print(f"Saved trace text to {OUT_TXT}")

    # ---- JSON output ----
    def pick_to_dict(pick, gs, h, rem):
        idx, _, y_true, pred, conf, probs = pick
        return {
            'test_index': idx,
            'true_label': y_true,
            'predicted': pred,
            'predicted_confidence': conf,
            'softmax_equivalent_h': h.tolist(),
            'gate_values_g': gs.tolist(),
            'remaining_mass_before_each_gate': rem.tolist(),
        }

    results = {
        'date': datetime.now().isoformat(),
        'config': CONFIG,
        'model': {
            'parameters': n_params,
            'final_test_accuracy': test_acc,
        },
        'correct_example': pick_to_dict(correct_pick, gs_c, h_c, rem_c),
        'wrong_example':   pick_to_dict(wrong_pick,   gs_w, h_w, rem_w),
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON to {OUT_JSON}")

    # ---- figure ----
    plot_trace(correct_pick, wrong_pick, gs_c, h_c, gs_w, h_w, OUT_PDF)


if __name__ == '__main__':
    main()
