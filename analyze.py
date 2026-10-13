import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def auc(rel, max_frac=0.1, n=1000):
    rel = np.asarray(rel, dtype=np.float64)
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return np.nan
    thr = np.linspace(0, max_frac, n)
    acc = (rel[None, :] <= thr[:, None]).mean(axis=1)
    return np.trapz(acc, thr) / max_frac * 100


def add_speed(df):
    """30 fps linear speed (m/s) per frame, from GT translations. Needs stride 1."""
    df = df.sort_values(['obj', 'frame']).copy()
    p = df[['tx', 'ty', 'tz']].to_numpy()
    f = df['frame'].to_numpy()
    o = df['obj'].to_numpy()
    speed = np.full(len(df), np.nan)
    same = (o[1:] == o[:-1]) & (f[1:] - f[:-1] > 0)
    dt = (f[1:] - f[:-1]) / 30.0
    d = np.linalg.norm(p[1:] - p[:-1], axis=1)
    speed[1:][same] = d[same] / dt[same]
    df['lin_speed'] = speed
    return df


def binned_auc(df, xkey, ykey, n_bins=5, min_n=100):
    x = df[xkey].to_numpy()
    rel = (df[ykey] / df['diam']).to_numpy()
    m = np.isfinite(x) & np.isfinite(rel)
    x, rel = x[m], rel[m]
    if x.size == 0:
        return [], [], []
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    cx, cy, cn = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= min_n:
            cx.append(x[sel].mean()); cy.append(auc(rel[sel])); cn.append(int(sel.sum()))
    return cx, cy, cn


def main(csv, fps=30, n_bins=5, out='auc_binned.png'):
    df = pd.read_csv(csv)
    df = add_speed(df)
    if fps == 30:
        df = df[df['sub'] == 0]
    df = df.rename(columns={'tz': 'depth'})

    # sanity check: should match the ALL row in 0_mean_all.xlsx
    for ykey, lab in [('adds', 'ADD-S'), ('add', 'ADD')]:
        print(f'pooled {lab} AUC: {auc(df[ykey] / df["diam"]):.2f}')

    xs = [('lin_speed', 'GT linear speed (m/s)'), ('depth', 'GT depth (m)')]
    objs = sorted(df['obj'].unique())

    fig, axes = plt.subplots(2, len(xs), figsize=(5.5 * len(xs), 7), squeeze=False)
    for j, (xkey, xlabel) in enumerate(xs):
        # row 0: pooled over all frames
        ax = axes[0, j]
        for ykey, lab in [('adds', 'ADD-S'), ('add', 'ADD')]:
            cx, cy, cn = binned_auc(df, xkey, ykey, n_bins)
            ax.plot(cx, cy, 'o-', label=lab)
            if ykey == 'adds':
                for a, b, n in zip(cx, cy, cn):
                    ax.annotate(str(n), (a, b), fontsize=6, xytext=(0, 4),
                                textcoords='offset points')
        ax.set_xlabel(xlabel); ax.set_ylabel('AUC (0-0.1d)'); ax.grid(alpha=0.3)
        ax.set_title('Pooled (labels = frames per bin)')
        if j == 0:
            ax.legend()

        # row 1: ADD-S per object, each with its own bins
        ax = axes[1, j]
        for o in objs:
            cx, cy, _ = binned_auc(df[df['obj'] == o], xkey, 'adds', n_bins, min_n=50)
            ax.plot(cx, cy, 'o-', ms=3, lw=1, label=o)
        ax.set_xlabel(xlabel); ax.set_ylabel('ADD-S AUC'); ax.grid(alpha=0.3)
        ax.set_title('Per sequence')
        if j == 0 and len(objs) <= 12:
            ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(out, dpi=200)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)