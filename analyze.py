import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from extrametrics import ROTATION_COLUMNS, rotation_angle_degrees


def auc(rel, max_frac=0.1, n=1000):
    rel = np.asarray(rel, dtype=np.float64)
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return np.nan
    thr = np.linspace(0, max_frac, n)
    acc = (rel[None, :] <= thr[:, None]).mean(axis=1)
    return np.trapz(acc, thr) / max_frac * 100


def add_speed(df):
    """GT linear (m/s) and angular (deg/s) speeds over the same intervals.

    RGB frame indices run at 30 Hz; sub=0..3 supplies the 120 Hz offsets.
    Filter to sub=0 before calling this for a 30 fps benchmark. Sequence starts
    and duplicate/non-increasing timestamps have no speed (NaN).
    """
    missing = [column for column in ROTATION_COLUMNS if column not in df]
    if missing:
        raise ValueError(
            'GT rotation columns are missing; regenerate per_frame.csv with '
            'the updated PoseEvalLogger before benchmarking angular velocity.'
        )
    df = df.sort_values(['obj', 'frame', 'sub'], kind='stable').copy()
    p = df[['tx', 'ty', 'tz']].to_numpy(dtype=float)
    rotations = df[ROTATION_COLUMNS].to_numpy(dtype=float).reshape(-1, 3, 3)
    t = df['frame'].to_numpy(dtype=float) / 30.0 + df['sub'].to_numpy(dtype=float) / 120.0
    o = df['obj'].to_numpy()
    dt = np.diff(t)
    same = (o[1:] == o[:-1]) & (dt > 0)
    linear = np.full(len(df), np.nan)
    angular = np.full(len(df), np.nan)
    distance = np.linalg.norm(np.diff(p, axis=0), axis=1)
    angle = rotation_angle_degrees(rotations[:-1], rotations[1:])
    linear[1:][same] = distance[same] / dt[same]
    angular[1:][same] = angle[same] / dt[same]
    df['lin_speed'] = linear
    df['ang_speed'] = angular
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
    if fps not in (30, 120):
        raise ValueError('fps must be 30 or 120')
    if fps == 30:
        df = df[df['sub'] == 0]
    df = add_speed(df)
    diameter = df['diam'].where(np.isfinite(df['diam']) & (df['diam'] > 0))
    df['depth'] = df['tz'] / diameter

    # sanity check: should match the ALL row in 0_mean_all.xlsx
    for ykey, lab in [('adds', 'ADD-S'), ('add', 'ADD')]:
        print(f'pooled {lab} AUC: {auc(df[ykey] / df["diam"]):.2f}')

    xs = [('lin_speed', 'GT linear speed (m/s)'),
          ('ang_speed', 'GT angular speed (deg/s)'),
          ('depth', 'GT depth / object diameter (dimensionless)')]
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
