# seq_auc_plot.py  (run from the repo root so the imports resolve)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.dataloader.my_reader_online_scaled import Event6DReaderScaled

VIDEO_DIRS = './data/Event6D'
XLSX = 'outputs/event6d_eval_30fps/0_eval_metric/0_mean_all.xlsx'
E2VID = False                      # match the flag you evaluated with
NAME_COL = 'Name' # column holding the sequence name
ADDS_COL = 'ADDS'
ADD_COL = 'ADD' # column holding ADD-S AUC
# Set both after the first run prints df.columns


def seq_stats(video_dir):
    r = Event6DReaderScaled(video_dir, ev_width=1280, ev_height=720)
    frame = 'event' if E2VID else 'rgb'
    zs, ps, Rs, tt = [], [], [], []
    for i in range(len(r.color_files)):
        T = r.get_gt_pose(i, 0, frame=frame)      # j=0 -> 30 fps samples
        if T is None:
            continue
        zs.append(T[2, 3]); ps.append(T[:3, 3]); Rs.append(T[:3, :3]); tt.append(i / 30.0)
    ps, tt = np.array(ps), np.array(tt)
    v = np.linalg.norm(np.diff(ps, axis=0), axis=1) / np.diff(tt)
    w = [np.degrees(np.arccos(np.clip((np.trace(Rs[k].T @ Rs[k + 1]) - 1) / 2, -1, 1)))
         / (tt[k + 1] - tt[k]) for k in range(len(Rs) - 1)]
    return dict(depth=np.mean(zs), lin_vel=v.mean(), ang_vel=np.mean(w),
                diam=r.get_gt_mesh_diamter())


with open(os.path.join(VIDEO_DIRS, 'test.txt')) as f:
    seqs = sorted(l.strip().split()[0] for l in f if l.strip())

stats_path = 'seq_stats.csv'
if os.path.exists(stats_path):                     # cached, delete to recompute
    stats_df = pd.read_csv(stats_path, index_col=0)
else:
    stats = {s.replace('/', '-'): seq_stats(os.path.join(VIDEO_DIRS, s)) for s in seqs}
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(stats_path)

df = pd.read_excel(XLSX, sheet_name='Summary_Metrics')
print(df.columns.tolist())
assert NAME_COL and ADDS_COL, 'Set NAME_COL and ADDS_COL from the printed columns'

df = df[df[NAME_COL].isin(stats_df.index)].copy()  # drops the ALL row
for k in ['depth', 'lin_vel', 'ang_vel', 'diam']:
    df[k] = df[NAME_COL].map(stats_df[k])
df['obj'] = df[NAME_COL].str.split('_').str[0]     # e.g. banana_1101-0003 -> banana

panels = [('lin_vel', 'Mean GT linear velocity (m/s)'),
          ('depth', 'Mean GT depth (m)'),
          ('diam', 'Object diameter')]
cmap = plt.get_cmap('tab10')
objs = sorted(df['obj'].unique())

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (k, lab) in zip(axes, panels):
    for n, o in enumerate(objs):
        d = df[df['obj'] == o]
        ax.scatter(d[k], d[ADD_COL], color=cmap(n % 10), label=o, s=30)
    ax.set_xlabel(lab); ax.set_ylabel('ADD AUC'); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig('add_auc_vs_seq.png', dpi=200)