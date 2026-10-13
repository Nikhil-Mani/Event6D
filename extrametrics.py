import numpy as np
import pandas as pd


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

class PoseEvalLogger:
    def __init__(self):
        self.rows = []

    def reset_sequence(self):
        pass  # kept so existing calls still work

    def update(self, metric, obj, frame, sub, diam):
        T_gt = np.asarray(metric['gt_pose'])
        self.rows.append(dict(
            obj=obj, frame=frame, sub=sub, diam=diam,
            adds=metric['adds'], add=metric['add'],
            mssd=metric['mssd'], mspd=metric['mspd'],
            tx=T_gt[0, 3], ty=T_gt[1, 3], tz=T_gt[2, 3],
        ))

    def save(self, path):
        pd.DataFrame(self.rows).to_csv(path, index=False)