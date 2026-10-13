import numpy as np
import pandas as pd


ROTATION_COLUMNS = [f'r{i}{j}' for i in range(3) for j in range(3)]


def rotation_angle_degrees(previous, current):
    """Shortest relative rotation angle for (..., 3, 3) GT matrices."""
    # trace(previous.T @ current), including batched rotation matrices.
    trace = np.sum(np.asarray(previous) * np.asarray(current), axis=(-2, -1))
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


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
            **dict(zip(ROTATION_COLUMNS, T_gt[:3, :3].reshape(-1))),
        ))

    def save(self, path):
        pd.DataFrame(self.rows).to_csv(path, index=False)
