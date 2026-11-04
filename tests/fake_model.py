"""A deterministic stand-in for nnU-Net networks, used to test the pipeline logic.

The "network" output depends on the voxel intensities, the voxel position in the
(z, y, x) array nnU-Net sees, the spacing and the task id, so any difference in how
the input is cropped, reoriented, resampled or split shows up in the result.
"""

from __future__ import annotations

import numpy as np

from boa_light.class_maps import class_map, class_map_5_parts, map_taskid_to_partname_ct

N_LABELS = {
    298: len(class_map["total"]),
    297: len(class_map["total"]),
    258: 2,
    150: 1,
    260: 1,
    315: 3,
    8: 2,
    570: 8,
    542: 11,
    543: 6,
}
for _tid, _part in map_taskid_to_partname_ct.items():
    N_LABELS[_tid] = len(class_map_5_parts[_part])


def _fake_bca(task_id: int, arr: np.ndarray) -> np.ndarray:
    """Anatomically plausible-ish BCA labels so that the aggregation code is exercised."""
    zs, ys, xs = arr.shape
    z, y, x = np.indices(arr.shape)
    body = arr > -300
    lab = np.zeros(arr.shape, np.uint8)
    if task_id == 543:  # body parts: torso, with "arms" at the lateral 10 %
        lab[body] = 1
        lab[body & (x < xs * 0.1)] = 5
        lab[body & (x > xs * 0.9)] = 6
        return lab
    # body regions: abdomen below ~52 %, thorax above ~48 % (overlap band)
    yc, xc = ys / 2, xs / 2
    r = np.sqrt(((y - yc) / ys) ** 2 + ((x - xc) / xs) ** 2)
    lab[body] = 1  # subcutaneous tissue
    lab[body & (r < 0.33)] = 2  # muscle
    inner = r < 0.25
    lab[inner & (z >= zs * 0.48)] = 4  # thoracic cavity
    lab[inner & (z < zs * 0.48)] = 3  # abdominal cavity
    lab[(r < 0.12) & (z >= zs * 0.48) & (z < zs * 0.53)] = 3  # overlap band
    lab[inner & (z >= zs * 0.55) & (z < zs * 0.9) & (r < 0.08)] = 9  # mediastinum
    lab[inner & (z >= zs * 0.6) & (z < zs * 0.7) & (r < 0.05)] = 7  # pericardium
    lab[body & (arr > 250)] = 5  # bone
    return lab


def fake_network(task_id: int, arr_zyx: np.ndarray, spacing_zyx) -> np.ndarray:
    if task_id in (542, 543):
        return _fake_bca(task_id, arr_zyx)
    n = N_LABELS[task_id]
    z, y, x = np.indices(arr_zyx.shape)
    s = int(round(spacing_zyx[0] * 10)) + int(round(spacing_zyx[2] * 3))
    sz, sy, sx = spacing_zyx
    # bands in physical units, so the same structure appears at every resolution
    block = (((z * sz) // 60) * 7 + ((y * sy) // 18) * 3 + ((x * sx) // 24) * 5).astype(np.int64)
    lab = (block + task_id + s) % n + 1
    if 291 <= task_id <= 295:  # each "total" part model only labels its share of the body
        lab[(block // n) % 5 != task_id - 291] = 0
    lab[arr_zyx < -300] = 0
    return lab.astype(np.uint8)


def fake_predict_fn(task_id, img, folds, step_size, return_folds=False):
    """Signature of ``ModelPool.predict``. Per-fold outputs differ from the ensemble."""
    data = np.asanyarray(img.dataobj).astype(np.float32)
    zyx = data.transpose(2, 1, 0)
    spacing = [float(s) for s in img.header.get_zooms()[:3]][::-1]
    seg = fake_network(task_id, zyx, spacing).transpose(2, 1, 0)
    if not return_folds:
        return seg
    fold_ids = list(range(5)) if folds is None else list(folds)
    if len(fold_ids) < 2:
        return seg, None
    n = N_LABELS[task_id]
    per_fold = {}
    for f in fold_ids:
        fs = seg.copy()
        fs[fs > 0] = (fs[fs > 0].astype(np.int64) + f) % n + 1  # deterministic, fold dependent
        per_fold[f] = fs.astype(np.uint8)
    return seg, per_fold
