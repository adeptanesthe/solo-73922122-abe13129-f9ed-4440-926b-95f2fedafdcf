"""Dawid–Skene 核心算法（NumPy 实现）。

约定
----
- 类别：0 = 无坑，1 = 有坑；类别先验固定为各 0.5。
- 混淆矩阵 conf[j][c][k] = P(标注员 j 投票 k | 真实类别 c)，M 步加一平滑。
  - 误标率（false positive rate）= conf[j][0][1]：无坑被投成有坑。
  - 漏标率（false negative rate）= conf[j][1][0]：有坑被投成无坑。
- 专家标签固定：E 步后把专家照片的后验重置为 one-hot，不再更新。
- 缺票不计：只对实际存在的投票做累加，不为缺失投票补默认值。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASS_PRIOR = (0.5, 0.5)  # 类别先验各半
SMOOTHING = 1.0           # 混淆矩阵加一平滑
MAX_ITER = 200
TOL = 1e-9


@dataclass(frozen=True)
class Vote:
    photo_idx: int
    annotator_idx: int
    label: int  # 0 = 无坑，1 = 有坑


def fit_dawid_skene(votes, n_photos, n_annotators, expert_labels,
                    max_iter=MAX_ITER, tol=TOL):
    """EM 拟合 Dawid–Skene 模型。

    参数
    ----
    votes : list[Vote]               组内投票（photo/annotator 为局部下标）
    n_photos, n_annotators : int     组内照片数、标注员数
    expert_labels : dict[int, int]   局部照片下标 -> 0/1，拟合中固定

    返回
    ----
    (confusion, posteriors, n_iter, converged)
    confusion: (n_annotators, 2, 2)；posteriors: (n_photos, 2)
    """
    confusion = np.array([[[0.7, 0.3], [0.3, 0.7]]] * n_annotators, dtype=float)
    log_prior = np.log(np.asarray(CLASS_PRIOR))

    vp = np.fromiter((v.photo_idx for v in votes), dtype=np.int64, count=len(votes))
    va = np.fromiter((v.annotator_idx for v in votes), dtype=np.int64, count=len(votes))
    vl = np.fromiter((v.label for v in votes), dtype=np.int64, count=len(votes))

    posteriors = np.broadcast_to(np.asarray(CLASS_PRIOR, dtype=float),
                                 (n_photos, 2)).copy()
    for i, c in expert_labels.items():
        posteriors[i] = (0.0, 1.0) if c == 1 else (1.0, 0.0)

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        # ---- E 步：更新未确认照片的后验（专家照片保持固定）----
        log_post = np.broadcast_to(log_prior, (n_photos, 2)).copy()
        if len(votes):
            np.add.at(log_post, vp, np.log(confusion[va, :, vl]))
        log_post -= log_post.max(axis=1, keepdims=True)
        np.exp(log_post, out=log_post)
        log_post /= log_post.sum(axis=1, keepdims=True)
        posteriors[:] = log_post
        for i, c in expert_labels.items():
            posteriors[i] = (0.0, 1.0) if c == 1 else (1.0, 0.0)

        # ---- M 步：加一平滑更新混淆矩阵 ----
        new_conf = np.full((n_annotators, 2, 2), SMOOTHING)
        if len(votes):
            weight = posteriors[vp]  # (n_votes, 2)：每票对两个真实类别的权重
            for c in (0, 1):
                np.add.at(new_conf[:, c, :], (va, vl), weight[:, c])
        new_conf /= new_conf.sum(axis=2, keepdims=True)

        delta = float(np.abs(new_conf - confusion).max())
        confusion = new_conf
        if delta < tol:
            converged = True
            break

    return confusion, posteriors, n_iter, converged


def connected_groups(votes, n_photos, n_annotators):
    """并查集：按投票边把相连的照片与标注员分组。

    返回 [{"photos": set, "annotators": set, "votes": list}]，只含至少有一条
    投票边的节点；没有任何投票的照片/标注员不属于任何组。
    """
    parent = list(range(n_photos + n_annotators))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for v in votes:
        ra, rb = find(v.photo_idx), find(n_photos + v.annotator_idx)
        if ra != rb:
            parent[rb] = ra

    groups = {}
    for v in votes:
        root = find(v.photo_idx)
        g = groups.setdefault(root, {"photos": set(), "annotators": set(), "votes": []})
        g["photos"].add(v.photo_idx)
        g["annotators"].add(v.annotator_idx)
        g["votes"].append(v)
    return list(groups.values())


def fit_groups(votes, n_photos, n_annotators, expert_labels):
    """对每个同时含正、负专家标签的连通组单独做 Dawid–Skene 拟合。

    返回 {全局标注员下标: (2,2) 混淆矩阵}；未拟合组里的标注员不出现。
    """
    fits = {}
    for g in connected_groups(votes, n_photos, n_annotators):
        g_expert = {p: c for p, c in expert_labels.items() if p in g["photos"]}
        if set(g_expert.values()) != {0, 1}:
            continue  # 正、负专家标签不齐的组不拟合
        photo_local = {p: i for i, p in enumerate(sorted(g["photos"]))}
        ann_local = {a: i for i, a in enumerate(sorted(g["annotators"]))}
        local_votes = [Vote(photo_local[v.photo_idx], ann_local[v.annotator_idx], v.label)
                       for v in g["votes"]]
        local_expert = {photo_local[p]: c for p, c in g_expert.items()}
        confusion, _, _, _ = fit_dawid_skene(
            local_votes, len(photo_local), len(ann_local), local_expert)
        for a, la in ann_local.items():
            fits[a] = confusion[la]
    return fits


def evaluate_photo(target_idx, votes, n_photos, n_annotators, expert_labels):
    """留一法评估一张照片：先从拟合数据中移除该照片及其投票，再分组拟合。

    返回 (该照片的全部投票, 可用投票, fits)。可用投票 = 由已拟合组中的
    标注员投出的票；无票可用时调用方应标为“待人工复核”。
    """
    remaining = [v for v in votes if v.photo_idx != target_idx]
    fits = fit_groups(remaining, n_photos, n_annotators, expert_labels)
    target_votes = [v for v in votes if v.photo_idx == target_idx]
    usable = [v for v in target_votes if v.annotator_idx in fits]
    return target_votes, usable, fits


def fit_groups_excluding_segments(exclude_photos, votes, n_photos,
                                  n_annotators, expert_labels):
    """按路段留一：剔除给定照片（整个路段）及其投票后再分组拟合。

    用于换版评估——估计某张照片的有坑概率时，Dawid–Skene 拟合只能使用
    其他路段的照片、投票与专家标签，避免同路段信息泄漏。下标仍沿用全局
    下标，调用方无需重编号。返回 {全局标注员下标: (2,2) 混淆矩阵}。
    """
    exclude = set(exclude_photos)
    remaining = [v for v in votes if v.photo_idx not in exclude]
    remaining_expert = {p: c for p, c in expert_labels.items()
                        if p not in exclude}
    return fit_groups(remaining, n_photos, n_annotators, remaining_expert)


def pothole_probability(usable_votes, fits):
    """用可用投票计算有坑概率（类别先验各半，混淆矩阵来自留一拟合）。"""
    log0 = float(np.log(CLASS_PRIOR[0]))
    log1 = float(np.log(CLASS_PRIOR[1]))
    for v in usable_votes:
        conf = fits[v.annotator_idx]
        log0 += float(np.log(conf[0, v.label]))
        log1 += float(np.log(conf[1, v.label]))
    m = max(log0, log1)
    e0, e1 = np.exp(log0 - m), np.exp(log1 - m)
    return float(e1 / (e0 + e1))
