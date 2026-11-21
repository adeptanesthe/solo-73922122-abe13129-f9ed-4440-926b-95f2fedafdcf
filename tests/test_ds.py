"""Dawid–Skene 核心逻辑测试。"""
import numpy as np
import pytest

from app.ds import (Vote, connected_groups, evaluate_photo, fit_dawid_skene,
                    fit_groups, pothole_probability)


def test_add_one_smoothing_exact():
    # 两张专家照片（p0 无坑、p1 有坑），唯一标注员两票都投“有坑”
    votes = [Vote(0, 0, 1), Vote(1, 0, 1)]
    conf, post, _, _ = fit_dawid_skene(votes, 2, 1, {0: 0, 1: 1})
    # 真实=0 行：观察到 1 次“有坑” → (1+0, 1+1)/3 = (1/3, 2/3)
    assert conf[0, 0] == pytest.approx([1 / 3, 2 / 3])
    # 真实=1 行：观察到 1 次“有坑” → (1/3, 2/3)
    assert conf[0, 1] == pytest.approx([1 / 3, 2 / 3])
    # 专家照片后验保持 one-hot（专家标签固定）
    assert post[0] == pytest.approx([1.0, 0.0])
    assert post[1] == pytest.approx([0.0, 1.0])


def test_missing_votes_not_counted():
    # 标注员 1 没有任何投票 → 混淆矩阵保持平滑先验 0.5/0.5
    votes = [Vote(0, 0, 1), Vote(1, 0, 0)]
    conf, _, _, _ = fit_dawid_skene(votes, 2, 2, {0: 1, 1: 0})
    assert conf[1] == pytest.approx(np.full((2, 2), 0.5))


def test_expert_labels_fixed():
    # 所有投票都说无坑，但专家标签为有坑 → 后验必须保持专家标签
    votes = [Vote(0, 0, 0), Vote(0, 1, 0)]
    _, post, _, _ = fit_dawid_skene(votes, 1, 2, {0: 1})
    assert post[0] == pytest.approx([0.0, 1.0])


def test_recovers_annotator_rates():
    # 仿真：已知三人真实误标/漏标率，检验 EM 能恢复。
    # 核心算法不限照片数（80 张上限是 API 层约束），这里用 300 张
    # 把估计噪声压低，从而可以用较紧的容差断言。
    rng = np.random.default_rng(7)
    true_conf = [
        [[0.9, 0.1], [0.1, 0.9]],   # 误标 0.1，漏标 0.1
        [[0.8, 0.2], [0.3, 0.7]],   # 误标 0.2，漏标 0.3
        [[0.7, 0.3], [0.1, 0.9]],   # 误标 0.3，漏标 0.1
    ]
    n_photos = 300
    truth = np.array([i % 2 for i in range(n_photos)])  # 正负各半
    pos = np.where(truth == 1)[0][:10]
    neg = np.where(truth == 0)[0][:10]
    expert = {int(i): 1 for i in pos} | {int(i): 0 for i in neg}  # 10 正 10 负
    votes = []
    for j, tc in enumerate(true_conf):
        for i in range(n_photos):
            if rng.random() < 0.15:
                continue  # 缺票
            c = truth[i]
            label = 1 if rng.random() < tc[c][1] else 0
            votes.append(Vote(i, j, label))
    conf, _, n_iter, converged = fit_dawid_skene(votes, n_photos, 3, expert)
    assert converged
    for j, tc in enumerate(true_conf):
        assert conf[j, 0, 1] == pytest.approx(tc[0][1], abs=0.1)  # 误标率
        assert conf[j, 1, 0] == pytest.approx(tc[1][0], abs=0.1)  # 漏标率


def test_connected_groups_split():
    # a0、a1 投 p0/p1；a2 只投 p2 → 两个连通组
    votes = [Vote(0, 0, 1), Vote(1, 0, 0), Vote(0, 1, 0), Vote(2, 2, 1)]
    groups = connected_groups(votes, 3, 3)
    sizes = sorted((len(g["photos"]), len(g["annotators"])) for g in groups)
    assert sizes == [(1, 1), (2, 2)]


def test_fit_groups_requires_both_expert_labels():
    votes = [Vote(0, 0, 1), Vote(1, 0, 0), Vote(2, 1, 1)]
    # 组 {p0,p1,a0} 只有正专家标签；组 {p2,a1} 没有专家标签 → 都不拟合
    assert fit_groups(votes, 3, 2, {0: 1}) == {}
    # 组 {p0,p1,a0} 只有负专家标签 → 同样不拟合
    assert fit_groups(votes, 3, 2, {1: 0}) == {}
    # 正、负专家标签齐全 → 拟合该组，a0 得到混淆矩阵
    fits = fit_groups(votes, 3, 2, {0: 1, 1: 0})
    assert set(fits) == {0}
    assert fits[0].shape == (2, 2)


def test_leave_one_out_removes_photo_and_votes():
    # a1 只投了待判照片 p2 → 留一后 a1 无票、不在任何组 → 无票可用
    votes = [Vote(0, 0, 1), Vote(1, 0, 0), Vote(2, 1, 1)]
    target, usable, _ = evaluate_photo(2, votes, 3, 2, {0: 1, 1: 0})
    assert len(target) == 1
    assert usable == []


def test_leave_one_out_excludes_target_votes_from_fit():
    # 待判照片 p2 上 a0 投了“有坑”；若该票进入拟合会抬高 a0 的可用性判断。
    # 构造：p0/p1 为专家照片（一正一负），a0 在 p0、p1 上投对；p2 只有 a0 一票。
    votes = [Vote(0, 0, 1), Vote(1, 0, 0), Vote(2, 0, 1)]
    target, usable, fits = evaluate_photo(2, votes, 3, 1, {0: 1, 1: 0})
    assert len(usable) == 1
    # 拟合只用 p0、p1 的两票：误标率 = 漏标率 = 1/3（加一平滑）
    assert fits[0][0, 1] == pytest.approx(1 / 3)
    assert fits[0][1, 0] == pytest.approx(1 / 3)


def test_pothole_probability_direction_and_value():
    # 两名标注员在 4 张专家照片（两正两负）上全部投对；待判照片两人都投“有坑”
    votes = []
    for j in range(2):
        for i, lab in enumerate([1, 1, 0, 0]):
            votes.append(Vote(i, j, lab))
    votes += [Vote(4, 0, 1), Vote(4, 1, 1)]
    expert = {0: 1, 1: 1, 2: 0, 3: 0}
    _, usable, fits = evaluate_photo(4, votes, 5, 2, expert)
    p = pothole_probability(usable, fits)
    # 每人误标率=漏标率=1/4 → p = (3/4)^2 / ((3/4)^2 + (1/4)^2) = 0.9
    assert p == pytest.approx(0.9, abs=1e-9)

    # 两人都投“无坑”时概率应对称地低
    votes2 = [v for v in votes if v.photo_idx != 4] + [Vote(4, 0, 0), Vote(4, 1, 0)]
    _, usable2, fits2 = evaluate_photo(4, votes2, 5, 2, expert)
    p2 = pothole_probability(usable2, fits2)
    assert p2 == pytest.approx(0.1, abs=1e-9)
