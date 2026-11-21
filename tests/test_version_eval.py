"""换版评估 /version-eval 的端到端测试。"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def expert_base_payload():
    """p1/p2 为专家照片（一正一负），a1、a2 在其上全部投对。

    路段：sE={p1,p2}，其余评估照片各自独立路段，保证留一路段拟合时
    拟合组恰好只含 p1、p2 两票/人 → 误标率=漏标率=1/3（加一平滑）。
    """
    return {
        "photos": ["p1", "p2", "p3", "p4"],
        "annotators": ["a1", "a2"],
        "votes": [
            {"photo": "p1", "annotator": "a1", "label": 1},
            {"photo": "p1", "annotator": "a2", "label": 1},
            {"photo": "p2", "annotator": "a1", "label": 0},
            {"photo": "p2", "annotator": "a2", "label": 0},
            {"photo": "p3", "annotator": "a1", "label": 1},
            {"photo": "p3", "annotator": "a2", "label": 1},
        ],
        "expert_labels": [
            {"photo": "p1", "label": 1},
            {"photo": "p2", "label": 0},
        ],
        "segments": [
            {"photo": "p1", "segment": "sE"},
            {"photo": "p2", "segment": "sE"},
            {"photo": "p3", "segment": "sA"},
            {"photo": "p4", "segment": "sB"},
        ],
    }


def test_version_eval_exact_probability_and_ipw():
    # p3：两人投“有坑”。留 sA 路段拟合只用 p1/p2（各一正一负专家票且都投对），
    # 加一平滑下每人混淆矩阵 [[2/3,1/3],[1/3,2/3]]，两张“有坑”票、先验各半
    # → 有坑概率 0.8。
    # p4：无任何投票 → 概率回退 0.5；专家标签 0，旧版判有坑(错)、新版判无坑(对)
    payload = expert_base_payload()
    # p4 的专家标签不改变 sA 拟合：无票照片不属于任何连通组，不参与拟合
    payload["expert_labels"].append({"photo": "p4", "label": 0})
    payload["eval_photos"] = [
        {"photo": "p3", "old_prediction": 0,
         "new_prediction": 1, "inclusion_probability": 0.5},
        {"photo": "p4", "old_prediction": 1,
         "new_prediction": 0, "inclusion_probability": 0.25},
    ]

    resp = client.post("/version-eval", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["comparable"] is True
    assert body["n_expert_audited"] == 1
    assert body["n_segments"] == 3  # sE、sA、sB

    rows = {r["photo"]: r for r in body["results"]}

    p3 = rows["p3"]
    assert p3["pothole_probability"] == pytest.approx(0.8, abs=1e-6)
    assert p3["probability_fallback"] is False
    assert p3["segment"] == "sA"
    assert {v["annotator"] for v in p3["votes_used"]} == {"a1", "a2"}
    # 预期判对差 = p - (1-p) = 0.6
    assert p3["expected_diff"] == pytest.approx(0.6, abs=1e-6)
    assert p3["expert_audited"] is False
    assert p3["actual_diff"] is None
    # 无专家标签：加权原值即预期差
    assert p3["weighted_diff"] == pytest.approx(0.6, abs=1e-6)

    p4 = rows["p4"]
    assert p4["probability_fallback"] is True
    assert p4["pothole_probability"] == pytest.approx(0.5, abs=1e-9)
    assert p4["votes_used"] == []
    # p=0.5 且两版预测不同 → 预期差为 0
    assert p4["expected_diff"] == pytest.approx(0.0, abs=1e-9)
    # 实际：新版对(1)、旧版错(0) → 实际差 +1
    assert p4["actual_diff"] == 1
    # 加权原值 = 0 + (1-0)/0.25 = 4，超出 +1 也不截断
    assert p4["weighted_diff"] == pytest.approx(4.0, abs=1e-6)

    # 全批平均保留加权原值：(0.6 + 4.0) / 2 = 2.3，超出 ±1 不截断
    assert body["mean_diff"] == pytest.approx(2.3, abs=1e-6)


def test_version_eval_no_expert_labels_not_comparable():
    payload = expert_base_payload()
    payload["expert_labels"] = []
    payload["eval_photos"] = [
        {"photo": "p3", "old_prediction": 0,
         "new_prediction": 1, "inclusion_probability": 0.5},
    ]
    resp = client.post("/version-eval", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["comparable"] is False
    assert body["mean_diff"] is None
    assert body["n_expert_audited"] == 0
    assert "无法比较" in body["message"]
    # 无专家标签 → 没有任何可拟合的组（要求正、负专家标签齐全）→ 回退 0.5
    row = body["results"][0]
    assert row["probability_fallback"] is True
    assert row["pothole_probability"] == pytest.approx(0.5, abs=1e-9)
    # p=0.5、两版预测不同 → 预期差 0；加权原值即预期差
    assert row["expected_diff"] == pytest.approx(0.0, abs=1e-9)
    assert row["weighted_diff"] == pytest.approx(0.0, abs=1e-9)


def test_version_eval_same_segment_photos_excluded_from_fit():
    # 三张照片同属一个路段；留一路段拟合时整段（含专家照片）被剔除
    # → 无拟合组 → 回退 0.5。若错误地只做照片级留一，p3 会得到高概率。
    payload = {
        "photos": ["p1", "p2", "p3"],
        "annotators": ["a1"],
        "votes": [
            {"photo": "p1", "annotator": "a1", "label": 1},
            {"photo": "p2", "annotator": "a1", "label": 0},
            {"photo": "p3", "annotator": "a1", "label": 1},
        ],
        "expert_labels": [
            {"photo": "p1", "label": 1},
            {"photo": "p2", "label": 0},
        ],
        "segments": [
            {"photo": "p1", "segment": "s1"},
            {"photo": "p2", "segment": "s1"},
            {"photo": "p3", "segment": "s1"},
        ],
        "eval_photos": [
            {"photo": "p3", "old_prediction": 0,
             "new_prediction": 1, "inclusion_probability": 0.5},
        ],
    }
    resp = client.post("/version-eval", json=payload)
    assert resp.status_code == 200
    row = resp.json()["results"][0]
    assert row["probability_fallback"] is True
    assert row["pothole_probability"] == pytest.approx(0.5, abs=1e-9)
    assert row["votes_ignored"] == 1


def test_version_eval_identical_predictions_zero_expected_diff():
    payload = expert_base_payload()
    payload["eval_photos"] = [
        {"photo": "p3", "old_prediction": 1,
         "new_prediction": 1, "inclusion_probability": 0.5},
    ]
    resp = client.post("/version-eval", json=payload)
    row = resp.json()["results"][0]
    # 两版预测相同 → 无论概率多少，预期差为 0
    assert row["expected_diff"] == pytest.approx(0.0, abs=1e-12)


def test_version_eval_actual_diff_and_weighted_formula():
    # p3 有专家标签 1、两版都投“有坑” → 实际差 0，加权原值仍为 0
    payload = expert_base_payload()
    payload["expert_labels"].append({"photo": "p3", "label": 1})
    payload["eval_photos"] = [
        {"photo": "p3", "old_prediction": 1,
         "new_prediction": 1, "inclusion_probability": 0.5},
    ]
    resp = client.post("/version-eval", json=payload)
    body = resp.json()
    assert body["comparable"] is True
    row = body["results"][0]
    assert row["expert_audited"] is True
    assert row["actual_diff"] == 0
    assert row["weighted_diff"] == pytest.approx(0.0, abs=1e-12)
    assert body["mean_diff"] == pytest.approx(0.0, abs=1e-12)


def test_version_eval_negative_actual_diff_unclipped():
    # p4 专家标签 0：旧版判无坑(对)、新版判有坑(错) → 实际差 -1
    payload = expert_base_payload()
    payload["expert_labels"].append({"photo": "p4", "label": 0})
    payload["eval_photos"] = [
        {"photo": "p4", "old_prediction": 0,
         "new_prediction": 1, "inclusion_probability": 0.1},
    ]
    resp = client.post("/version-eval", json=payload)
    body = resp.json()
    row = body["results"][0]
    assert row["probability_fallback"] is True
    assert row["actual_diff"] == -1
    # 预期差 0 → 加权原值 = -1/0.1 = -10，超出 -1 不截断
    assert row["weighted_diff"] == pytest.approx(-10.0, abs=1e-9)
    assert body["mean_diff"] == pytest.approx(-10.0, abs=1e-9)


@pytest.mark.parametrize("mutate", [
    lambda p: p["eval_photos"][0].update({"inclusion_probability": 0.005}),
    lambda p: p["eval_photos"][0].update({"inclusion_probability": 1.5}),
    lambda p: p["eval_photos"][0].update({"photo": "ghost"}),
    lambda p: p["eval_photos"].append(dict(p["eval_photos"][0])),
    lambda p: p["votes"].append(
        {"photo": "ghost", "annotator": "a1", "label": 1}),
    lambda p: p["segments"].remove(p["segments"][-1]),       # 缺一张照片的路段
    lambda p: p["segments"].append(dict(p["segments"][0])),  # 路段映射重复
])
def test_version_eval_invalid_requests_422(mutate):
    payload = expert_base_payload()
    payload["eval_photos"] = [
        {"photo": "p3", "old_prediction": 0,
         "new_prediction": 1, "inclusion_probability": 0.5},
    ]
    mutate(payload)
    resp = client.post("/version-eval", json=payload)
    assert resp.status_code == 422


def test_version_eval_limit_photos():
    payload = expert_base_payload()
    payload["photos"] = [f"p{i}" for i in range(81)]
    payload["segments"] = [{"photo": f"p{i}", "segment": "s1"}
                           for i in range(81)]
    payload["eval_photos"] = [
        {"photo": "p3", "old_prediction": 0,
         "new_prediction": 1, "inclusion_probability": 0.5},
    ]
    resp = client.post("/version-eval", json=payload)
    assert resp.status_code == 422


def test_version_eval_segment_fit_is_cached_per_segment():
    # 同一路段两张照片：两次评估共用一次留一路段拟合，概率口径一致。
    payload = {
        "photos": ["p1", "p2", "p3", "p4"],
        "annotators": ["a1", "a2"],
        "votes": [
            {"photo": "p1", "annotator": "a1", "label": 1},
            {"photo": "p1", "annotator": "a2", "label": 1},
            {"photo": "p2", "annotator": "a1", "label": 0},
            {"photo": "p2", "annotator": "a2", "label": 0},
            {"photo": "p3", "annotator": "a1", "label": 1},
            {"photo": "p4", "annotator": "a2", "label": 0},
        ],
        "expert_labels": [
            {"photo": "p1", "label": 1},
            {"photo": "p2", "label": 0},
        ],
        "segments": [
            {"photo": "p1", "segment": "sE"},
            {"photo": "p2", "segment": "sE"},
            {"photo": "p3", "segment": "sA"},
            {"photo": "p4", "segment": "sA"},
        ],
        "eval_photos": [
            {"photo": "p3", "old_prediction": 0,
             "new_prediction": 1, "inclusion_probability": 0.5},
            {"photo": "p4", "old_prediction": 1,
             "new_prediction": 0, "inclusion_probability": 0.5},
        ],
    }
    resp = client.post("/version-eval", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_segments"] == 2  # sE 与评估路段 sA
    rows = {r["photo"]: r for r in body["results"]}
    # 留一路段拟合只含 p1/p2，每人误标率=漏标率=1/3，先验各半：
    # p3 单票“有坑” → 2/3；p4 单票“无坑” → 1/3（输出保留 6 位小数）
    assert rows["p3"]["pothole_probability"] == pytest.approx(2 / 3, abs=1e-6)
    assert rows["p4"]["pothole_probability"] == pytest.approx(1 / 3, abs=1e-6)
