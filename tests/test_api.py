"""API 端到端测试。"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def base_payload():
    return {
        "photos": ["p1", "p2", "p3", "p4", "p5", "p10"],
        "annotators": ["a1", "a2", "a5"],
        "votes": [
            {"photo": "p1", "annotator": "a1", "label": 1},
            {"photo": "p2", "annotator": "a1", "label": 0},
            {"photo": "p3", "annotator": "a1", "label": 1},
            {"photo": "p4", "annotator": "a1", "label": 0},
            {"photo": "p5", "annotator": "a1", "label": 1},
            {"photo": "p1", "annotator": "a2", "label": 1},
            {"photo": "p2", "annotator": "a2", "label": 0},
            {"photo": "p3", "annotator": "a2", "label": 1},
            {"photo": "p4", "annotator": "a2", "label": 0},
            {"photo": "p5", "annotator": "a2", "label": 1},
            {"photo": "p10", "annotator": "a5", "label": 1},  # a5 只投这一票
        ],
        "expert_labels": [
            {"photo": "p1", "label": 1},
            {"photo": "p2", "label": 0},
        ],
    }


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_review_happy_path():
    resp = client.post("/review", json=base_payload())
    assert resp.status_code == 200
    body = resp.json()
    # 专家已确认的照片不出现在复核结果里
    assert {c["photo"] for c in body["confirmed"]} == {"p1", "p2"}
    results = {r["photo"]: r for r in body["results"]}
    assert set(results) == {"p3", "p4", "p5", "p10"}

    # p5：两名已拟合标注员都投“有坑” → 有评分且概率 > 0.5
    assert results["p5"]["status"] == "scored"
    assert 0.5 < results["p5"]["pothole_probability"] <= 1.0
    assert {v["annotator"] for v in results["p5"]["votes_used"]} == {"a1", "a2"}
    for v in results["p5"]["votes_used"]:
        assert 0.0 <= v["false_positive_rate"] <= 1.0
        assert 0.0 <= v["false_negative_rate"] <= 1.0

    # p4：两票都投“无坑” → 概率 < 0.5
    assert results["p4"]["status"] == "scored"
    assert 0.0 <= results["p4"]["pothole_probability"] < 0.5

    # p10：唯一投票人 a5 留一后不在任何组 → 待人工复核
    assert results["p10"]["status"] == "pending_manual_review"
    assert results["p10"]["pothole_probability"] is None
    assert results["p10"]["votes_ignored"] == 1


def test_review_photo_without_any_vote_is_pending():
    payload = base_payload()
    payload["photos"].append("p99")  # 无人投票
    resp = client.post("/review", json=payload)
    assert resp.status_code == 200
    results = {r["photo"]: r for r in resp.json()["results"]}
    assert results["p99"]["status"] == "pending_manual_review"
    assert results["p99"]["votes_ignored"] == 0


def test_limits_photos():
    payload = base_payload()
    payload["photos"] = [f"p{i}" for i in range(81)]  # 超过 80 张
    resp = client.post("/review", json=payload)
    assert resp.status_code == 422


def test_limits_annotators():
    payload = base_payload()
    payload["annotators"] = [f"a{i}" for i in range(13)]  # 超过 12 人
    resp = client.post("/review", json=payload)
    assert resp.status_code == 422


@pytest.mark.parametrize("mutate", [
    lambda p: p["votes"].append({"photo": "ghost", "annotator": "a1", "label": 1}),
    lambda p: p["votes"].append({"photo": "p1", "annotator": "ghost", "label": 1}),
    lambda p: p["votes"].append({"photo": "p1", "annotator": "a1", "label": 0}),
    lambda p: p["expert_labels"].append({"photo": "ghost", "label": 1}),
    lambda p: p["expert_labels"].append({"photo": "p1", "label": 0}),
    lambda p: p["photos"].append("p1"),
])
def test_invalid_requests_422(mutate):
    payload = base_payload()
    mutate(payload)
    resp = client.post("/review", json=payload)
    assert resp.status_code == 422


def test_annotator_stats():
    resp = client.post("/annotator-stats", json=base_payload())
    assert resp.status_code == 200
    stats = {s["annotator"]: s for s in resp.json()["annotators"]}
    assert set(stats) == {"a1", "a2", "a5"}
    assert stats["a1"]["status"] == "fitted"
    assert stats["a2"]["status"] == "fitted"
    # a1、a2 在专家照片上全投对，误标/漏标率应较低
    assert stats["a1"]["false_positive_rate"] < 0.5
    assert stats["a1"]["false_negative_rate"] < 0.5
    # a5 所在组没有专家标签 → 未拟合
    assert stats["a5"]["status"] == "not_fitted"
    assert stats["a5"]["false_positive_rate"] is None
