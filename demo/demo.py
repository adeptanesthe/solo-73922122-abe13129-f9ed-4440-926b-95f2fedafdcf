#!/usr/bin/env python3
"""演示请求：读取同目录 payload.json，依次调用 /review 与 /annotator-stats。

用法：
    python3 demo/demo.py [--base-url http://127.0.0.1:8000]
"""
import argparse
import json
import urllib.request
from pathlib import Path


def post(base_url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    payload = json.loads(
        Path(__file__).with_name("payload.json").read_text(encoding="utf-8"))

    review = post(args.base_url, "/review", payload)
    print("== 复核结果（/review）==")
    confirmed = {c["photo"]: c["expert_label"] for c in review["confirmed"]}
    print("专家已确认：" + ", ".join(
        f"{p}={'有坑' if l else '无坑'}" for p, l in confirmed.items()))
    for r in review["results"]:
        if r["status"] == "scored":
            used = ", ".join(
                f"{v['annotator']} 投{'有坑' if v['label'] else '无坑'}"
                f"(误标 {v['false_positive_rate']:.2f}/漏标 {v['false_negative_rate']:.2f})"
                for v in r["votes_used"])
            print(f"{r['photo']}: 有坑概率 {r['pothole_probability']:.3f}  [{used}]")
        else:
            print(f"{r['photo']}: 待人工复核（{r['note']}）")

    stats = post(args.base_url, "/annotator-stats", payload)
    print("\n== 标注员汇总（/annotator-stats）==")
    for s in stats["annotators"]:
        if s["status"] == "fitted":
            print(f"{s['annotator']}: 误标率 {s['false_positive_rate']:.3f}，"
                  f"漏标率 {s['false_negative_rate']:.3f}（{s['n_votes']} 票）")
        else:
            print(f"{s['annotator']}: 未拟合（{s['note']}；{s['n_votes']} 票）")


if __name__ == "__main__":
    main()
