#!/usr/bin/env python3
"""演示请求：读取 demo 目录的 payload，依次调用三个接口。

- demo/payload.json            → /review 与 /annotator-stats
- demo/version_eval_payload.json → /version-eval

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


def show_review(review: dict) -> None:
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


def show_annotator_stats(stats: dict) -> None:
    print("\n== 标注员汇总（/annotator-stats）==")
    for s in stats["annotators"]:
        if s["status"] == "fitted":
            print(f"{s['annotator']}: 误标率 {s['false_positive_rate']:.3f}，"
                  f"漏标率 {s['false_negative_rate']:.3f}（{s['n_votes']} 票）")
        else:
            print(f"{s['annotator']}: 未拟合（{s['note']}；{s['n_votes']} 票）")


def show_version_eval(ver: dict) -> None:
    print("\n== 换版评估（/version-eval）==")
    print(f"路段数 {ver['n_segments']}，专家核验 {ver['n_expert_audited']} 张")
    for r in ver["results"]:
        fallback = "（无票→0.5）" if r["probability_fallback"] else ""
        print(
            f"{r['photo']}[{r['segment']}] 旧{r['old_prediction']}→新{r['new_prediction']} "
            f"π={r['inclusion_probability']:<4} 有坑概率 {r['pothole_probability']:.3f}{fallback} "
            f"预期差 {r['expected_diff']:+.3f}", end="")
        if r["expert_audited"]:
            print(f" 实际差 {r['actual_diff']:+d} 加权原值 {r['weighted_diff']:+.3f}", end="")
        print()
    if ver["comparable"]:
        print(f"全批平均判对差（新版减旧版，未截断）：{ver['mean_diff']:+.6f}")
    else:
        print(f"无法比较：{ver['message']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    here = Path(__file__).parent
    payload = json.loads((here / "payload.json").read_text(encoding="utf-8"))
    show_review(post(args.base_url, "/review", payload))
    show_annotator_stats(post(args.base_url, "/annotator-stats", payload))

    version_payload = json.loads(
        (here / "version_eval_payload.json").read_text(encoding="utf-8"))
    show_version_eval(post(args.base_url, "/version-eval", version_payload))


if __name__ == "__main__":
    main()
