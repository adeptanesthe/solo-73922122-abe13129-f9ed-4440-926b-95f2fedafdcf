"""坑槽照片标注复核 API。

质检员汇总多人标注后调用：
- POST /review           留一法逐张复核未确认照片，给出有坑概率或“待人工复核”。
- POST /annotator-stats  在完整数据上分组拟合，汇总每人的误标率/漏标率。
- GET  /health           健康检查。
"""
from __future__ import annotations

from fastapi import FastAPI

from . import ds
from .schemas import (AnnotatorStat, AnnotatorStatsResponse, ConfirmedPhoto,
                      PhotoResult, ReviewRequest, ReviewResponse, VoteUsed)

app = FastAPI(
    title="坑槽照片标注复核 API",
    version="1.0.0",
    description=(
        "Dawid–Skene 复核：类别先验各半、混淆矩阵加一平滑、专家标签固定、缺票不计。"
        "评估每张未确认照片时先移除该照片及其投票，按剩余投票把相连的照片和标注员分组，"
        "仅拟合同时含正、负专家标签的组，并只用这些组中标注员的投票计算有坑概率。"
    ),
)


def _prepare(req: ReviewRequest):
    """把字符串编号映射为下标，构造投票与专家标签。"""
    photo_idx = {p: i for i, p in enumerate(req.photos)}
    ann_idx = {a: i for i, a in enumerate(req.annotators)}
    votes = [ds.Vote(photo_idx[v.photo], ann_idx[v.annotator], v.label)
             for v in req.votes]
    expert = {photo_idx[e.photo]: e.label for e in req.expert_labels}
    return photo_idx, ann_idx, votes, expert


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest):
    photo_idx, ann_idx, votes, expert = _prepare(req)
    idx_photo = {i: p for p, i in photo_idx.items()}
    idx_ann = {i: a for a, i in ann_idx.items()}
    n_photos, n_ann = len(req.photos), len(req.annotators)

    results: list[PhotoResult] = []
    for photo in req.photos:
        i = photo_idx[photo]
        if i in expert:
            continue  # 专家已确认的照片无需复核
        target_votes, usable, fits = ds.evaluate_photo(
            i, votes, n_photos, n_ann, expert)
        if not usable:
            note = ("无票可用：该照片没有任何投票" if not target_votes
                    else "无票可用：投票的标注员均不在已拟合的组中")
            results.append(PhotoResult(
                photo=photo, status="pending_manual_review",
                votes_ignored=len(target_votes), note=note))
            continue
        prob = ds.pothole_probability(usable, fits)
        used = [VoteUsed(
            annotator=idx_ann[v.annotator_idx],
            label=v.label,
            false_positive_rate=round(float(fits[v.annotator_idx][0, 1]), 6),
            false_negative_rate=round(float(fits[v.annotator_idx][1, 0]), 6),
        ) for v in usable]
        results.append(PhotoResult(
            photo=photo, status="scored",
            pothole_probability=round(prob, 6),
            votes_used=used,
            votes_ignored=len(target_votes) - len(usable)))

    return ReviewResponse(
        n_photos=n_photos,
        n_annotators=n_ann,
        confirmed=[ConfirmedPhoto(photo=idx_photo[i], expert_label=c)
                   for i, c in sorted(expert.items())],
        results=results,
    )


@app.post("/annotator-stats", response_model=AnnotatorStatsResponse)
def annotator_stats(req: ReviewRequest):
    """在完整数据上分组拟合，汇总每人的误标率/漏标率（供质检员汇总）。"""
    _, ann_idx, votes, expert = _prepare(req)
    fits = ds.fit_groups(votes, len(req.photos), len(req.annotators), expert)
    n_votes = [0] * len(req.annotators)
    for v in votes:
        n_votes[v.annotator_idx] += 1

    rows: list[AnnotatorStat] = []
    for a, name in enumerate(req.annotators):
        if a in fits:
            conf = fits[a]
            rows.append(AnnotatorStat(
                annotator=name, status="fitted", n_votes=n_votes[a],
                false_positive_rate=round(float(conf[0, 1]), 6),
                false_negative_rate=round(float(conf[1, 0]), 6)))
        else:
            rows.append(AnnotatorStat(
                annotator=name, status="not_fitted", n_votes=n_votes[a],
                note="所在连通组缺少正或负专家标签（或无任何投票），未参与拟合"))
    return AnnotatorStatsResponse(annotators=rows)
