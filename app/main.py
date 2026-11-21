"""坑槽照片标注复核与换版评估 API。

质检员汇总多人标注后调用：
- POST /review           留一法逐张复核未确认照片，给出有坑概率或“待人工复核”。
- POST /annotator-stats  在完整数据上分组拟合，汇总每人的误标率/漏标率。
- POST /version-eval     换版评估：按路段留一估计有坑概率，结合两版预测与
                         专家抽查概率 π，估计全批平均判对差（新版减旧版）。
- GET  /health           健康检查。
"""
from __future__ import annotations

from fastapi import FastAPI

from . import ds
from .schemas import (AnnotatorStat, AnnotatorStatsResponse, ConfirmedPhoto,
                      PhotoResult, ReviewRequest, ReviewResponse,
                      VersionEvalPhotoResult, VersionEvalRequest,
                      VersionEvalResponse, VoteUsed)

app = FastAPI(
    title="坑槽照片标注复核 API",
    version="1.1.0",
    description=(
        "Dawid–Skene 复核：类别先验各半、混淆矩阵加一平滑、专家标签固定、缺票不计。"
        "评估每张未确认照片时先移除该照片及其投票，按剩余投票把相连的照片和标注员分组，"
        "仅拟合同时含正、负专家标签的组，并只用这些组中标注员的投票计算有坑概率。"
        "换版评估（/version-eval）改为按路段留一：拟合只使用其他路段的照片、投票与专家标签。"
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


def _expected_correct_diff(p, old_pred, new_pred):
    """预期判对差（新版减旧版）。

    单版预期判对率：预测有坑时为 p，预测无坑时为 1-p。
    """
    old_rate = p if old_pred == 1 else 1.0 - p
    new_rate = p if new_pred == 1 else 1.0 - p
    return new_rate - old_rate


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


@app.post("/version-eval", response_model=VersionEvalResponse)
def version_eval(req: VersionEvalRequest):
    """换版评估：估计新版相对旧版的全批平均判对差。

    流程（逐张照片）：
    1. 按路段留一：Dawid–Skene 拟合只用其他路段的照片、投票与专家标签；
    2. 用本张照片的可用票计算有坑概率，无票可用时按 0.5；
    3. 由有坑概率与两版预测求预期判对差（新版减旧版）；
    4. 有专家标签的照片，再把 (实际差-预期差)/π 加回预期差（逆概率加权）；
    5. 全批平均，保留加权原值不截断（可能超出 ±1）。
    整批没有专家核验时返回 comparable=False，不给出平均差。
    """
    photo_idx, ann_idx, votes, expert = _prepare(req)
    idx_ann = {i: a for a, i in ann_idx.items()}
    n_photos, n_ann = len(req.photos), len(req.annotators)

    # 每张照片的路段；路段 -> 该路段全部照片下标（同路段共用一次留一路段拟合）
    photo_segment: dict[int, str] = {}
    segment_photos: dict[str, list[int]] = {}
    for s in req.segments:
        i = photo_idx[s.photo]
        photo_segment[i] = s.segment
        segment_photos.setdefault(s.segment, []).append(i)
    segment_fits: dict[str, dict] = {}

    results: list[VersionEvalPhotoResult] = []
    for item in req.eval_photos:
        i = photo_idx[item.photo]
        segment = photo_segment[i]
        if segment not in segment_fits:
            segment_fits[segment] = ds.fit_groups_excluding_segments(
                segment_photos[segment], votes, n_photos, n_ann, expert)
        fits = segment_fits[segment]

        target_votes = [v for v in votes if v.photo_idx == i]
        usable = [v for v in target_votes if v.annotator_idx in fits]
        fallback = not usable
        if fallback:
            prob = 0.5
        else:
            prob = ds.pothole_probability(usable, fits)

        used = [VoteUsed(
            annotator=idx_ann[v.annotator_idx],
            label=v.label,
            false_positive_rate=round(float(fits[v.annotator_idx][0, 1]), 6),
            false_negative_rate=round(float(fits[v.annotator_idx][1, 0]), 6),
        ) for v in usable]

        expected = _expected_correct_diff(
            prob, item.old_prediction, item.new_prediction)

        actual = None
        weighted = expected
        if i in expert:
            truth = expert[i]
            actual = int(item.new_prediction == truth) - \
                int(item.old_prediction == truth)
            # 逆概率加权：预期差 + (实际差 - 预期差) / π
            weighted = expected + (actual - expected) / \
                item.inclusion_probability

        results.append(VersionEvalPhotoResult(
            photo=item.photo,
            segment=segment,
            old_prediction=item.old_prediction,
            new_prediction=item.new_prediction,
            inclusion_probability=item.inclusion_probability,
            pothole_probability=round(prob, 6),
            votes_used=used,
            votes_ignored=len(target_votes) - len(usable),
            probability_fallback=fallback,
            expected_diff=round(expected, 6),
            actual_diff=actual,
            expert_audited=i in expert,
            weighted_diff=round(weighted, 6),
        ))

    n_audited = sum(1 for r in results if r.expert_audited)
    if n_audited == 0:
        return VersionEvalResponse(
            comparable=False,
            message="整批没有专家核验照片，无法比较两版检测器，"
                    "仅返回逐张预期判对差；mean_diff 为 null。",
            n_photos=len(results),
            n_expert_audited=0,
            n_segments=len(segment_photos),
            mean_diff=None,
            results=results,
        )

    mean_diff = sum(r.weighted_diff for r in results) / len(results)
    return VersionEvalResponse(
        comparable=True,
        message="OK",
        n_photos=len(results),
        n_expert_audited=n_audited,
        n_segments=len(segment_photos),
        mean_diff=round(mean_diff, 6),
        results=results,
    )
