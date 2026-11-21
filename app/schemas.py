"""请求/响应模型与入参校验。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

MAX_PHOTOS = 80       # 最多 80 张照片
MAX_ANNOTATORS = 12   # 最多 12 名标注员

Label = Literal[0, 1]  # 0 = 无坑，1 = 有坑


class VoteIn(BaseModel):
    photo: str = Field(description="照片编号")
    annotator: str = Field(description="标注员编号")
    label: Label = Field(description="投票：1=有坑，0=无坑")


class ExpertLabelIn(BaseModel):
    photo: str = Field(description="照片编号")
    label: Label = Field(description="专家标签：1=有坑，0=无坑")


def _check_references(photos, annotators, votes, expert_labels):
    """共用的引用/重复校验，供两个请求模型调用。"""
    if len(set(photos)) != len(photos):
        raise ValueError("照片编号重复")
    if len(set(annotators)) != len(annotators):
        raise ValueError("标注员编号重复")
    photo_set, ann_set = set(photos), set(annotators)
    seen_votes: set[tuple[str, str]] = set()
    for v in votes:
        if v.photo not in photo_set:
            raise ValueError(f"投票引用了未知照片: {v.photo}")
        if v.annotator not in ann_set:
            raise ValueError(f"投票引用了未知标注员: {v.annotator}")
        key = (v.photo, v.annotator)
        if key in seen_votes:
            raise ValueError(f"重复投票: 照片 {v.photo} / 标注员 {v.annotator}")
        seen_votes.add(key)
    seen_expert: set[str] = set()
    for e in expert_labels:
        if e.photo not in photo_set:
            raise ValueError(f"专家标签引用了未知照片: {e.photo}")
        if e.photo in seen_expert:
            raise ValueError(f"照片 {e.photo} 的专家标签重复")
        seen_expert.add(e.photo)


class ReviewRequest(BaseModel):
    photos: list[str] = Field(min_length=1, max_length=MAX_PHOTOS,
                              description="照片编号列表，最多 80 个")
    annotators: list[str] = Field(min_length=1, max_length=MAX_ANNOTATORS,
                                  description="标注员编号列表，最多 12 个")
    votes: list[VoteIn] = Field(default_factory=list,
                                description="投票记录；缺票即不出现，不计入")
    expert_labels: list[ExpertLabelIn] = Field(default_factory=list,
                                               description="部分照片的专家标签，拟合中固定")

    @model_validator(mode="after")
    def check_references(self):
        _check_references(self.photos, self.annotators,
                          self.votes, self.expert_labels)
        return self


class PhotoSegmentIn(BaseModel):
    photo: str = Field(description="照片编号，须出现在 photos 中")
    segment: str = Field(description="路段编号；拟合该照片概率时剔除同路段全部照片")


class VersionEvalPhotoIn(BaseModel):
    photo: str = Field(description="照片编号，须出现在 photos 中")
    old_prediction: Label = Field(description="旧版检测器预测：1=有坑，0=无坑")
    new_prediction: Label = Field(description="新版检测器预测：1=有坑，0=无坑")
    inclusion_probability: float = Field(
        ge=0.01, le=1.0,
        description="入选专家抽查的概率 π，取值 0.01～1")


class VersionEvalRequest(BaseModel):
    photos: list[str] = Field(min_length=1, max_length=MAX_PHOTOS,
                              description="照片编号列表，最多 80 个")
    annotators: list[str] = Field(min_length=1, max_length=MAX_ANNOTATORS,
                                  description="标注员编号列表，最多 12 个")
    votes: list[VoteIn] = Field(default_factory=list,
                                description="投票记录；缺票即不出现，不计入")
    expert_labels: list[ExpertLabelIn] = Field(default_factory=list,
                                               description="专家抽查标签，拟合中固定")
    segments: list[PhotoSegmentIn] = Field(
        description="每张照片的路段编号，须与 photos 一一对应、不可重复或遗漏")
    eval_photos: list[VersionEvalPhotoIn] = Field(
        min_length=1,
        description="每张照片的两版预测与入选概率 π")

    @model_validator(mode="after")
    def check_references(self):
        _check_references(self.photos, self.annotators,
                          self.votes, self.expert_labels)
        photo_set = set(self.photos)
        seen_seg: set[str] = set()
        for s in self.segments:
            if s.photo not in photo_set:
                raise ValueError(f"路段映射引用了未知照片: {s.photo}")
            if s.photo in seen_seg:
                raise ValueError(f"照片 {s.photo} 的路段映射重复")
            seen_seg.add(s.photo)
        if seen_seg != photo_set:
            missing = sorted(photo_set - seen_seg)
            raise ValueError(f"以下照片缺少路段编号: {', '.join(missing)}")
        seen_eval: set[str] = set()
        for e in self.eval_photos:
            if e.photo not in photo_set:
                raise ValueError(f"换版评估引用了未知照片: {e.photo}")
            if e.photo in seen_eval:
                raise ValueError(f"照片 {e.photo} 的换版评估记录重复")
            seen_eval.add(e.photo)
        return self


class VoteUsed(BaseModel):
    annotator: str
    label: int
    false_positive_rate: float = Field(description="误标率 P(投有坑|真无坑)")
    false_negative_rate: float = Field(description="漏标率 P(投无坑|真有坑)")


class PhotoResult(BaseModel):
    photo: str
    status: Literal["scored", "pending_manual_review"]
    pothole_probability: float | None = Field(
        default=None, description="有坑概率；待人工复核时为 null")
    votes_used: list[VoteUsed] = Field(default_factory=list)
    votes_ignored: int = Field(default=0, description="因标注员不在已拟合组而未采用的票数")
    note: str = ""


class ConfirmedPhoto(BaseModel):
    photo: str
    expert_label: int


class ReviewResponse(BaseModel):
    n_photos: int
    n_annotators: int
    confirmed: list[ConfirmedPhoto] = Field(description="专家已确认的照片，不参与复核")
    results: list[PhotoResult] = Field(description="每张未确认照片的复核结论")


class AnnotatorStat(BaseModel):
    annotator: str
    status: Literal["fitted", "not_fitted"]
    n_votes: int
    false_positive_rate: float | None = Field(default=None, description="误标率")
    false_negative_rate: float | None = Field(default=None, description="漏标率")
    note: str | None = None


class AnnotatorStatsResponse(BaseModel):
    annotators: list[AnnotatorStat]


class VersionEvalPhotoResult(BaseModel):
    photo: str
    segment: str
    old_prediction: int = Field(description="旧版预测：1=有坑，0=无坑")
    new_prediction: int = Field(description="新版预测：1=有坑，0=无坑")
    inclusion_probability: float = Field(description="入选专家抽查的概率 π")
    pothole_probability: float = Field(
        description="有坑概率：按路段留一的 Dawid–Skene 拟合与本照片可用票计算；"
                    "无票可用时按 0.5")
    votes_used: list[VoteUsed] = Field(
        default_factory=list, description="计算概率实际采用的票；无票可用时为空")
    votes_ignored: int = Field(default=0, description="因标注员不在已拟合组而未采用的票数")
    probability_fallback: bool = Field(
        description="True 表示无票可用，有坑概率按 0.5 处理")
    expected_diff: float = Field(
        description="预期判对差（新版减旧版），基于有坑概率，范围 ±1 内")
    actual_diff: int | None = Field(
        default=None, description="实际判对差（新版减旧版）；无专家标签时为 null")
    expert_audited: bool = Field(description="该照片是否有专家抽查标签")
    weighted_diff: float = Field(
        description="该照片计入平均值的加权原值：无专家标签即预期差；"
                    "有专家标签为预期差 + (实际差-预期差)/π，不做截断，可能超出 ±1")


class VersionEvalResponse(BaseModel):
    comparable: bool = Field(description="整批是否含专家核验；False 时无法比较")
    message: str = Field(description="comparable=False 时的提示语")
    n_photos: int
    n_expert_audited: int = Field(description="整批带专家标签的照片数")
    n_segments: int = Field(description="路段编号去重后的数量")
    mean_diff: float | None = Field(
        default=None,
        description="全批平均判对差（新版减旧版），保留加权原值不截断；"
                    "无专家核验时为 null")
    results: list[VersionEvalPhotoResult] = Field(
        description="每张照片的概率、预期差、实际差与加权原值")
