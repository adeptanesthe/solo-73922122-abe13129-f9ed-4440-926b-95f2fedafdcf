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
        if len(set(self.photos)) != len(self.photos):
            raise ValueError("照片编号重复")
        if len(set(self.annotators)) != len(self.annotators):
            raise ValueError("标注员编号重复")
        photo_set, ann_set = set(self.photos), set(self.annotators)
        seen_votes: set[tuple[str, str]] = set()
        for v in self.votes:
            if v.photo not in photo_set:
                raise ValueError(f"投票引用了未知照片: {v.photo}")
            if v.annotator not in ann_set:
                raise ValueError(f"投票引用了未知标注员: {v.annotator}")
            key = (v.photo, v.annotator)
            if key in seen_votes:
                raise ValueError(f"重复投票: 照片 {v.photo} / 标注员 {v.annotator}")
            seen_votes.add(key)
        seen_expert: set[str] = set()
        for e in self.expert_labels:
            if e.photo not in photo_set:
                raise ValueError(f"专家标签引用了未知照片: {e.photo}")
            if e.photo in seen_expert:
                raise ValueError(f"照片 {e.photo} 的专家标签重复")
            seen_expert.add(e.photo)
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
