# 坑槽照片标注复核 API

质检员汇总多人标注后，用本服务复核未确认照片。基于 Dawid–Skene 模型
（FastAPI + NumPy）估计每名标注员的误标率/漏标率，并对每张未确认照片
给出有坑概率；无票可用时标为“待人工复核”。

## 方法

**模型假设**

- 类别：0 = 无坑，1 = 有坑；类别先验固定为各 0.5。
- 每名标注员一个 2×2 混淆矩阵 `conf[c][k] = P(投票 k | 真实类别 c)`，
  M 步加一平滑：
  - 误标率 = `conf[0][1]`（无坑被投成有坑）
  - 漏标率 = `conf[1][0]`（有坑被投成无坑）
- 专家标签固定：E 步后专家照片的后验重置为 one-hot，不再更新。
- 缺票不计：只对实际存在的投票累加，不为缺失投票补默认值。

**逐张复核（留一法）**

评估每张未确认照片时：

1. 从拟合数据中移除该照片及其全部投票；
2. 按剩余投票把相连的照片和标注员做并查集分组；
3. 每个**同时含正、负专家标签**的组单独做 EM 拟合（其余组不拟合）；
4. 只用已拟合组中标注员投给该照片的票，按贝叶斯公式（先验各半）
   计算有坑概率；
5. 无票可用 → `pending_manual_review`（待人工复核）。

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
```

服务监听 `http://127.0.0.1:8000`，交互式文档见 `/docs`。

### 本地运行

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 演示请求

```bash
python3 demo/demo.py            # 读取 demo/payload.json，调用 /review 与 /annotator-stats
# 或手动：
curl -X POST http://127.0.0.1:8000/review \
  -H 'Content-Type: application/json' \
  -d @demo/payload.json
```

## API

### `POST /review`

请求体（`demo/payload.json` 为完整示例）：

| 字段 | 说明 |
| --- | --- |
| `photos` | 照片编号列表，1–80 个，不可重复 |
| `annotators` | 标注员编号列表，1–12 个，不可重复 |
| `votes` | 投票记录 `[{photo, annotator, label}]`，`label`：1=有坑，0=无坑；缺票即不出现；同一照片同一标注员只能投一票 |
| `expert_labels` | 专家标签 `[{photo, label}]`，拟合中固定 |

响应：

- `confirmed`：专家已确认的照片（不参与复核）；
- `results`：每张未确认照片的复核结论：
  - `status: "scored"` → `pothole_probability`（有坑概率）与
    `votes_used`（每票的标注员及其误标率/漏标率）；
  - `status: "pending_manual_review"` → 无票可用，待人工复核；
  - `votes_ignored`：因标注员不在已拟合组而未采用的票数。

### `POST /annotator-stats`

请求体同上。在完整数据上按同样规则分组拟合，汇总每名标注员的
误标率/漏标率；所在组缺少正或负专家标签（或无投票）的标注员标为
`not_fitted`。

### `GET /health`

健康检查，返回 `{"status": "ok"}`。

## 测试

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

覆盖：加一平滑的精确数值、专家标签固定、缺票不计、连通分组、
“正负专家标签齐全才拟合”、留一法移除目标照片及其投票、概率方向与
精确值、API 入参上限（80 张/12 人）、非法引用与重复投票的 422 响应、
端到端复核流程，以及基于仿真数据的误标/漏标率恢复检验。

## 设计说明与限制

- 组内要求**正负专家标签齐全**是为了锚定两个类别，避免 EM 的标签
  交换不可辨识问题；不满足的组不拟合，其中标注员的票不用于复核。
- 每张未确认照片独立做一次留一拟合，因此同一标注员在不同照片的
  结果里误标/漏标率可能略有差异；汇总视图请用 `/annotator-stats`。
- 规模为 ≤80 张照片、≤12 名标注员，留一法最多触发 80 次分组拟合，
  每次是数百条投票上的小规模 EM，延迟在毫秒级。
- 结果为有坑概率，是否判为坑槽的阈值（如 0.5）由业务方决定，
  服务不代为下结论。
