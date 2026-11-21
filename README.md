# 坑槽照片标注复核 API

质检员汇总多人标注后，用本服务复核未确认照片、评估两版坑槽检测器的换版效果。
基于 Dawid–Skene 模型（FastAPI + NumPy）估计每名标注员的误标率/漏标率，
并对每张未确认照片给出有坑概率；无票可用时按规则回退或标为“待人工复核”。

## 接口一览

- `POST /review`：留一法逐张复核未确认照片，给出有坑概率或“待人工复核”。
- `POST /annotator-stats`：在完整数据上分组拟合，汇总每人的误标率/漏标率。
- `POST /version-eval`：**换版评估**。按路段留一估计每张照片的有坑概率，
  结合两版检测器预测与专家抽查概率 π，估计全批平均判对差（新版减旧版）。
- `GET /health`：健康检查。

## 方法

**模型假设**

- 类别：0 = 无坑，1 = 有坑；类别先验固定为各 0.5。
- 每名标注员一个 2×2 混淆矩阵 `conf[c][k] = P(投票 k | 真实类别 c)`，
  M 步加一平滑：
  - 误标率 = `conf[0][1]`（无坑被投成有坑）
  - 漏标率 = `conf[1][0]`（有坑被投成无坑）
- 专家标签固定：E 步后专家照片的后验重置为 one-hot，不再更新。
- 缺票不计：只对实际存在的投票累加，不为缺失投票补默认值。

**逐张复核（留一法，`/review`）**

评估每张未确认照片时：

1. 从拟合数据中移除该照片及其全部投票；
2. 按剩余投票把相连的照片和标注员做并查集分组；
3. 每个**同时含正、负专家标签**的组单独做 EM 拟合（其余组不拟合）；
4. 只用已拟合组中标注员投给该照片的票，按贝叶斯公式（先验各半）
   计算有坑概率；
5. 无票可用 → `pending_manual_review`（待人工复核）。

**换版评估（按路段留一，`/version-eval`）**

每张照片额外携带路段编号、两版检测器预测（有坑/无坑）与入选专家抽查的
概率 π（0.01～1）。逐张照片：

1. **按路段留一**：Dawid–Skene 拟合只使用**其他路段**的照片、投票与专家
   标签（同路段照片整段剔除，避免同路段信息泄漏）；同一路段的照片共用
   一次留一路段拟合；
2. 用本张照片的可用票（标注员须在已拟合组中）按贝叶斯公式计算有坑概率
   `p`；**无票可用时 `p = 0.5`**；
3. 单版预期判对率：预测有坑时为 `p`、预测无坑时为 `1-p`；两版相减得到
   **预期判对差**（新版减旧版）；
4. 有专家标签的照片，再计算实际判对差（新版判对指示减旧版判对指示），
   并按逆概率加权把差额加回：
   `加权原值 = 预期差 + (实际差 − 预期差) / π`；
   无专家标签的照片，加权原值即预期差；
5. 全批平均 = 所有照片加权原值的算术平均，**保留加权原值不做截断，
   即使超出 ±1**；
6. 整批没有任何专家核验照片时返回 `comparable=false`、`mean_diff=null`
   并给出提示，逐张预期差仍照常返回。

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
python3 demo/demo.py            # 依次调用 /review、/annotator-stats、/version-eval
# 或手动：
curl -X POST http://127.0.0.1:8000/review \
  -H 'Content-Type: application/json' \
  -d @demo/payload.json
curl -X POST http://127.0.0.1:8000/version-eval \
  -H 'Content-Type: application/json' \
  -d @demo/version_eval_payload.json
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

### `POST /version-eval`

请求体（`demo/version_eval_payload.json` 为完整示例）。沿用票表与专家
标签，另加：

| 字段 | 说明 |
| --- | --- |
| `segments` | 每张照片的路段编号 `[{photo, segment}]`，须与 `photos` 一一对应，不可重复或遗漏 |
| `eval_photos` | 逐照片评估记录 `[{photo, old_prediction, new_prediction, inclusion_probability}]`；两版预测取 0/1（1=有坑）；`inclusion_probability` 即入选概率 π，范围 0.01～1 |

响应：

- `comparable`：整批是否含专家核验；`false` 时 `mean_diff=null` 并在
  `message` 中提示无法比较；
- `mean_diff`：全批平均判对差（新版减旧版），保留加权原值，不截断；
- `results`：逐照片的 `pothole_probability`（无票回退 0.5 时
  `probability_fallback=true`）、`expected_diff`（预期判对差）、
  `actual_diff`（无专家标签为 null）、`expert_audited` 与
  `weighted_diff`（计入平均值的加权原值，可能超出 ±1）。

### `GET /health`

健康检查，返回 `{"status": "ok"}`。

## 测试

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

覆盖：加一平滑的精确数值、专家标签固定、缺票不计、连通分组、
“正负专家标签齐全才拟合”、留一法移除目标照片及其投票、概率方向与
精确值、按路段留一剔除整段、API 入参上限（80 张/12 人）与 π 范围
（0.01～1）、非法引用/重复/路段缺失的 422 响应、预期判对差与逆概率
加权公式（含超出 ±1 不截断、无专家核验不可比较）、端到端复核与换版
评估流程，以及基于仿真数据的误标/漏标率恢复检验。

## 设计说明与限制

- 组内要求**正负专家标签齐全**是为了锚定两个类别，避免 EM 的标签
  交换不可辨识问题；不满足的组不拟合，其中标注员的票不用于复核。
- 每张未确认照片独立做一次留一拟合，因此同一标注员在不同照片的
  结果里误标/漏标率可能略有差异；汇总视图请用 `/annotator-stats`。
- 换版评估以**路段**为留一单位：同一路段的照片共用一次拟合，既避免
  同路段信息泄漏，也把拟合次数降到路段数量级；无票可用（或票的标注员
  不在已拟合组）时按需求以 0.5 作为有坑概率。
- 逆概率加权项可能把单张照片的加权原值放大到 ±1 之外（π 越小放大越
  明显），服务按需求保留原值不截断，平均值同样不截断。
- 规模为 ≤80 张照片、≤12 名标注员，留一法最多触发 80 次分组拟合，
  每次是数百条投票上的小规模 EM，延迟在毫秒级。
- 结果为有坑概率，是否判为坑槽的阈值（如 0.5）由业务方决定，
  服务不代为下结论。
