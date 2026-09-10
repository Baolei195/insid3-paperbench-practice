# 固定种子后的 INSID3 聚类合并评分

## 背景与资料

INSID3 利用去偏特征完成跨图语义比较，同时保留原始特征中的图内结构。这个任务只处理种子聚类已经确定后的聚类评分和掩码生成，不要求实现特征提取、位置去偏、候选定位、聚类或种子选择。

请阅读以下直接相关资料，并据此独立实现 NumPy 版本：

- [论文第 3.1 节末尾及第 3.4 节，公式（12）—（14）](https://arxiv.org/html/2603.28480v1#S3.SS4)。
- 作者提交 `0c165a10cf52ab91f335883d06260de86854adbe` 中 [`models/insid3.py` 第 331–364 行](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L331-L364)。
- 同一提交的 [`utils/clustering.py::compute_cluster_prototypes` 第 28–44 行](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/utils/clustering.py#L28-L44)。

**本题以该固定提交的代码行为为准。** 论文说明方法动机；代码额外包含候选覆盖率权重，且跨图聚类分数的计算粒度、种子权重处理及阈值比较必须按指定代码保留。

## 需要完成的文件和接口

新建 `src/insid3_aggregation.py`，实现：

```python
def aggregate_from_seed(
    original_features: np.ndarray,
    debiased_features: np.ndarray,
    reference_prototype: np.ndarray,
    cluster_labels: np.ndarray,
    candidate_mask: np.ndarray,
    seed_id: int,
    threshold: float,
) -> dict[str, np.ndarray]:
    """返回每个聚类的合并分数以及目标网格上的最终掩码。"""
```

| 参数 | 定义 |
| --- | --- |
| `original_features` | 目标图的原始特征，形状 `(H, W, C)`。每个非零 patch 已经做过 L2 归一化。 |
| `debiased_features` | 同一目标图的位置去偏特征，形状 `(H, W, C)`。每个非零 patch 已经做过 L2 归一化，与原始特征逐位置对应。 |
| `reference_prototype` | 已形成并归一化的参考去偏原型，形状 `(C,)`；允许零向量。 |
| `cluster_labels` | 形状 `(H, W)` 的整数标签。标签恰为连续整数 `0,…,K−1`，每个标签至少出现一次，无 `-1` 标签。 |
| `candidate_mask` | 形状 `(H, W)` 的布尔或 0/1 候选 patch 掩码。它表示候选定位结果，并非最终输出掩码。 |
| `seed_id` | 已经选定的合法聚类标签；种子聚类至少有一个 patch 被 `candidate_mask` 选中。不得重新选择种子。 |
| `threshold` | 有限的标量阈值。按指定代码使用严格大于比较，不强制把种子并入最终掩码。 |

输入形状均非空，特征和阈值均为有限实数。评测只提供满足上述条件的输入，不考察非法输入的错误信息。需要归一化的聚类均值采用 `x / max(||x||₂, 1e-12)`，零均值保持为零。

浮点分数按 `rtol=1e-6, atol=1e-7` 与期望比较，允许正确的 float32 或 float64 结果；布尔掩码要求完全一致。该容差明显小于关键反例中的分数差异。实现可以正常使用 `src` 包内辅助模块和 Python 标准库，评测加载方式不限制这些组织形式。

## 输出与行为要求

返回字典包含：

- `cluster_scores`：形状 `(K,)` 的浮点数组，数组下标就是聚类标签，内容为作者代码应用候选覆盖率权重后的最终聚类分数。
- `mask`：形状 `(H, W)` 的布尔数组，根据每个 patch 所属聚类的最终分数与阈值生成。

需要保留作者代码对两种特征空间、聚类内统计和种子权重的不同处理。分数需要对所有聚类定义，聚类面积按全部所属 patch 计算。输出掩码以完整聚类为单位，不限于候选 patch。不得修改输入、硬编码样例、下载模型或依赖 GPU。只使用 Python 3.10+ 与 NumPy。

评测者将提供 `bonus/test_bonus_task.py`，在仓库根目录执行：

```sh
python -m pytest -q bonus/test_bonus_task.py
```

提交完整的 `src/insid3_aggregation.py`，而非仅返回方法解释。
