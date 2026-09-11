# Day 49 · L1、L2 与权重衰减

这里保存了文章中两个完整案例的源代码。运行代码后，可以重新生成文章中的 L1 和 L2 对比图。

## 两份代码分别做什么

### L1 案例

`generate_l1_feature_selection.py` 会生成一组包含 8 个特征的模拟数据，其中只有前 3 个特征真正影响答案。

程序会比较三种训练结果：

- 不限制权重大小；
- 使用 L1，让不重要特征的权重直接变成 0；
- 使用 L2，把各个权重压小，但通常不会直接变成 0。

运行后会生成 `Day-49-l1-feature-selection.png`。

### L2 案例

`generate_l2_regularization.py` 会模拟一种更容易过拟合的情况：训练数据只有 45 条，却有 30 个候选特征，而且大多数特征只是噪声。

程序会比较不使用正则化与使用 L2 时的权重大小和测试误差，还会观察 L2 强度太小、适中或太大时分别会发生什么。

运行后会生成 `Day-49-l2-regularization.png`。

## 怎样运行

先在仓库根目录安装项目需要的 Python 和工具库：

```bash
uv sync
```

在仓库根目录运行两份代码：

```bash
uv run python days/day-049-l1-l2-weight-decay/generate_l1_feature_selection.py
uv run python days/day-049-l1-l2-weight-decay/generate_l2_regularization.py
```

两张图片会保存在 `days/day-049-l1-l2-weight-decay/` 目录中。建议先阅读 Day 49 正文，再结合正文中的图解查看代码。
