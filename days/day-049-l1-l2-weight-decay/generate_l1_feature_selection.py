from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent


def standardize(train, test):
    """把训练集和测试集转换到相同的数据尺度。

    标准化公式是：(原始值 - 均值) / 标准差。

    均值和标准差只能根据训练集计算，因为模型是在训练集的数据尺度下
    学习参数的。测试集必须沿用同一组均值和标准差，才能保证同一个数值
    在训练和测试时表达相同的含义。

    如果根据整个测试集重新计算，不仅会改变特征的含义，还会让评估过程
    利用本应未知的测试集分布信息，这种问题叫作数据泄漏。
    """
    # axis=0 表示分别计算每一列特征的均值和标准差。
    mean = train.mean(axis=0)
    scale = train.std(axis=0)

    # 训练集和测试集都使用从训练集得到的 mean 与 scale。
    return (train - mean) / scale, (test - mean) / scale


def soft_threshold(value, threshold):
    """L1 的近端算子，也叫软阈值函数。

    它先把参数的绝对值减少 threshold；如果减少后绝对值小于 0，
    就直接截断为 0。这一步正是 L1 能制造稀疏系数的关键。
    """
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def fit_l1(X, y, strength, steps=6000):
    """训练一个带 L1 正则化的线性回归模型。

    X 是输入特征，y 是正确答案，strength 控制 L1 惩罚强度，
    steps 表示重复更新多少次。函数还会保存每一步的结果，供绘图使用。
    """
    n = len(y)
    # 学习率决定每次更新参数时迈多大一步。
    # np.linalg.norm(X, 2) 会根据输入数据估计“更新可能有多猛烈”。
    # 数据带来的变化越猛烈，学习率就应该越小，避免一步跨得太远。
    data_scale = np.linalg.norm(X, 2) ** 2 / n
    learning_rate = 0.20 / data_scale
    weights = np.zeros(X.shape[1])
    weight_history = []
    data_loss_history = []
    objective_history = []

    for _ in range(steps):
        # 第 1 步：用当前参数做预测。每一行样本都会得到一个预测值。
        predictions = X @ weights

        # 第 2 步：预测值减去正确答案，得到每个样本的预测误差。
        errors = predictions - y

        # 第 3 步：根据所有样本的误差，计算每个参数应该调整的方向和幅度。
        # 除以 n 是取所有训练样本的平均影响。
        gradient = X.T @ errors / n

        # 第 4 步：先按照普通梯度下降更新，让预测误差变小。
        updated_weights = weights - learning_rate * gradient

        # 第 5 步：再执行 L1 软阈值，把小参数继续拉向 0。
        # 参数绝对值小于这个阈值时，会被直接变成 0。
        threshold = learning_rate * strength
        weights = soft_threshold(updated_weights, threshold)

        # 到这里，本轮真正的参数训练已经完成。
        # 下面只是可选的“记录区”：重新计算损失并保存历史，供绘图观察。
        # 它们不会参与参数更新；如果不需要画图或监控训练，可以全部删除。
        updated_predictions = X @ weights
        data_loss = 0.5 * np.mean((updated_predictions - y) ** 2)
        objective = data_loss + strength * np.sum(np.abs(weights))

        # copy() 保存当前这一轮参数的独立副本。
        # 如果直接保存 weights，后续更新可能影响已经记录的内容。
        weight_history.append(weights.copy())
        data_loss_history.append(data_loss)
        objective_history.append(objective)

    return (
        weights,
        np.asarray(weight_history),
        np.asarray(data_loss_history),
        np.asarray(objective_history),
    )


# 固定随机种子，保证每次运行都得到同一张示意图。
rng = np.random.default_rng(49)
n_train, n_test, n_features = 140, 2000, 8
# 只有前三个特征有真实作用；后五个特征的真实系数为 0，代表噪声。
true_weights = np.array([3.0, -2.2, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0])

# 生成模型的输入数据 X。
# X_train 有 140 行、8 列：每一行是一条训练样本，每一列是一个特征。
# X_test 有 2000 行、8 列，用来检查模型面对新数据时的预测效果。
# rng.normal() 默认从均值为 0、标准差为 1 的正态分布中随机取数。
X_train = rng.normal(size=(n_train, n_features))
X_test = rng.normal(size=(n_test, n_features))

# 按照事先设定的真实规律生成正确答案 y。
# 矩阵乘法 X @ true_weights 表示对每条样本计算：
# y = 3.0*x1 - 2.2*x2 + 1.4*x3。
# 因为 x4 到 x8 对应的真实系数都是 0，所以它们不会真正影响答案。
# 最后再加入均值为 0、标准差为 1.1 的随机噪声，模拟现实数据中的
# 测量误差、遗漏因素和其他无法完全解释的波动。
y_train = X_train @ true_weights + rng.normal(scale=1.1, size=n_train)
y_test = X_test @ true_weights + rng.normal(scale=1.1, size=n_test)

# 标准化每一列特征，具体处理和原因见 standardize() 中的注释。
X_train, X_test = standardize(X_train, X_test)

# 这里没有单独训练截距（bias）参数，所以先减去训练集目标值的平均数，
# 让训练目标以 0 为中心。模型只需要学习 8 个特征对应的权重。
# 预测测试集时，会在 X_test @ weights 后把 y_mean 加回来，恢复原来的尺度。
y_mean = y_train.mean()
y_train_centered = y_train - y_mean

# np.linalg.lstsq() 用来求解最小二乘问题，也就是寻找一组权重，
# 让模型预测值与正确答案之间的平方误差尽可能小。
# 它返回多个结果，[0] 表示只取其中的最优权重。
# 这里没有加入任何正则化，用作无正则化的对照结果。
weights_ols = np.linalg.lstsq(X_train, y_train_centered, rcond=None)[0]

# 使用前面定义的 fit_l1()，通过反复迭代得到 L1 正则化的权重。
weights_l1, weight_history, data_loss_history, objective_history = fit_l1(
    X_train, y_train_centered, strength=0.18
)

# np.linalg.solve(A, b) 用来求解 A @ weights = b 这样的线性方程。
# 下面的 A 和 b 来自 L2 正则化线性回归的数学公式。
# l2_strength 控制 L2 惩罚强度；单位矩阵 I 让惩罚作用于每个权重。
l2_strength = 0.08
l2_matrix = (
    X_train.T @ X_train / n_train
    + l2_strength * np.eye(n_features)
)
l2_target = X_train.T @ y_train_centered / n_train
weights_l2 = np.linalg.solve(l2_matrix, l2_target)

models = {
    "No regularization": weights_ols,
    "L1 regularization": weights_l1,
    "L2 regularization": weights_l2,
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), dpi=180)
axes = axes.ravel()
positions = np.arange(n_features)
width = 0.24
colors = ["#8A94A6", "#E45756", "#4C78A8"]

for offset, ((name, weights), color) in enumerate(zip(models.items(), colors)):
    axes[0].bar(
        positions + (offset - 1) * width,
        weights,
        width=width,
        label=name,
        color=color,
        alpha=0.92,
    )

axes[0].axhline(0, color="#333333", linewidth=0.8)
axes[0].set_xticks(positions, [f"x{i + 1}" for i in positions])
axes[0].set_title("Final learned coefficients")
axes[0].set_ylabel("coefficient value")
axes[0].legend(frameon=False, fontsize=8)
axes[0].text(
    4.9,
    2.35,
    "x4–x8 are noise features",
    ha="center",
    fontsize=9,
    color="#5B6472",
)

names = list(models)
nonzero_counts = [np.count_nonzero(np.abs(weights) > 1e-8) for weights in models.values()]
test_mse = [
    np.mean((y_test - (X_test @ weights + y_mean)) ** 2)
    for weights in models.values()
]

bars = axes[1].bar(names, nonzero_counts, color=colors, width=0.62)
axes[1].set_ylim(0, n_features + 1.2)
axes[1].set_ylabel("number of non-zero coefficients")
axes[1].set_title("L1 produces a sparse model")
axes[1].tick_params(axis="x", labelrotation=12)

for bar, count, mse in zip(bars, nonzero_counts, test_mse):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        count + 0.18,
        f"{count} non-zero\ntest MSE {mse:.2f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

# 记录的系数历史展示了训练期间每个参数如何变化：
# 有用特征逐步靠近真实值，噪声特征则被软阈值压到 0。
for feature_index in range(n_features):
    axes[2].plot(
        weight_history[:, feature_index],
        label=f"x{feature_index + 1}",
        linewidth=1.5,
    )
axes[2].axhline(0, color="#333333", linewidth=0.8)
axes[2].set_title("L1 coefficient paths during training")
axes[2].set_xlabel("training step")
axes[2].set_ylabel("coefficient value")
axes[2].legend(ncol=4, frameon=False, fontsize=8)
axes[2].set_xlim(0, 300)  # 放大早期迭代，避免后期平台期遮住收敛过程。

# 同时画数据损失和“数据损失 + L1 惩罚”，说明优化的目标是什么。
axes[3].plot(data_loss_history, label="data loss", color="#4C78A8")
axes[3].plot(objective_history, label="data loss + L1 penalty", color="#E45756")
axes[3].set_title("Objective during L1 training")
axes[3].set_xlabel("training step")
axes[3].set_ylabel("loss")
axes[3].legend(frameon=False, fontsize=8)
axes[3].set_xlim(0, 300)

fig.suptitle(
    "Synthetic example: only x1, x2 and x3 contain real signal",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "Day-49-l1-feature-selection.png",
    bbox_inches="tight",
    facecolor="white",
)
