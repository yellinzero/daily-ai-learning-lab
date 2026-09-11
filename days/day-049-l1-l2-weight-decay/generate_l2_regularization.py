from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent


def standardize(train, test):
    """使用训练集的统计量，把训练集和测试集转换到相同尺度。"""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    return (train - mean) / scale, (test - mean) / scale


def fit_with_gradient_descent(X, y, l2_strength, steps=800):
    """用梯度下降训练线性回归。

    l2_strength 控制 L2 惩罚强度；设为 0 时，不施加任何正则化。
    函数同时记录权重整体大小和预测损失，供后面的图表使用。
    """
    n = len(y)
    data_scale = np.linalg.norm(X, 2) ** 2 / n
    learning_rate = 0.45 / (data_scale + l2_strength)
    weights = np.zeros(X.shape[1])
    norm_history = []
    data_loss_history = []

    for _ in range(steps):
        predictions = X @ weights
        errors = predictions - y

        # data_gradient 根据预测误差调整权重，目标是让答案预测得更准确。
        data_gradient = X.T @ errors / n

        # L2 惩罚项的梯度是 λw。这里的 λ 就是 l2_strength。
        # 某个权重越大，对应的 l2_gradient 就越大，受到的收缩也越强。
        # 它通常会让权重逐渐变小，但不会像 L1 那样直接截断为 0。
        l2_gradient = l2_strength * weights

        # 将“拟合数据”和“压小权重”两个方向相加，再更新模型参数。
        weights = weights - learning_rate * (data_gradient + l2_gradient)

        # 以下历史数据只用于绘图，不参与下一步参数更新。
        updated_errors = X @ weights - y
        data_loss_history.append(0.5 * np.mean(updated_errors**2))
        # np.linalg.norm(weights) 把全部权重合并成一个数，
        # 用来衡量这组权重整体有多大，而不是统计非零权重的数量。
        norm_history.append(np.linalg.norm(weights))

    return weights, np.asarray(norm_history), np.asarray(data_loss_history)


# 固定随机种子，使文章中的数据和图片可以重复生成。
rng = np.random.default_rng(51)
n_train, n_test, n_features = 45, 3000, 30

# 30 个候选特征中只有前 5 个真正影响答案，其余 25 个是噪声特征。
true_weights = np.zeros(n_features)
true_weights[:5] = [3.0, -2.2, 1.4, 1.0, -0.8]

X_train = rng.normal(size=(n_train, n_features))
X_test = rng.normal(size=(n_test, n_features))
y_train = X_train @ true_weights + rng.normal(scale=2.0, size=n_train)
y_test = X_test @ true_weights + rng.normal(scale=2.0, size=n_test)

X_train, X_test = standardize(X_train, X_test)
y_mean = y_train.mean()
y_train_centered = y_train - y_mean

# 无正则化的最小二乘解。
weights_ols = np.linalg.lstsq(X_train, y_train_centered, rcond=None)[0]

# 选择一个适中的 L2 强度，并用梯度下降记录训练过程。
selected_strength = 0.30
weights_l2, l2_norm_history, l2_loss_history = fit_with_gradient_descent(
    X_train, y_train_centered, selected_strength
)
_, ols_norm_history, ols_loss_history = fit_with_gradient_descent(
    X_train, y_train_centered, l2_strength=0.0
)


def test_mse(weights):
    predictions = X_test @ weights + y_mean
    return np.mean((predictions - y_test) ** 2)


# np.logspace(-3, 1, 45) 会在 0.001 到 10 之间生成 45 个候选值。
# 使用对数间隔，是因为正则化强度往往需要跨多个数量级尝试。
# 我们依次测试这些 L2 强度，观察“太弱、适中、太强”的不同结果。
strengths = np.logspace(-3, 1, 45)
sweep_mse = []
sweep_norm = []
for strength in strengths:
    # 对于线性回归，L2 解可以直接写成一个线性方程。
    # np.linalg.solve(matrix, target) 会直接求出该 strength 下的权重，
    # 这里主要用于快速扫描许多 λ，而不是展示逐步训练过程。
    matrix = (
        X_train.T @ X_train / n_train
        + strength * np.eye(n_features)
    )
    target = X_train.T @ y_train_centered / n_train
    weights = np.linalg.solve(matrix, target)
    sweep_mse.append(test_mse(weights))
    sweep_norm.append(np.linalg.norm(weights))

sweep_mse = np.asarray(sweep_mse)
sweep_norm = np.asarray(sweep_norm)

# 开始绘图。
fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), dpi=180)
positions = np.arange(n_features)

axes[0, 0].plot(positions, true_weights, "o-", label="true coefficients")
axes[0, 0].plot(positions, weights_ols, "o-", markersize=3, label="no regularization")
axes[0, 0].plot(positions, weights_l2, "o-", markersize=3, label="L2 regularization")
axes[0, 0].axhline(0, color="#333333", linewidth=0.8)
axes[0, 0].axvspan(4.5, 29.5, color="#DCE3EC", alpha=0.45)
axes[0, 0].text(17, 3.2, "x6–x30 are noise features", ha="center", color="#596273")
axes[0, 0].set_title("L2 shrinks unstable coefficients")
axes[0, 0].set_xlabel("feature index")
axes[0, 0].set_ylabel("coefficient value")
axes[0, 0].legend(frameon=False, fontsize=8)

model_names = ["No regularization", "L2 regularization"]
model_mse = [test_mse(weights_ols), test_mse(weights_l2)]
model_norm = [np.linalg.norm(weights_ols), np.linalg.norm(weights_l2)]
bars = axes[0, 1].bar(model_names, model_mse, color=["#8A94A6", "#4C78A8"])
axes[0, 1].set_title("Smaller weights can generalize better")
axes[0, 1].set_ylabel("test MSE (lower is better)")
axes[0, 1].tick_params(axis="x", labelrotation=10)
axes[0, 1].set_ylim(0, max(model_mse) * 1.30)
for bar, mse, norm in zip(bars, model_mse, model_norm):
    axes[0, 1].text(
        bar.get_x() + bar.get_width() / 2,
        mse + 0.2,
        f"test MSE {mse:.2f}\nweight norm {norm:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

axes[1, 0].plot(ols_norm_history, label="no regularization", color="#8A94A6")
axes[1, 0].plot(l2_norm_history, label="L2 regularization", color="#4C78A8")
axes[1, 0].set_xlim(0, 250)
axes[1, 0].set_title("Overall weight size during training")
axes[1, 0].set_xlabel("training step")
axes[1, 0].set_ylabel("L2 norm of all weights")
axes[1, 0].legend(frameon=False, fontsize=8)

left_axis = axes[1, 1]
right_axis = left_axis.twinx()
left_axis.plot(strengths, sweep_mse, color="#E45756", label="test MSE")
right_axis.plot(strengths, sweep_norm, color="#4C78A8", label="weight norm")
left_axis.axvline(selected_strength, color="#333333", linestyle="--", linewidth=1)
left_axis.set_xscale("log")
left_axis.set_title("L2 strength needs to be tuned")
left_axis.set_xlabel("L2 strength λ")
left_axis.set_ylabel("test MSE", color="#E45756")
right_axis.set_ylabel("weight norm", color="#4C78A8")
left_axis.tick_params(axis="y", labelcolor="#E45756")
right_axis.tick_params(axis="y", labelcolor="#4C78A8")
left_axis.text(selected_strength * 1.12, max(sweep_mse) - 0.8, "selected λ=0.30", fontsize=8)

fig.suptitle(
    "L2 example: 45 training samples and 30 candidate features",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "Day-49-l2-regularization.png",
    bbox_inches="tight",
    facecolor="white",
)
