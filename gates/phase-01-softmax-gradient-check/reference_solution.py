"""阶段 1 综合练习的参考计算。"""

import numpy as np


X = 2.0
W = 0.5
H = 0.001
CORRECT_CLASS = 0


def stable_softmax(logits):
    """减去最大值后再计算 Softmax，避免指数运算溢出。"""
    shifted_logits = logits - np.max(logits)
    exp_values = np.exp(shifted_logits)
    return exp_values / np.sum(exp_values)


def forward(weight):
    """根据参数 w 完成一次前向计算，并返回 logits、概率和损失。"""
    logits = np.array([weight * X, 0.0], dtype=np.float64)
    probabilities = stable_softmax(logits)
    loss = -np.log(probabilities[CORRECT_CLASS])
    return logits, probabilities, loss


def relative_error(value_a, value_b):
    """计算两个数的相对误差。"""
    denominator = max(1e-12, abs(value_a) + abs(value_b))
    return abs(value_a - value_b) / denominator


def main():
    logits, probabilities, loss = forward(W)

    # 正确类别是第一个类别，因此 dL/dz1 = p1 - 1。
    # 又因为 z1 = w*x，所以 dz1/dw = x。
    loss_to_z1_gradient = probabilities[0] - 1.0
    z1_to_w_gradient = X
    analytic_gradient = loss_to_z1_gradient * z1_to_w_gradient

    logits_plus, probabilities_plus, loss_plus = forward(W + H)
    logits_minus, probabilities_minus, loss_minus = forward(W - H)
    numerical_gradient = (loss_plus - loss_minus) / (2.0 * H)

    absolute_error = abs(analytic_gradient - numerical_gradient)
    rel_error = relative_error(analytic_gradient, numerical_gradient)

    print("Task 1 · 前向计算")
    print("logits：", logits)
    print("Softmax 概率：", probabilities)
    print("Cross-Entropy loss：", loss)

    print("\nTask 2 · 解析梯度")
    print("dL/dz1：", loss_to_z1_gradient)
    print("dz1/dw：", z1_to_w_gradient)
    print("dL/dw：", analytic_gradient)

    print("\nTask 3 · 数值梯度")
    print("w + h：", W + H)
    print("logits(w + h)：", logits_plus)
    print("Softmax 概率(w + h)：", probabilities_plus)
    print("loss(w + h)：", loss_plus)
    print("w - h：", W - H)
    print("logits(w - h)：", logits_minus)
    print("Softmax 概率(w - h)：", probabilities_minus)
    print("loss(w - h)：", loss_minus)
    print("中心差分数值梯度：", numerical_gradient)

    print("\nTask 4 · 比较两种梯度")
    print("解析梯度：", analytic_gradient)
    print("数值梯度：", numerical_gradient)
    print("绝对误差：", absolute_error)
    print("相对误差：", rel_error)

    np.testing.assert_allclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert rel_error < 1e-5


if __name__ == "__main__":
    main()
