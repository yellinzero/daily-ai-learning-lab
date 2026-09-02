# 每天学一点 AI 知识配套代码

这是《每天学一点 AI 知识》系列的配套仓库，面向从零开始学习 AI 和大模型的读者。

该仓库不包含课程文章，仅提供相关的配套代码、阶段练习和参考答案。如需阅读课程文章，请通过下方入口查看。

## 在哪里阅读课程

- [微信公众号文集](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzYzNTI2ODIzOA==&action=getalbum&album_id=4613016257359511554#wechat_redirect)
- [Quaily 文集](https://quaily.com/yelling_universe/packs/65e7a745-2922-4eac-bb8f-7886676c03e4)
- 微信公众号：搜索“有趣独特实验室”
- 小红书：搜索“FuniqLab”

## 目前发布到哪里

目前已发布的课程到 **Day 40**。

### 阶段 0：认识 AI 与大模型（Day 01—20）

<details>
<summary>展开已发布课程</summary>

| 天数 | 主题 |
|---:|---|
| 01 | AI、机器学习与传统程序 |
| 02 | 模型、模型类与参数 |
| 03 | 参数量、xB 与数值精度 |
| 04 | 大语言模型行为：预测下一个 Token |
| 05 | 上下文是什么 |
| 06 | Tokenizer 与 Embedding |
| 07 | 训练、推理与前向传播 |
| 08 | Encoder、Decoder 与三种 Transformer 架构 |
| 09 | 三种 Attention |
| 10 | 隐状态、Logits 与 Softmax |
| 11 | Query、Key、Value 与因果掩码 |
| 12 | 自回归生成与解码策略 |
| 13 | Prefill、Decode 与 KV Cache |
| 14 | CPU、GPU、内存与显存 |
| 15 | NVIDIA 为什么领先 |
| 16 | 预训练、SFT 与后训练 |
| 17 | RLHF 与 DPO |
| 18 | 大语言模型与多模态模型 |
| 19 | Reasoning 与推理强度 |
| 20 | 模型蒸馏与能力迁移 |

</details>

### 阶段 1：数学、概率与数值计算（Day 21—38）

<details>
<summary>展开已发布课程</summary>

| 天数 | 主题 |
|---:|---|
| 21 | 标量、向量、矩阵与张量 |
| 22 | Shape、索引与 Broadcasting |
| 23 | 点积、矩阵乘法与线性变换 |
| 24 | 随机变量与概率分布 |
| 25 | 联合概率、条件概率与贝叶斯公式 |
| 26 | 极限、导数与积分 |
| 27 | 期望、方差与协方差 |
| 28 | 常见分布与采样 |
| 29 | Likelihood、MLE 与 MAP |
| 30 | 对数、NLL 与 LogSumExp |
| 31 | 熵、交叉熵与 KL Divergence |
| 32 | Perplexity 与概率校准 |
| 33 | 导数、偏导数与梯度 |
| 34 | 链式法则 |
| 35 | Jacobian、VJP 与 JVP |
| 36 | 计算图与自动微分 |
| 37 | 浮点数与数值稳定性 |
| 38 | 数值梯度检查 |

</details>

### 阶段 2：神经网络与训练稳定性（已发布到 Day 40）

| 天数 | 主题 |
|---:|---|
| 39 | 线性层 Wx+b |
| 40 | ReLU、Sigmoid、Tanh 与 GELU |

## 配套代码与练习

| 内容 | 入口 |
|---|---|
| Phase 1 Gate · Softmax、Cross-Entropy 与数值梯度检查 | [查看代码](gates/phase-01-softmax-gradient-check) |

## 怎样运行

本仓库使用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本和工具库。

先安装 uv。macOS 和 Linux 可以运行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows 可以运行：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

下载仓库后，在仓库根目录安装项目所需的 Python 和工具库：

```bash
uv sync
```

运行某份代码时，在原有 Python 命令前加上 `uv run`。例如：

```bash
uv run python gates/phase-01-softmax-gradient-check/reference_solution.py
```

各课程目录中的 `README.md` 提供了对应的完整运行命令。

## 目录说明

```text
daily-ai-learning-lab/
├── README.md          # 仓库首页，也就是当前文件
├── pyproject.toml     # Python 版本和工具库配置
├── uv.lock            # 锁定实际使用的工具库版本
├── days/              # 单篇课程的完整示例代码
└── gates/             # 每个学习阶段结束后的综合练习
```

- `days/` 保存单篇课程的配套代码。
- `gates/` 保存阶段综合练习、检查方法和参考答案。

## 关于参考答案

建议先自己完成练习，再使用参考答案核对计算过程和结果。

## 使用许可

仓库中的代码采用 [MIT License](LICENSE)。课程文章、图片和其他文字内容如无单独说明，不会因为代码采用 MIT License 而自动采用相同许可。
