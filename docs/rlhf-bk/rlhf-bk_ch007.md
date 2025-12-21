# 奖励模型

奖励模型在现代强化学习与人类反馈强化（RLHF）方法中处于核心地位，因为它们是学习复杂人类偏好的地方。它们使得我们的模型能够从难以指定的信号中学习。它们将数据中的复杂特征压缩成可以在下游训练中使用的表示——这是一种再次展示现代深度学习复杂容量的魔法。这些模型作为代理目标，通过以下章节研究的核心优化过程。

奖励模型在历史上被广泛用于强化学习研究，作为环境奖励的代理 [[55]](ch021.xhtml#ref-sutton2018reinforcement)。在现代形式中，奖励模型被提出作为一种研究价值对齐问题的工具 [[33]](ch021.xhtml#ref-leike2018scalable)。这些模型通常接受某种形式的输入，并输出一个单一的奖励标量值。这种奖励可以有多种形式——在传统的强化学习问题中，它试图近似问题的确切环境奖励，但在 RLHF 中，我们将看到奖励模型实际上输出的是某个输入“高质量”的概率（即在一对偏好关系中选择的答案）。RLHF 的奖励建模实践与逆强化学习密切相关，其中问题是在给定行为轨迹的情况下近似代理的奖励函数 [[96]](ch021.xhtml#ref-ng2000algorithms)，以及其他深度强化学习的领域。高层问题陈述是相同的，但实现和关注领域完全不同，因此它们通常被视为完全不同的研究领域。

最常见的奖励模型，通常称为布拉德利-特里奖励模型，也是本章的主要焦点，它预测一段文本接近“首选”文本的概率，这是基于训练比较的。在本节稍后，我们还将将这些模型与结果奖励模型（ORMs）、过程奖励模型（PRM）和其他类型的奖励模型进行比较。

*在本章中，我们使用<semantics><mi>x</mi><annotation encoding="application/x-tex">x</annotation></semantics>来表示提示，使用<semantics><mi>y</mi><annotation encoding="application/x-tex">y</annotation></semantics>来表示完成。这种符号在语言模型文献中很常见，其中方法操作的是完整的提示-完成对，而不是单个标记。*

## 训练奖励模型

奖励模型的经典实现源自于布拉德利-特里偏好模型 [[125]](ch021.xhtml#ref-BradleyTerry)。对于如何训练标准奖励模型，RLHF 有两种流行的表达方式——它们在数学上是等价的。首先，布拉德利-特里偏好模型定义了在两个物品 <semantics><mi>i</mi><annotation encoding="application/x-tex">i</annotation></semantics> 和 <semantics><mi>j</mi><annotation encoding="application/x-tex">j</annotation></semantics> 之间的成对比较中，一个评判者更喜欢 <semantics><mi>i</mi><annotation encoding="application/x-tex">i</annotation></semantics> 而不是 <semantics><mi>j</mi><annotation encoding="application/x-tex">j</annotation></semantics> 的概率：

<semantics><mrow><mi>P</mi><mo stretchy="false" form="prefix">(</mo><mi>i</mi><mo>></mo><mi>j</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mfrac><msub><mi>p</mi><mi>i</mi></msub><mrow><msub><mi>p</mi><mi>i</mi></msub><mo>+</mo><msub><mi>p</mi><mi>j</mi></msub></mrow></mfrac><mi>.</mi><mrow><mo stretchy="false" form="prefix">(</mo><mn>11</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">P(i > j) = \frac{p_i}{p_i + p_j}.\qquad{(11)}</annotation></semantics>

布拉德利-特里模型假设每个物品都有一个潜在的强度 <semantics><mrow><msub><mi>p</mi><mi>i</mi></msub><mo>></mo><mn>0</mn></mrow><annotation encoding="application/x-tex">p_i > 0</annotation></semantics>，并且观察到的偏好是这些潜在强度的有噪声的反映。通常使用无界分数重新参数化布拉德利-特里模型，其中 <semantics><mrow><msub><mi>p</mi><mi>i</mi></msub><mo>=</mo><msup><mi>e</mi><msub><mi>r</mi><mi>i</mi></msub></msup></mrow><annotation encoding="application/x-tex">p_i = e^{r_i}</annotation></semantics>，这导致以下形式：

<semantics><mrow><mi>P</mi><mo stretchy="false" form="prefix">(</mo><mi>i</mi><mo>></mo><mi>j</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mfrac><msup><mi>e</mi><msub><mi>r</mi><mi>i</mi></msub></msup><mrow><msup><mi>e</mi><msub><mi>r</mi><mi>i</mi></msub></msup><mo>+</mo><msup><mi>e</mi><msub><mi>r</mi><mi>j</mi></msub></msup></mrow></mfrac><mo>=</mo><mi>σ</mi><mo stretchy="false" form="prefix">(</mo><msub><mi>r</mi><mi>i</mi></msub><mo>−</mo><msub><mi>r</mi><mi>j</mi></msub><mo stretchy="false" form="postfix">)</mo><mi>.</mi><mrow><mo stretchy="false" form="prefix">(</mo><mn>12</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">P(i > j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i-r_j).\qquad{(12)}</annotation></semantics>

只有分数的差异才是重要的：对所有<semantics><msub><mi>r</mi><mi>i</mi></msub><annotation encoding="application/x-tex">r_i</annotation></semantics>添加相同的常数不会改变<semantics><mrow><mi>P</mi><mo stretchy="false" form="prefix">(</mo><mi>i</mi><mo>></mo><mi>j</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">P(i > j)</annotation></semantics>。这些形式不是自然法则，但它们是对人类偏好的有用近似，在 RLHF 中通常工作得很好。

为了训练奖励模型，我们必须制定一个满足上述关系的损失函数。在实践中，这是通过将语言模型转换为输出标量分数的模型来完成的，通常是通过一个小的线性头部来产生单个 logit。给定提示<semantics><mi>x</mi><annotation encoding="application/x-tex">x</annotation></semantics>和两个采样的完成<semantics><msub><mi>y</mi><mn>1</mn></msub><annotation encoding="application/x-tex">y_1</annotation></semantics>和<semantics><msub><mi>y</mi><mn>2</mn></msub><annotation encoding="application/x-tex">y_2</annotation></semantics>，我们使用奖励模型<semantics><msub><mi>r</mi><mi>θ</mi></msub><annotation encoding="application/x-tex">r_\theta</annotation></semantics>对它们进行评分，并将条件分数写成<semantics><mrow><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">r_\theta(y_i \mid x)</annotation></semantics>.

在成对比较中，给定奖励模型的成功概率变为：

<semantics><mrow><mi>P</mi><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>></mo><msub><mi>y</mi><mn>2</mn></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mfrac><mrow><mrow><mi mathvariant="normal">exp</mi><mo>⁡</mo></mrow><mrow><mo stretchy="true" form="prefix">(</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="true" form="postfix">)</mo></mrow></mrow><mrow><mrow><mi mathvariant="normal">exp</mi><mo>⁡</mo></mrow><mrow><mo stretchy="true" form="prefix">(</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mn>1</mn></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="true" form="postfix">)</mo></mrow><mo>+</mo><mrow><mi mathvariant="normal">exp</mi><mo>⁡</mo></mrow><mrow><mo stretchy="true" form="prefix">(</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mn>2</mn></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="true" form="postfix">)</mo></mrow></mrow></mfrac><mi>.</mi><mrow><mo stretchy="false" form="prefix">(</mo><mn>13</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">P(y_1 > y_2 \mid x) = \frac{\exp\left(r_\theta(y_1 \mid x)\right)}{\exp\left(r_\theta(y_1 \mid x)\right) + \exp\left(r_\theta(y_2 \mid x)\right)}.\qquad{(13)}</annotation></semantics>

我们将首选的完成项表示为 <semantics><msub><mi>y</mi><mi>c</mi></msub><annotation encoding="application/x-tex">y_c</annotation></semantics>（选择）和被拒绝的完成项表示为 <semantics><msub><mi>y</mi><mi>r</mi></msub><annotation encoding="application/x-tex">y_r</annotation></semantics>。

然后，通过最大化上述函数的对数似然（或者等价地，最小化负对数似然），我们可以得到用于训练奖励模型的损失函数：

第一种形式，正如[[3]](ch021.xhtml#ref-ouyang2022training)和其他作品所述：<semantics><mrow><mi>ℒ</mi><mo stretchy="false" form="prefix">(</mo><mi>θ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mi>−</mi><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mrow><mo stretchy="true" form="prefix">(</mo><mi>σ</mi><mrow><mo stretchy="true" form="prefix">(</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="postfix">(</mo><msub><mi>y</mi><mi>r</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="true" form="postfix">)</mo></mrow><mo stretchy="true" form="postfix">)</mo></mrow><mrow><mo stretchy="false" form="prefix">(</mo><mn>15</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)\qquad{(15)}</annotation></semantics>

其次，正如[[18]](ch021.xhtml#ref-askell2021general)和其他作品所述：<semantics><mrow><mi>ℒ</mi><mo stretchy="false" form="prefix">(</mo><mi>θ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mrow><mo stretchy="true" form="prefix">(</mo><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>r</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>∣</mi><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow></msup><mo stretchy="true" form="postfix">)</mo></mrow><mrow><mo stretchy="false" form="prefix">(</mo><mn>16</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">\mathcal{L}(\theta) = \log \left( 1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)} \right)\qquad{(16)}</annotation></semantics>

这些通过让 <semantics><mrow><mi mathvariant="normal">Δ</mi><mo>=</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>r</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">\Delta = r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x)</annotation></semantics> 和使用 <semantics><mrow><mi>σ</mi><mo stretchy="false" form="prefix">(</mo><mi mathvariant="normal">Δ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mi>−</mi><mi mathvariant="normal">Δ</mi></mrow></msup></mrow></mfrac></mrow><annotation encoding="application/x-tex">\sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}</annotation></semantics> 来实现，这暗示了 <semantics><mrow><mi>−</mi><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mi>σ</mi><mo stretchy="false" form="prefix">(</mo><mi mathvariant="normal">Δ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mo stretchy="false" form="prefix">(</mo><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mi>−</mi><mi mathvariant="normal">Δ</mi></mrow></msup><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mrow><mo stretchy="true" form="prefix">(</mo><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>r</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="postfix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow></msup><mo stretchy="true" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">-\log\sigma(\Delta) = \log(1 + e^{-\Delta}) = \log\left(1 + e^{r_{\theta}(y_r \mid x) - r_{\theta}(y_c \mid x)}\right)</annotation></semantics>。它们都出现在强化学习与人类反馈（RLHF）文献中。

## 架构

最常见的奖励模型实现方式是通过类似于 Transformer 的 `AutoModelForSequenceClassification` 的抽象，它将一个小型线性头部附加到执行两个结果之间分类的语言模型上——选择和拒绝。在推理时间，模型从模型中输出作为单个 logit 的 *文本片段被选择的概率*。

其他实现选项也存在，例如直接从最终嵌入中取一个线性层，但在开源工具中它们较少见。

## 实现示例

实现奖励建模损失相当简单。更多的实现挑战在于设置单独的数据加载器和推理管道。给定正确的数据加载器，其中包含标记化、选择和拒绝的提示以及完成项，损失实现如下：

```py
[](#cb1-1)import torch.nn as nn
[](#cb1-2)# inputs_chosen / inputs_rejected include the prompt tokens x and the respective
[](#cb1-3)# completion tokens (y_c or y_r) that the reward model scores jointly.
[](#cb1-4)rewards_chosen = model(**inputs_chosen)
[](#cb1-5)rewards_rejected = model(**inputs_rejected)
[](#cb1-6)
[](#cb1-7)loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

对于更大的图景，这通常是在一个因果语言模型中，额外添加了一个头部（并且与上述损失一起学习），它从最终的隐藏状态转换到输入的分数。此模型的结构如下：

```py
[](#cb2-1)import torch
[](#cb2-2)import torch.nn as nn
[](#cb2-3)import torch.nn.functional as F
[](#cb2-4)
[](#cb2-5)class BradleyTerryRewardModel(nn.Module):
[](#cb2-6)    """
[](#cb2-7) Standard scalar reward model for Bradley-Terry preference learning.
[](#cb2-8)
[](#cb2-9) Usage (pairwise BT loss):
[](#cb2-10) rewards_chosen = model(**inputs_chosen)    # (batch,)
[](#cb2-11) rewards_rejected = model(**inputs_rejected)  # (batch,)
[](#cb2-12) loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
[](#cb2-13) """
[](#cb2-14)    def __init__(self, base_lm):
[](#cb2-15)        super().__init__()
[](#cb2-16)        self.lm = base_lm  # e.g., AutoModelForCausalLM
[](#cb2-17)        self.head = nn.Linear(self.lm.config.hidden_size, 1)
[](#cb2-18)
[](#cb2-19)    def _sequence_rep(self, hidden, attention_mask):
[](#cb2-20)        """
[](#cb2-21) Get a single vector per sequence to score.
[](#cb2-22) Default: last non-padding token (EOS token); if no mask, last token.
[](#cb2-23) hidden: (batch, seq_len, hidden_size)
[](#cb2-24) attention_mask: (batch, seq_len)
[](#cb2-25) """
[](#cb2-26)
[](#cb2-27)        # Index of last non-pad token in each sequence
[](#cb2-28)        # attention_mask is 1 for real tokens, 0 for padding
[](#cb2-29)        lengths = attention_mask.sum(dim=1) - 1  # (batch,)
[](#cb2-30)        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
[](#cb2-31)        return hidden[batch_idx, lengths]  # (batch, hidden_size)
[](#cb2-32)
[](#cb2-33)    def forward(self, input_ids, attention_mask):
[](#cb2-34)        """
[](#cb2-35) A forward pass designed to show inference structure of a standard reward model.
[](#cb2-36) To train one, this function will need to be modified to compute rewards from both
[](#cb2-37) chosen and rejected inputs, applying the loss above.
[](#cb2-38) """
[](#cb2-39)        outputs = self.lm(
[](#cb2-40)            input_ids=input_ids,
[](#cb2-41)            attention_mask=attention_mask,
[](#cb2-42)            output_hidden_states=True,
[](#cb2-43)            return_dict=True,
[](#cb2-44)        )
[](#cb2-45)        # Final hidden states: (batch, seq_len, hidden_size)
[](#cb2-46)        hidden = outputs.hidden_states[-1]
[](#cb2-47)
[](#cb2-48)        # One scalar reward per sequence: (batch,)
[](#cb2-49)        seq_repr = self._sequence_rep(hidden, attention_mask)
[](#cb2-50)        rewards = self.head(seq_repr).squeeze(-1)
[](#cb2-51)
[](#cb2-52)        return rewards
```

在本节及以下内容中，奖励模型的大部分实现复杂性（以及大量训练后）都围绕着正确构建数据加载器和分布式学习系统。请注意，在训练奖励模型时，最常见的方法是仅训练 1 个 epoch 以避免过拟合。

## 变体

奖励建模是 RLHF 中相对未被充分探索的领域。传统的奖励建模损失在许多流行的工作中已经被修改，但这些修改尚未形成单一的最佳实践。

### 偏好间隔损失

在注释者提供李克特量表上的分数或排名的情况下，可以使用关系量的幅度进行训练。最常见的方法是将数据沿偏好方向二值化，将相对评分或排名的混合信息或强度简化为仅选择和拒绝的完成项。偏好幅度的附加信息已被用于改进模型训练，但尚未作为标准实践收敛。Llama 2 提出使用两个数据点之间的间隔，<semantics><mrow><mi>m</mi><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>,</mo><msub><mi>y</mi><mi>r</mi></msub><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">m(y_c, y_r)</annotation></semantics>，来区分偏好的幅度：

<semantics><mrow><mi>ℒ</mi><mo stretchy="false" form="prefix">(</mo><mi>θ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mi>−</mi><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mrow><mo stretchy="true" form="prefix">(</mo><mi>σ</mi><mrow><mo stretchy="true" form="prefix">(</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>r</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><mi>m</mi><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>,</mo><msub><mi>y</mi><mi>r</mi></msub><mo stretchy="false" form="postfix">)</mo><mo>−</mo><mi>m</mi><mo stretchy="true" form="postfix">)</mo></mrow><mo stretchy="true" form="postfix">)</mo></mrow><mrow><mo stretchy="false" form="prefix">(</mo><mn>17</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right)\qquad{(17)}</annotation></semantics>

例如，每个完成项通常根据质量从 1 到 5 进行排名。在所选样本被分配 5 分而拒绝 2 分的情况下，边缘 <semantics><mrow><mi>m</mi><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>,</mo><msub><mi>y</mi><mi>r</mi></msub><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mn>5</mn><mo>−</mo><mn>2</mn><mo>=</mo><mn>3</mn></mrow><annotation encoding="application/x-tex">m(y_c, y_r)= 5 - 2 = 3</annotation></semantics>。可以探索其他计算边缘的函数。

注意，在 Llama 3 中，由于团队观察到在扩展后改进效果逐渐减少，因此移除了边缘项。

### 每个提示下的多重比较平衡

InstructGPT 研究了在每个提示中使用不同数量的完成项对的影响，但在奖励模型训练中保持它们的平衡 [[3]](ch021.xhtml#ref-ouyang2022training)。为了做到这一点，他们为每个提示的每个比较加权更新损失。在实现层面，这可以通过将具有相同提示的所有示例包含在同一个训练批次中自动完成，从而自然地加权不同的对——不这样做会导致对提示的过度拟合。损失函数变为：

<semantics><mrow><mi>ℒ</mi><mo stretchy="false" form="prefix">(</mo><mi>θ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mi>−</mi><mfrac><mn>1</mn><mrow><mo stretchy="true" form="prefix">(</mo><mfrac linethickness="0"><mi>K</mi><mn>2</mn></mfrac><mo stretchy="true" form="postfix">)</mo></mrow></mfrac><msub><mi>𝔼</mi><mrow><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo>,</mo><msub><mi>y</mi><mi>c</mi></msub><mo>,</mo><msub><mi>y</mi><mi>r</mi></msub><mo stretchy="false" form="postfix">)</mo><mo>∼</mo><mi>D</mi></mrow></msub><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mrow><mo stretchy="true" form="prefix">(</mo><mi>σ</mi><mrow><mo stretchy="true" form="prefix">(</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>r</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="true" form="postfix">)</mo></mrow><mo stretchy="true" form="postfix">)</mo></mrow><mrow><mo stretchy="false" form="prefix">(</mo><mn>18</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">\mathcal{L}(\theta) = - \frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_c, y_r)\sim D} \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) \right) \right)\qquad{(18)}</annotation></semantics>

### K-wise 损失函数

对于 RLHF，有许多其他公式可以创建适合人类偏好的模型。其中一个例子，在流行的早期 RLHF 模型 Starling 7B 和 34B 中使用，是基于 Plackett-Luce 模型的 K-wise 损失函数 [[126]](ch021.xhtml#ref-zhu2024starling)，[[127]](ch021.xhtml#ref-liu2019learning)。

Zhu 等人于 2023 年[[128]](ch021.xhtml#ref-zhu2023principled)将设置形式化为如下。在提示或状态<semantics><msup><mi>s</mi><mi>i</mi></msup><annotation encoding="application/x-tex">s^i</annotation></semantics>下，<semantics><mi>K</mi><annotation encoding="application/x-tex">K</annotation></semantics>动作<semantics><mrow><mo stretchy="false" form="prefix">(</mo><msubsup><mi>a</mi><mn>0</mn><mi>i</mi></msubsup><mo>,</mo><msubsup><mi>a</mi><mn>1</mn><mi>i</mi></msubsup><mo>,</mo><mi>⋯</mi><mo>,</mo><msubsup><mi>a</mi><mrow><mi>K</mi><mo>−</mo><mn>1</mn></mrow><mi>i</mi></msubsup><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">(a_0^i, a_1^i, \cdots, a_{K-1}^i)</annotation></semantics>是从<semantics><mrow><mi>P</mi><mo stretchy="false" form="prefix">(</mo><msub><mi>a</mi><mn>0</mn></msub><mo>,</mo><mi>⋯</mi><mo>,</mo><msub><mi>a</mi><mrow><mi>K</mi><mo>−</mo><mn>1</mn></mrow></msub><mo stretchy="false" form="prefix">|</mo><msup><mi>s</mi><mi>i</mi></msup><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">P(a_0,\cdots,a_{K-1}|s^i)</annotation></semantics>中采样的。然后，使用标签器通过<semantics><mrow><msup><mi>σ</mi><mi>i</mi></msup><mo>:</mo><mo stretchy="false" form="prefix">[</mo><mi>K</mi><mo stretchy="false" form="postfix">]</mo><mo>↦</mo><mo stretchy="false" form="prefix">[</mo><mi>K</mi><mo stretchy="false" form="postfix">]</mo></mrow><annotation encoding="application/x-tex">\sigma^i: [K] \mapsto [K]</annotation></semantics>对偏好进行排序，其中<semantics><mrow><msup><mi>σ</mi><mi>i</mi></msup><mo stretchy="false" form="prefix">(</mo><mn>0</mn><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">\sigma^i(0)</annotation></semantics>是最受偏好的动作。这产生了一个偏好模型，它捕捉以下内容：

<semantics><mrow><mi>P</mi><mo stretchy="false" form="prefix">(</mo><msup><mi>σ</mi><mi>i</mi></msup><mo stretchy="false" form="prefix">|</mo><msup><mi>s</mi><mi>i</mi></msup><mo>,</mo><msubsup><mi>a</mi><mn>0</mn><mi>i</mi></msubsup><mo>,</mo><msubsup><mi>a</mi><mn>1</mn><mi>i</mi></msubsup><mo>,</mo><mi>…</mi><mo>,</mo><msubsup><mi>a</mi><mrow><mi>K</mi><mo>−</mo><mn>1</mn></mrow><mi>i</mi></msubsup><mo stretchy="false" form="postfix">)</mo><mo>=</mo><munderover><mo>∏</mo><mrow><mi>k</mi><mo>=</mo><mn>0</mn></mrow><mrow><mi>K</mi><mo>−</mo><mn>1</mn></mrow></munderover><mfrac><mrow><mrow><mi mathvariant="normal">exp</mi><mo>⁡</mo></mrow><mo stretchy="false" form="prefix">(</mo><msub><mi>r</mi><mrow><mi>θ</mi><mo>⋆</mo></mrow></msub><mo stretchy="false" form="prefix">(</mo><msup><mi>s</mi><mi>i</mi></msup><mo>,</mo><msubsup><mi>a</mi><mrow><msup><mi>σ</mi><mi>i</mi></msup><mo stretchy="false" form="prefix">(</mo><mi>k</mi><mo stretchy="false" form="postfix">)</mo></mrow><mi>i</mi></msubsup><mo stretchy="false" form="postfix">)</mo><mo stretchy="false" form="postfix">)</mo></mrow><mrow><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mi>k</mi></mrow><mrow><mi>K</mi><mo>−</mo><mn>1</mn></mrow></munderover><mrow><mi mathvariant="normal">exp</mi><mo>⁡</mo></mrow><mo stretchy="false" form="prefix">(</mo><msub><mi>r</mi><mrow><mi>θ</mi><mo>⋆</mo></mrow></msub><mo stretchy="false" form="prefix">(</mo><msup><mi>s</mi><mi>i</mi></msup><mo>,</mo><msubsup><mi>a</mi><mrow><msup><mi>σ</mi><mi>i</mi></msup><mo stretchy="false" form="prefix">(</mo><mi>j</mi><mo stretchy="false" form="postfix">)</mo></mrow><mi>i</mi></msubsup><mo stretchy="false" form="postfix">)</mo><mo stretchy="false" form="postfix">)</mo></mrow></mfrac><mrow><mo stretchy="false" form="prefix">(</mo><mn>19</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">P(\sigma^i|s^i,a_0^i,a_1^i,\ldots,a_{K-1}^i) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta\star}(s^i,a_{\sigma^i(k)}^i))}{\sum_{j=k}^{K-1}\exp(r_{\theta\star}(s^i,a_{\sigma^i(j)}^i))}\qquad{(19)}</annotation></semantics>

当 <semantics><mrow><mi>K</mi><mo>=</mo><mn>2</mn></mrow><annotation encoding="application/x-tex">K = 2</annotation></semantics> 时，这简化为布拉德利-特里（BT）模型用于成对比较。无论如何，一旦训练完成，这些模型在 RLHF 训练期间与其他奖励模型的使用方式相似。

## 结果奖励模型

对于语言模型和其他 AI 系统的*偏好调整*，大多数都是使用上面讨论的 Bradley Terry 模型进行的。对于推理密集型任务，可以使用结果奖励模型（ORM）。ORM 的训练数据构建方式与标准偏好调整类似。在这里，我们有一个问题陈述或提示，<semantics><mi>x</mi><annotation encoding="application/x-tex">x</annotation></semantics>和两个补全<semantics><msub><mi>y</mi><mn>1</mn></msub><annotation encoding="application/x-tex">y_1</annotation></semantics>和<semantics><msub><mi>y</mi><mn>2</mn></msub><annotation encoding="application/x-tex">y_2</annotation></semantics>。这里使用的归纳偏差是，一个补全应该是问题的正确解决方案，另一个是错误的，从而得到<semantics><mrow><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>,</mo><msub><mi>y</mi><mrow><mi>i</mi><mi>c</mi></mrow></msub><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">(y_c,y_{ic})</annotation></semantics>。

使用的模型形状与标准奖励模型非常相似，都是附加了一个线性层到可以输出单个 logit 的模型（在 RM 的情况下）——对于 ORM，接下来的训练目标略有不同[[129]](ch021.xhtml#ref-cobbe2021gsm8k)：

> [我们]使用联合目标来训练验证器，模型除了学习对模型补全进行正确或错误标记之外，还要学习原始的语言模型目标。在架构上，这意味着我们的验证器是语言模型，具有一个小的标量头，该头在每个标记的基础上输出预测。我们将这个标量头实现为一个单偏置参数和一个单增益参数，它们作用于语言模型最终反嵌入层输出的 logits。

为了翻译，这被实现为一个语言模型头，它可以对每个标记预测两个类别（1 表示正确，0 表示错误），而不是传统 RM 的分类头，该头输出整个序列的一个 logit。正式来说，根据[[130]](ch021.xhtml#ref-lyu2025exploring)，这可以表示为：

<semantics><mrow><msub><mi>ℒ</mi><mtext mathvariant="normal">CE</mtext></msub><mo stretchy="false" form="prefix">(</mo><mi>θ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mi>−</mi><msub><mi>𝔼</mi><mrow><mo stretchy="false" form="prefix">(</mo><mi>s</mi><mo>,</mo><mi>r</mi><mo stretchy="false" form="postfix">)</mo><mo>∼</mo><mi>𝒟</mi></mrow></msub><mo stretchy="false" form="prefix">[</mo><mi>r</mi><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><msub><mi>p</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><mi>s</mi><mo stretchy="false" form="postfix">)</mo><mo>+</mo><mo stretchy="false" form="prefix">(</mo><mn>1</mn><mo>−</mo><mi>r</mi><mo stretchy="false" form="postfix">)</mo><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mo stretchy="false" form="prefix">(</mo><mn>1</mn><mo>−</mo><msub><mi>p</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><mi>s</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="false" form="postfix">)</mo><mo stretchy="false" form="postfix">]</mo><mrow><mo stretchy="false" form="prefix">(</mo><mn>20</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">\mathcal{L}_{\text{CE}}(\theta) = -\mathbb{E}_{(s,r)\sim \mathcal{D}}[r\log p_\theta(s) + (1-r)\log(1-p_\theta(s))]\qquad{(20)}</annotation></semantics>

其中，<semantics><mrow><mi>r</mi><mo>∈</mo><mrow><mn>0</mn><mo>,</mo><mn>1</mn></mrow></mrow><annotation encoding="application/x-tex">r \in {0,1}</annotation></semantics>是一个二进制标签，其中 1 表示对给定提示的正确答案，0 表示错误答案，而<semantics><mrow><msub><mi>p</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><mi>s</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">p_\theta(s)</annotation></semantics>是模型训练中预测正确性概率的标量。

实现结果奖励模型（以及其他类型，如我们将通过过程奖励模型看到的）涉及根据完成是否为正确样本应用基于标记的交叉熵损失。这比标准布拉德利-特里奖励模型的有序选择-拒绝性质更接近语言模型损失，因为它不需要这种结构。

模型结构可能如下所示：

```py
[](#cb3-1)import torch.nn as nn
[](#cb3-2)import torch.nn.functional as F
[](#cb3-3)
[](#cb3-4)class OutcomeRewardModel(nn.Module):
[](#cb3-5)    def __init__(self, base_lm):
[](#cb3-6)        super().__init__()
[](#cb3-7)        self.lm = base_lm  # e.g., AutoModelForCausalLM
[](#cb3-8)        self.head = nn.Linear(self.lm.config.hidden_size, 1)
[](#cb3-9)
[](#cb3-10)    def forward(self, input_ids, attention_mask=None, labels=None):
[](#cb3-11)        """
[](#cb3-12) The input data here will be tokenized prompts and completions along with labels
[](#cb3-13) per prompt for correctness.
[](#cb3-14) """
[](#cb3-15)        outputs = self.lm(
[](#cb3-16)            input_ids=input_ids,
[](#cb3-17)            attention_mask=attention_mask,
[](#cb3-18)            output_hidden_states=True,
[](#cb3-19)            return_dict=True,
[](#cb3-20)        )
[](#cb3-21)        # Final hidden states: (batch, seq_len, hidden_size)
[](#cb3-22)        hidden = outputs.hidden_states[-1]
[](#cb3-23)        # One scalar logit per token: (batch, seq_len)
[](#cb3-24)        logits = self.head(hidden).squeeze(-1)
[](#cb3-25)
[](#cb3-26)        # Only compute loss on completion tokens (labels 0 or 1)
[](#cb3-27)        # Prompt tokens have labels = -100
[](#cb3-28)        mask = labels != -100
[](#cb3-29)        if mask.any():
[](#cb3-30)            loss = F.binary_cross_entropy_with_logits(
[](#cb3-31)                logits[mask], labels[mask].float()
[](#cb3-32)            )
[](#cb3-33)        return loss, logits
```

损失函数的简化版本如下：

```py
[](#cb4-1)# Assume model already has: model.lm (backbone) + model.head
[](#cb4-2)hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
[](#cb4-3)logits_per_token = model.head(hidden).squeeze(-1)  # (batch, seq_len)
[](#cb4-4)# This will sometimes be compressed as model.forward() in other implementations
[](#cb4-5)
[](#cb4-6)# Binary labels: 1=correct, 0=incorrect (prompt tokens masked as -100)
[](#cb4-7)mask = labels != -100
[](#cb4-8)loss = F.binary_cross_entropy_with_logits(
[](#cb4-9)    logits_per_token[mask], labels[mask].float()
[](#cb4-10))
```

这里的重要直觉是，ORM 会在序列中的每个标记处输出一个正确性的概率。这个过程可能是有噪声的，因为更新和损失传播是按标记进行的，这取决于结果和注意力映射。

这些模型仍在使用中，但在开源 RLHF 工具中支持较少。例如，在开创性工作 *Let’s Verify Step by Step* [[45]](ch021.xhtml#ref-lightman2023let) 中使用了相同类型的 ORM，但没有使用损失中的语言建模预测部分。然后，最终的损失是每个标记预测最终答案是否正确的交叉熵损失。

由于缺乏支持，结果奖励模型（ORM）被以多种方式使用。一些文献，例如 [[130]](ch021.xhtml#ref-lyu2025exploring)，继续使用 Cobbe 等人于 2021 年提出的原始定义。而另一些则不这样做。

## 流程奖励模型

流程奖励模型（PRMs），最初被称为流程监督奖励模型，是训练用于在每个思维推理过程的链中每一步输出分数的奖励模型。这些模型与仅在 EOS 标记处输出分数的标准 RM 或在每个标记处输出分数的 ORM 不同。流程奖励模型需要在每个推理步骤的末尾进行监督，然后以类似的方式进行训练，其中步骤中的标记被训练到它们的相关目标——在 PRMs 中目标是步骤，而在 ORMs 中是整个响应。

根据 [[45]](ch021.xhtml#ref-lightman2023let)，二分类标签的流程奖励模型（PRM）通常使用每步交叉熵损失进行优化：

<semantics><mrow><msub><mi>ℒ</mi><mtext mathvariant="normal">PRM</mtext></msub><mo stretchy="false" form="prefix">(</mo><mi>θ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mi>−</mi><msub><mi>𝔼</mi><mrow><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo>,</mo><mi>s</mi><mo stretchy="false" form="postfix">)</mo><mo>∼</mo><mi>𝒟</mi></mrow></msub><mrow><mo stretchy="true" form="prefix">[</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>K</mi></munderover><msub><mi>y</mi><msub><mi>s</mi><mi>i</mi></msub></msub><mi mathvariant="normal">log</mi><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>s</mi><mi>i</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>+</mo><mo stretchy="false" form="prefix">(</mo><mn>1</mn><mo>−</mo><msub><mi>y</mi><msub><mi>s</mi><mi>i</mi></msub></msub><mo stretchy="false" form="postfix">)</mo><mi mathvariant="normal">log</mi><mrow><mo stretchy="true" form="prefix">(</mo><mn>1</mn><mo>−</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>s</mi><mi>i</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="true" form="postfix">)</mo></mrow><mo stretchy="true" form="postfix">]</mo></mrow><mrow><mo stretchy="false" form="prefix">(</mo><mn>21</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow><annotation encoding="application/x-tex">\mathcal{L}_{\text{PRM}}(\theta) = - \mathbb{E}_{(x, s) \sim \mathcal{D}} \left[ \sum_{i=1}^{K} y_{s_i} \log r_\theta(s_i \mid x) + (1 - y_{s_i}) \log \left(1 - r_\theta(s_i \mid x)\right) \right] \qquad{(21)}</annotation></semantics>

其中 <semantics><mi>s</mi><annotation encoding="application/x-tex">s</annotation></semantics> 是一个带有 <semantics><mi>K</mi><annotation encoding="application/x-tex">K</annotation></semantics> 个标记步骤的采样思维链，<semantics><mrow><msub><mi>y</mi><msub><mi>s</mi><mi>i</mi></msub></msub><mo>∈</mo><mo stretchy="false" form="prefix">{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false" form="postfix">}</mo></mrow><annotation encoding="application/x-tex">y_{s_i} \in \{0,1\}</annotation></semantics> 表示第 <semantics><mi>i</mi><annotation encoding="application/x-tex">i</annotation></semantics> 个步骤是否正确，而 <semantics><mrow><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>s</mi><mi>i</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">r_\theta(s_i \mid x)</annotation></semantics> 是 PRM 预测的第 <semantics><msub><mi>s</mi><mi>i</mi></msub><annotation encoding="application/x-tex">s_i</annotation></semantics> 步骤在给定原始提示 <semantics><mi>x</mi><annotation encoding="application/x-tex">x</annotation></semantics> 下有效的概率。

这里是一个如何将每步标签打包到训练器中的示例，来自 HuggingFace 的 TRL（Transformer Reinforcement Learning）[[42]](ch021.xhtml#ref-vonwerra2022trl)：

```py
# Get the ID of the separator token and add it to the completions
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]

# Create the label 
labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]
```

传统上，PRMs 使用语言模型头部进行训练，仅在推理步骤的末尾输出一个标记，例如在对应于双换行符或其他特殊标记的标记处。这些预测通常为-1 表示错误，0 表示中性，1 表示正确。这些标签不一定与模型是否在正确路径上相关，但如果是正确步骤。

下面展示了 PRM 的一个示例构建。

```py
[](#cb6-1)import torch.nn as nn
[](#cb6-2)import torch.nn.functional as F
[](#cb6-3)
[](#cb6-4)class ProcessRewardModel(nn.Module):
[](#cb6-5)    def __init__(self, base_lm, num_classes=3):
[](#cb6-6)        super().__init__()
[](#cb6-7)        self.lm = base_lm  # e.g., AutoModelForCausalLM
[](#cb6-8)        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)
[](#cb6-9)
[](#cb6-10)    def forward(self, input_ids, attention_mask=None, labels=None):
[](#cb6-11)        """
[](#cb6-12) The inputs are tokenizer prompts and completions, where the the end of a 
[](#cb6-13) "reasoning step" is denoted by another non-padding token. 
[](#cb6-14) labels will be a list of labels, True, False, and Neutral (3 labels) which
[](#cb6-15) will be predicted by the model.
[](#cb6-16) """
[](#cb6-17)        outputs = self.lm(
[](#cb6-18)            input_ids=input_ids,
[](#cb6-19)            attention_mask=attention_mask,
[](#cb6-20)            output_hidden_states=True,
[](#cb6-21)            return_dict=True,
[](#cb6-22)        )
[](#cb6-23)        # Final hidden states: (batch, seq_len, hidden_size)
[](#cb6-24)        hidden = outputs.hidden_states[-1]
[](#cb6-25)        # One logit vector per token: (batch, seq_len, num_classes)
[](#cb6-26)        logits = self.head(hidden)
[](#cb6-27)
[](#cb6-28)        # Only compute loss at step boundaries (where labels != -100)
[](#cb6-29)        # Labels map: -1 -> 0, 0 -> 1, 1 -> 2 (class indices)
[](#cb6-30)        mask = labels != -100
[](#cb6-31)        if mask.any():
[](#cb6-32)            loss = F.cross_entropy(
[](#cb6-33)                logits[mask], labels[mask]
[](#cb6-34)            )
[](#cb6-35)        return loss, logits
```

核心损失函数看起来与结果奖励模型非常相似，标签应用在不同的间隔。

```py
[](#cb7-1)# Assume model outputs 3-class logits per token
[](#cb7-2)hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
[](#cb7-3)logits = model.head(hidden)  # (batch, seq_len, 3)
[](#cb7-4)
[](#cb7-5)# 3-class labels at step boundaries only: 0=-1, 1=0, 2=1 (others masked as -100)
[](#cb7-6)mask = labels != -100
[](#cb7-7)loss = F.cross_entropy(logits[mask], labels[mask])
```

## 奖励模型 vs. 结果 RM vs. 过程 RM vs. 价值函数

涵盖的各种奖励模型类型表明了在 RLHF 和其他后训练方法中“质量”可以衡量的范围。以下是对模型预测内容和训练方式的总结。

表 4：比较奖励模型类型。

| 模型类别 | 预测的内容 | 训练方式 | LM 结构 |
| --- | --- | --- | --- |
| **奖励模型** | 通过 EOS 标记选择的响应的概率来衡量文本质量 | 在完成之间的成对（或 N-wise）比较的对比损失 | 在 LM 特征之上的回归或分类头部 |
| **结果奖励模型** | 每个标记的正确答案概率 | 标记的结果对（例如，在可验证领域上的成功/失败） | 每个标记的语言模型交叉熵，其中每个标签都是结果级别标签 |
| **过程奖励模型** | 推理步骤结束时的中间步骤的奖励或分数 | 使用中间反馈或逐步注释（推理步骤中每 token 训练）进行训练 | 每个推理步骤只运行一次推理的语言模型头部，预测三个类别 -1, 0, 1 |
| **价值函数** | 给定当前状态下的预期回报 | 通过回归到序列中的每个点进行训练 | 每 token 输出一个分类 |

一些注意事项，鉴于上述表格有许多边缘情况。

+   在偏好调整和推理训练中，价值函数通常有一个折扣因子为 1，这使得价值函数更接近结果奖励模型，但具有不同的训练损失。

+   通过从中间状态进行回滚并收集结果数据来监督过程奖励模型。这融合了多个想法，但如果损失是按推理步骤标签计算的，则最好将其称为 PRM。

## 生成奖励建模

由于偏好数据的成本，出现了一个大型研究领域，即使用现有的语言模型作为人类偏好的评判者或在其他评估环境中 [[131]](ch021.xhtml#ref-zheng2023judging)。其核心思想是向语言模型提供如何评判的指令，一个提示，以及两个完成（就像对人类标注者所做的那样）。以下是一个示例提示，来自这里的一个开创性工作，用于聊天评估 MT-Bench [[131]](ch021.xhtml#ref-zheng2023judging)：

```py
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.
You should choose the assistant that follows the user's instructions and answers the user's question better.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.
Begin your evaluation by comparing the two responses and provide a short explanation.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
Do not allow the length of the responses to influence your evaluation.
Do not favor certain names of the assistants.
Be as objective as possible.
After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
```

由于 LLM-as-a-judge 在评估中的有效性，催生了许多其他评估，如 AlpacaEval [[132]](ch021.xhtml#ref-dubois2024length)、Arena-Hard [[133]](ch021.xhtml#ref-li2024crowdsourced)和 WildBench [[134]](ch021.xhtml#ref-lin2024wildbench)，许多人开始使用 LLM-as-a-judge 而不是奖励模型来创建和使用偏好数据。

出现了一个研究领域，专门研究如何使用所谓的“生成奖励模型” [[135]](ch021.xhtml#ref-mahan2024generative) [[136]](ch021.xhtml#ref-zhang2024generative) [[137]](ch021.xhtml#ref-ankner2024critique)（包括专门训练以成为有效评判者的模型 [[138]](ch021.xhtml#ref-kim2023prometheus)），但在 RM 评估中，它们往往落后于现有的奖励模型，这表明奖励建模是当前 RLHF 的重要技术。

提高 LLM-as-a-judge 工作流程鲁棒性的一个常见技巧是使用 0 的采样温度以减少评分的方差。

## 进一步阅读

奖励建模的学术文献在 2024 年确立了自己。早期奖励建模的大部分进展在于建立基准和识别行为模式。第一个 RM 基准，RewardBench，为测试奖励模型提供了共同的基础设施 [[139]](ch021.xhtml#ref-lambert2024rewardbench)。从那时起，RM 评估已经扩展到类似于可用于通用后训练模型的评估类型，其中一些评估测试了在已知真实答案的领域或与“感觉”更相似的领域上的预测准确性 [[139]](ch021.xhtml#ref-lambert2024rewardbench) 或与 LLM 作为裁判或与其他基准的相关性 [[140]](ch021.xhtml#ref-wen2024rethinking)。

新基准的例子包括：

+   **纯文本（通用聊天/偏好）：** RMB [[141]](ch021.xhtml#ref-zhou2024rmb)，RewardBench2 [[112]](ch021.xhtml#ref-malik2025rewardbench)，偏好代理评估 [[142]](ch021.xhtml#ref-frick2024evaluate)，或 RM-Bench [[143]](ch021.xhtml#ref-liu2024rm)。

+   **专门化的纯文本（数学等）：** 多语言奖励基准 (M-RewardBench) [[144]](ch021.xhtml#ref-gureja2024m)，用于检索增强生成 (RAG) 的 RAG-RewardBench [[145]](ch021.xhtml#ref-jin2024rag)，用于拼写错误的 ReWordBench [[146]](ch021.xhtml#ref-wu2025rewordbench)，RewardMATH [[147]](ch021.xhtml#ref-kim2024evaluating)，或 AceMath-RewardBench [[148]](ch021.xhtml#ref-liu2024acemath)。

+   **过程 RMs：** PRM Bench [[149]](ch021.xhtml#ref-song2025prmbench) 或 ProcessBench [[150]](ch021.xhtml#ref-zheng2024processbench) 以及 VisualProcessBench [[151]](ch021.xhtml#ref-wang2025visualprm) 或 ViLBench [[152]](ch021.xhtml#ref-tu2025vilbench) 的视觉基准。

+   **代理式 RMs：** Agent-RewardBench [[153]](ch021.xhtml#ref-men2025agentrewardbench) 或 CUARewardBench [[154]](ch021.xhtml#ref-lin2025cuarewardbench)。

+   **多模态：** MJ-Bench [[155]](ch021.xhtml#ref-chen2024mj), 多模态奖励基准 [[156]](ch021.xhtml#ref-yasunaga2025multimodal), VL 奖励基准 [[157]](ch021.xhtml#ref-li2024vlrewardbench), 或 VLRMBench [[158]](ch021.xhtml#ref-ruan2025vlrmbench)。

要了解奖励模型训练的进展，可以参考新的奖励模型训练方法，包括方面条件模型 [[159]](ch021.xhtml#ref-wang2024interpretable)，高质量的人类数据集 [[160]](ch021.xhtml#ref-wang2024helpsteer2) [[111]](ch021.xhtml#ref-wang2024helpsteer2p)，扩展实验 [[25]](ch021.xhtml#ref-adler2024nemotron)，广泛的实验 [[44]](ch021.xhtml#ref-touvron2023llama)，或去偏数据 [[161]](ch021.xhtml#ref-park2024offsetbias)。
