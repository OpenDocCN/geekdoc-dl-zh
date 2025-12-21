# 正则化

在整个 RLHF 优化过程中，使用了许多正则化步骤来防止奖励模型的过度优化。在这些情况下，过度优化看起来像是输出无意义文本的模型。一些优化“脱轨”的例子包括模型可以输出可遵循的数学推理，但答案极端错误，重复文本，切换语言或使用过多的特殊字符。本章涵盖了用于控制模型优化的不同方法。

在撰写本文时，最受欢迎的变体是使用 KL 距离来衡量当前策略与参考策略在生成样本之间的距离。“KL 距离”是训练过程中表示*优化距离*的口语化术语，尽管 KL 散度——衡量两个概率分布分离的数学方法——并不满足成为真正距离度量的形式属性（简单地称这个数字为距离比称其为分布差异的数值度量要容易）。文献中已经出现了许多其他正则化技术，然后在下一轮模型迭代中消失。也就是说，在生成核心 KL 距离之外的正则化通常用于稳定实验设置，然后在下一代中简化。尽管如此，了解约束 RLHF 中优化的工具仍然很重要。

*在本章中，我们使用<semantics><mi>x</mi><annotation encoding="application/x-tex">x</annotation></semantics>来表示提示，使用<semantics><mi>y</mi><annotation encoding="application/x-tex">y</annotation></semantics>来表示完成。这种符号在语言模型文献中很常见，其中方法作用于完整的提示-完成对，而不是单个标记。*

当在具有奖励模型的 RLHF 框架中使用时，一般公式如下：

<semantics><mrow><mi>r</mi><mo>=</mo><msub><mi>r</mi><mi>θ</mi></msub><mo>−</mo><mi>λ</mi><msub><mi>r</mi><mtext mathvariant="normal">reg.</mtext></msub><mrow><mo stretchy="false" form="prefix">(</mo><mn>22</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow> <annotation encoding="application/x-tex">r = r_\theta - \lambda r_{\text{reg.}} \qquad{(22)}</annotation></semantics>

参考实现如下：

<semantics><mrow><mi>r</mi><mo>=</mo><msub><mi>r</mi><mi>θ</mi></msub><mo>−</mo><msub><mi>λ</mi><mtext mathvariant="normal">KL</mtext></msub><msub><mi>𝒟</mi><mtext mathvariant="normal">KL</mtext></msub><mrow><mo stretchy="true" form="prefix">(</mo><msub><mi>π</mi><mtext mathvariant="normal">RL</mtext></msub><mo stretchy="false" form="prefix">(</mo><mi>y</mi><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="false" form="postfix">∥</mo><msub><mi>π</mi><mtext mathvariant="normal">ref</mtext></msub><mo stretchy="false" form="prefix">(</mo><mi>y</mi><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="true" form="postfix">)</mo></mrow><mrow><mo stretchy="false" form="prefix">(</mo><mn>23</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow> <annotation encoding="application/x-tex">r = r_\theta - \lambda_{\text{KL}} \mathcal{D}_{\text{KL}} \left( \pi_{\text{RL}}(y \mid x) \, \| \, \pi_{\text{ref}}(y \mid x) \right) \qquad{(23)}</annotation></semantics>

## 强化学习优化中的 KL 散度

对于数学定义，请参阅第三章关于问题设置的内容。回忆一下，KL 散度作为概率差异的度量定义为以下：

<semantics><mrow><msub><mi>𝒟</mi><mtext mathvariant="normal">KL</mtext></msub><mo stretchy="false" form="prefix">(</mo><mi>P</mi><mo stretchy="false" form="prefix">|</mo><mo stretchy="false" form="prefix">|</mo><mi>Q</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><munder><mo>∑</mo><mrow><mi>x</mi><mo>∈</mo><mi>𝒳</mi></mrow></munder><mi>P</mi><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mrow><mo stretchy="true" form="prefix">(</mo><mfrac><mrow><mi>P</mi><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow><mrow><mi>Q</mi><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow></mfrac><mo stretchy="true" form="postfix">)</mo></mrow><mrow><mo stretchy="false" form="prefix">(</mo><mn>24</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow> <annotation encoding="application/x-tex">\mathcal{D}_{\text{KL}}(P || Q) = \sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right) \qquad{(24)}</annotation></semantics>

在 RLHF 中，我们通常关注的两个分布是新模型版本的分布，例如 <semantics><mrow><mi>P</mi><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">P(x)</annotation></semantics>，以及参考策略的分布，例如 <semantics><mrow><mi>Q</mi><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">Q(x)</annotation></semantics>。不同的优化器使用不同的 KL 方向。在这本书中，最常用的“KL 惩罚”是相对于参考策略的反向 KL。在实践中，这简化为从 RL 模型中采样标记并从参考模型中计算概率的蒙特卡洛估计。直观上，这种反向 KL 具有数值属性，当新模型，<semantics><mi>P</mi><annotation encoding="application/x-tex">P</annotation></semantics> 或 <semantics><msub><mi>π</mi><mtext mathvariant="normal">RL</mtext></msub><annotation encoding="application/x-tex">\pi_{\text{RL}}</annotation></semantics>，在原始参考模型分配低概率的区域赋予大量概率时，会施加较大的惩罚。

另一个 KL 方向在 ML 中仍然经常被使用，例如在一些 RL 算法的内部信任区域计算中。这种惩罚直观上惩罚新模型，当其更新在<semantics><mi>Q</mi><annotation encoding="application/x-tex">Q</annotation></semantics> 或 <semantics><msub><mi>π</mi><mtext mathvariant="normal">ref</mtext></msub><annotation encoding="application/x-tex">\pi_{\text{ref}}</annotation></semantics>中的高概率区域不应用概率时。这更接近于用于蒸馏或行为克隆的目标。

### 适用于各代代的参考模型

KL 惩罚通常通过比较训练过程中生成的标记与静态参考模型之间的距离来实现。其直觉是，你正在训练的模型具有你希望保持接近的风格。这个参考模型通常是调整过的指令模型，但也可以是之前的 RL 检查点。通过简单的替换，我们从模型中采样的概率分布变为 <semantics><mrow><msub><mi>π</mi><mtext mathvariant="normal">RL</mtext></msub><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">\pi_{\text{RL}}(x)</annotation></semantics> 和 <semantics><mrow><msub><mi>π</mi><mtext mathvariant="normal">ref</mtext></msub><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">\pi_{\text{ref}}(x)</annotation></semantics>，如上式 23 所示（通常在标准定义中，当应用于 RL KL 惩罚时，表示为 <semantics><mi>P</mi><annotation encoding="application/x-tex">P</annotation></semantics> 和 <semantics><mi>Q</mi><annotation encoding="application/x-tex">Q</annotation></semantics>）。这种 KL 散度惩罚最早应用于对话代理，在大语言模型流行之前，但 KL 控制很快就被确立为微调预训练模型的核心技术[[162]](ch021.xhtml#ref-jaques2017sequence)，[[163]](ch021.xhtml#ref-jaques2020human)。

### 实现示例

在实践中，KL 散度的实现通常被近似[[164]](ch021.xhtml#ref-schulman2016klapprox)，这使得实现变得远为简单。根据上述定义，当直接从分布 <semantics><mrow><mi>P</mi><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">P(x)</annotation></semantics> 中采样时，KL 的总和可以转换为期望。在这种情况下，分布 <semantics><mrow><mi>P</mi><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">P(x)</annotation></semantics> 是当前正在训练的模型的生成分布（即不是参考模型）。然后，KL 散度的计算变为以下：

<semantics><mrow><msub><mi>𝒟</mi><mtext mathvariant="normal">KL</mtext></msub><mo stretchy="false" form="prefix">(</mo><mi>P</mi><mo stretchy="false" form="prefix">|</mo><mo stretchy="false" form="prefix">|</mo><mi>Q</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><msub><mi>𝔼</mi><mrow><mi>x</mi><mo>∼</mo><mi>P</mi></mrow></msub><mrow><mo stretchy="true" form="prefix">[</mo><mi mathvariant="normal">log</mi><mi>P</mi><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><mi mathvariant="normal">log</mi><mi>Q</mi><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="true" form="postfix">]</mo></mrow><mi>.</mi><mrow><mo stretchy="false" form="prefix">(</mo><mn>25</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow> <annotation encoding="application/x-tex">\mathcal{D}_{\text{KL}}(P \,||\, Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right]. \qquad{(25)}</annotation></semantics>

这种模式实现起来要简单得多，尤其是在直接处理语言模型训练中经常使用的对数概率时。

```py
[](#cb1-1)# Step 1: sample (or otherwise generate) a sequence from your policy
[](#cb1-2)generated_tokens = model.generate(inputs)
[](#cb1-3)
[](#cb1-4)# Step 2: score that generated sequence under both models
[](#cb1-5)#    for autoregressive LMs, you usually do:
[](#cb1-6)#      inputs_for_scoring = generated_tokens[:, :-1]
[](#cb1-7)#      labels           = generated_tokens[:, 1:]
[](#cb1-8)logits       = model.forward(generated_tokens[:, :-1]).logits
[](#cb1-9)ref_logits   = ref_model.forward(generated_tokens[:, :-1]).logits
[](#cb1-10)
[](#cb1-11)# convert to log-probs, then align labels to index into the logits
[](#cb1-12)logprobs     = F.log_softmax(logits, dim=-1)
[](#cb1-13)ref_logprobs = F.log_softmax(ref_logits, dim=-1)
[](#cb1-14)
[](#cb1-15)# gather the log-probs of the actual next tokens
[](#cb1-16)token_logprobs     = logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
[](#cb1-17)ref_token_logprobs = ref_logprobs.gather(-1, generated_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
[](#cb1-18)
[](#cb1-19)# now you can sum (or average) those to get the sequence log-prob,
[](#cb1-20)# and compute KL:
[](#cb1-21)seq_logprob     = token_logprobs.sum(dim=-1)
[](#cb1-22)ref_seq_logprob = ref_token_logprobs.sum(dim=-1)
[](#cb1-23)
[](#cb1-24)kl_approx = seq_logprob - ref_seq_logprob
[](#cb1-25)kl_full   = F.kl_div(ref_logprobs, logprobs, reduction='batchmean')
```

一些示例实现包括 [TRL](https://github.com/huggingface/trl/blob/5c21de30ae210e4251ead85517ba8dfe3f210e81/trl/trainer/ppo_trainer.py#L1150) 和 [Hamish Ivison 的 Jax 代码](https://github.com/hamishivi/EasyLM/blob/main/EasyLM/models/llama/llama_train_ppo.py#L278)。

## 预训练梯度

观察正则化的另一种方式是，你可能有一个希望模型保持接近的*数据集*，就像在 InstructGPT [[3]](ch021.xhtml#ref-ouyang2022training)中所做的那样，“为了修复公共 NLP 数据集上的性能退化”。为了实现这一点，他们修改了 RLHF 的训练目标。以等式 22 为例，我们可以通过从 RL 策略模型中采样，从用于 RLHF 的 RL 数据集中的提示 <semantics><mi>x</mi><annotation encoding="application/x-tex">x</annotation></semantics> 中获取 <semantics><mi>y</mi><annotation encoding="application/x-tex">y</annotation></semantics>，将其转化为一个优化目标函数，得到： <semantics><mrow><mi>J</mi><mo stretchy="false" form="prefix">(</mo><mi>θ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><msub><mi>𝔼</mi><mrow><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false" form="postfix">)</mo><mo>∼</mo><msub><mi>𝒟</mi><msub><mi>π</mi><mrow><mtext mathvariant="normal">RL</mtext><mo>,</mo><mi>θ</mi></mrow></msub></msub></mrow></msub><mrow><mo stretchy="true" form="prefix">[</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><mi>y</mi><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><mi>λ</mi><msub><mi>r</mi><mtext mathvariant="normal">reg.</mtext></msub><mo stretchy="true" form="postfix">]</mo></mrow><mrow><mo stretchy="false" form="prefix">(</mo><mn>26</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow> <annotation encoding="application/x-tex">J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right] \qquad{(26)}</annotation></semantics>

然后，我们可以在预训练期间使用的标准自回归下一个标记预测损失上添加额外的奖励，这个损失是在从预训练语料库（或另一个数据集）中抽取的一组文档上计算的，以保持文本一致性：

<semantics><mrow><mi>J</mi><mo stretchy="false" form="prefix">(</mo><mi>θ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><msub><mi>𝔼</mi><mrow><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo stretchy="false" form="postfix">)</mo><mo>∼</mo><msub><mi>𝒟</mi><msub><mi>π</mi><mrow><mtext mathvariant="normal">RL</mtext><mo>,</mo><mi>θ</mi></mrow></msub></msub></mrow></msub><mrow><mo stretchy="true" form="prefix">[</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="postfix">(</mo><mi>y</mi><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><mi>λ</mi><msub><mi>r</mi><mtext mathvariant="normal">reg.</mtext></msub><mo stretchy="true" form="postfix">]</mo></mrow><mo>+</mo><mi>γ</mi><msub><mi>𝔼</mi><mrow><mi>x</mi><mo>∼</mo><msub><mi>𝒟</mi><mtext mathvariant="normal">pretrain</mtext></msub></mrow></msub><mrow><mo stretchy="true" form="prefix">[</mo><mi mathvariant="normal">log</mi><mo stretchy="false" form="prefix">(</mo><msub><mi>π</mi><mrow><mtext mathvariant="normal">RL</mtext><mo>,</mo><mi>θ</mi></mrow></msub><mo stretchy="false" form="prefix">(</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo stretchy="false" form="postfix">)</mo><mo stretchy="true" form="postfix">]</mo></mrow><mrow><mo stretchy="false" form="prefix">)</mo></mrow></mrow> <annotation encoding="application/x-tex">J(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\pi_{\text{RL},\theta}}} \left[ r_{\theta}(y \mid x) - \lambda r_{\text{reg.}} \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log(\pi_{\text{RL},\theta}(x)) \right] \qquad{(27)}</annotation></semantics>

最近的研究提出了使用负对数似然项来平衡直接偏好优化（DPO）[[165]](ch021.xhtml#ref-pang2024iterative)的优化。鉴于 DPO 损失的成对性质，可以对相同的损失修改进行奖励模型训练，约束模型预测准确文本（未发表工作的实验室的谣言）。

其中<semantics><msub><mi>P</mi><mi>θ</mi></msub><annotation encoding="application/x-tex">P_{\theta}</annotation></semantics>是可训练的策略模型，<semantics><msub><mi>P</mi><mtext mathvariant="normal">ref.</mtext></msub><annotation encoding="application/x-tex">P_{\text{ref.}}</annotation></semantics>是一个固定的参考模型（通常是 SFT 检查点），而<semantics><mrow><mo stretchy="false" form="prefix">(</mo><msubsup><mi>c</mi><mi>i</mi><mi>w</mi></msubsup><mo>,</mo><msubsup><mi>y</mi><mi>i</mi><mi>w</mi></msubsup><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">(c_i^w, y_i^w)</annotation></semantics>和<semantics><mrow><mo stretchy="false" form="prefix">(</mo><msubsup><mi>c</mi><mi>i</mi><mi>l</mi></msubsup><mo>,</mo><msubsup><mi>y</mi><mi>i</mi><mi>l</mi></msubsup><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">(c_i^l, y_i^l)</annotation></semantics>表示对于提示<semantics><msub><mi>x</mi><mi>i</mi></msub><annotation encoding="application/x-tex">x_i</annotation></semantics>的获胜和失败完成。第一个项是标准的 DPO 逻辑损失：它通过使用似然比差异来增加获胜和失败之间的边界，<semantics><mrow><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mstyle displaystyle="false"><mfrac><msub><mi>P</mi><mi>θ</mi></msub><msub><mi>P</mi><mtext mathvariant="normal">ref.</mtext></msub></mfrac></mstyle></mrow><annotation encoding="application/x-tex">\log \tfrac{P_{\theta}}{P_{\text{ref.}}}</annotation></semantics>，而<semantics><mi>β</mi><annotation encoding="application/x-tex">\beta</annotation></semantics>控制这种偏好信号从参考中拉开的强度。第二个项是对获胜完成的长度归一化负对数似然惩罚，由<semantics><mi>α</mi><annotation encoding="application/x-tex">\alpha</annotation></semantics>加权，这有助于在绝对语言建模意义上保持首选文本的高可能性，而不仅仅是相对于被拒绝样本的相对更好。

## 其他正则化

在 RLHF 堆栈的其他部分中，控制优化定义得不够明确。大多数奖励模型除了标准的对比损失函数之外没有其他正则化。直接对齐算法通过<semantics><mi>β</mi><annotation encoding="application/x-tex">\beta</annotation></semantics>参数（参见直接对齐章节）以不同的方式处理 KL 发散的正则化。

Llama 2 提出了奖励模型训练的边界损失 [[44]](ch021.xhtml#ref-touvron2023llama):

<semantics><mrow><mi>ℒ</mi><mo stretchy="false" form="prefix">(</mo><mi>θ</mi><mo stretchy="false" form="postfix">)</mo><mo>=</mo><mi>−</mi><mrow><mi mathvariant="normal">log</mi><mo>⁡</mo></mrow><mrow><mo stretchy="true" form="prefix">(</mo><mi>σ</mi><mrow><mo stretchy="true" form="prefix">(</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><msub><mi>r</mi><mi>θ</mi></msub><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>r</mi></msub><mo>∣</mo><mi>x</mi><mo stretchy="false" form="postfix">)</mo><mo>−</mo><mi>m</mi><mo stretchy="false" form="postfix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>,</mo><msub><mi>y</mi><mi>r</mi></msub><mo stretchy="false" form="postfix">)</mo><mo stretchy="true" form="postfix">)</mo></mrow><mo stretchy="true" form="postfix">)</mo></mrow><mrow><mo stretchy="false" form="prefix">(</mo><mn>30</mn><mo stretchy="false" form="postfix">)</mo></mrow></mrow> <annotation encoding="application/x-tex">\mathcal{L}(\theta) = - \log \left( \sigma \left( r_{\theta}(y_c \mid x) - r_{\theta}(y_r \mid x) - m(y_c, y_r) \right) \right) \qquad{(30)}</annotation></semantics>

其中 <semantics><mrow><mi>m</mi><mo stretchy="false" form="prefix">(</mo><msub><mi>y</mi><mi>c</mi></msub><mo>,</mo><msub><mi>y</mi><mi>r</mi></msub><mo stretchy="false" form="postfix">)</mo></mrow><annotation encoding="application/x-tex">m(y_c, y_r)</annotation></semantics> 是两个数据点之间的边缘，这两个数据点 <semantics><msub><mi>y</mi><mi>c</mi></msub><annotation encoding="application/x-tex">y_c</annotation></semantics> 和 <semantics><msub><mi>y</mi><mi>r</mi></msub><annotation encoding="application/x-tex">y_r</annotation></semantics> 代表了两个注释者之间评分的 delta 数值差异。这可以通过让注释者在数值尺度上对输出进行评分来实现，或者使用量化的排名方法，例如 [李克特量表](https://en.wikipedia.org/wiki/Likert_scale)。

奖励边缘在直接对齐文献中得到了广泛的应用，例如奖励加权 DPO（Reward weighted DPO），“奖励感知偏好优化”（“Reward-aware Preference Optimization”，RPO），它将奖励模型得分整合到 DPO 损失的更新规则中 [[25]](ch021.xhtml#ref-adler2024nemotron)，或者具有回归损失公式中奖励 delta 加权的 REBEL [[166]](ch021.xhtml#ref-gao2024rebel)。
