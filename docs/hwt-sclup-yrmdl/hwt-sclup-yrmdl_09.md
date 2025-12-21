# 在 TPUs 上部署 LLaMA 3-70B

> 原文：[`jax-ml.github.io/scaling-book/applied-inference`](https://jax-ml.github.io/scaling-book/applied-inference)

《如何扩展你的模型》第八部分 How To Scale Your Model (第七部分：推理 | 第九部分：分析)

让我们仔细看看如何在 TPU v5e 上部署 LLaMA 3-70B 模型。部署不同模型在屋顶线上的成本是多少？它们的 KV 缓存有多大？我们应该使用多大的批量大小？推理过程中参数和激活是如何分片的？让我们来估算一下生产环境中的延迟和吞吐量。  ### 内容

LLaMA 部署的故事是什么？

+   思考吞吐量

+   关于预填充的思考

可视化延迟吞吐量权衡练习题

*本节将探讨如何部署 LLaMA-3 以及如何高效地完成部署。与之前的“应用”部分一样，在查阅答案之前，请尝试用笔和纸自己解决问题！*

## LLaMA 部署的故事

让我们回顾一下 LLaMA 3-70B 的样子（参见 第六部分 以获取参考）：

| **超参数** | **值** |
| --- | --- |
| \(n_\text{layers}\) (L) | 80 |
| \(d_\text{model}\) (D) | 8,192 |
| \(d_{ff}\) (F) | 28,672 |
| \(n_\text{heads}\) (N) | 64 |
| \(n_\text{kv heads}\) (K) | 8 |
| \(d_\text{qkv}\) (H) | 128 |
| \(n_\text{embeddings}\) (V) | 128,256 |

让我们从一个问题开始：**我们应该在什么硬件上部署？** 答案基本上是，在 FLOPs/美元上最便宜的那个。 这并不总是正确的，有时更多的 HBM 或 ICI 带宽比 FLOPs 更关键，但这是一个好的经验法则。 因此，我们通常希望部署在 TPU v5e 上，我们当前的专用推理芯片（截至 2025 年 2 月的价格来自 [Google Cloud 定价](https://cloud.google.com/tpu/pricing)）：

| **TPU 类型** | **bfloat16 FLOPs/s** | **Google Cloud 美元/小时** | **FLOPs/$** |
| --- | --- | --- | --- |
| H100 | 9.9e14 | $10.8 | 3.3e17 |
| v5p | 4.59e14 | $4.2 | 3.9e17 |
| v5e | 1.97e14 | $1.2 | **5.8e17** |

每个 TPU v5e 都有 16GB 的 HBM，这将要求我们非常积极地分片我们的模型。让我们先思考一些可能对我们很重要的基本量：

**问题：LLaMA 3-70B 每个标记的 KV 缓存有多大？*你可以假设我们以 int8 格式存储它们。这决定了在给定拓扑结构上我们的批量大小可以有多大。*

在你思考过后点击这里！

LLaMA 3-70B 有 8 个 KV 头，因此每个标记的大小为 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`。

**注意这有多大！** 如果我们有一个 32k 个 token 的序列长度（这是常见的），这将使用 `162e3 * 32,768 = 5.3GB / sequence`。对于 BS=240，这将达到 1.3TB！由于 TPU v5e 每个只有 16GB，我们需要大约 `(70e9 + 1.3e12) / 16e9 = 86` 个 TPU v5e 芯片才能容纳这么多内存。也请注意，这与 70GB 的模型参数相比有多大。

**问题：** 假设我们希望在批处理大小为 32 和 8192 序列长度的情况下服务 L3 70B，并且所有内容（参数和 KV）都在 int8 中。这将使用多少总内存？我们能在最小的切片上服务这个吗？

答案

由于我们的 KV 在 int8 中是 `160e3` 字节，我们的总 KV 内存是 `160e3 * 8192 * 32 = 41.9e9` 字节。我们的参数是 `70e9` 字节，因为我们每个参数有 1 个字节。因此，我们的总内存使用量是 `41.9e9 + 70e9 = 112GB`。

我们能使用的最小切片将包含 `112e9 / 16e9 = 7` 个 TPU，或者（四舍五入到偶数大小），TPU v5e `4x2`。这将非常紧凑，我们可能无法完全考虑到其他开销，因此我们可能至少需要 `4x4`（或者减少批处理大小）。

**问题：** 在这个批处理大小和量化在 TPU v5e `4x2`上，我们大致可以期望每个解码步骤的多少延迟？吞吐量（tokens / sec / chip）是多少？对于 `4x4` 呢？**假设我们在 bfloat16 中执行 FLOPs，并且所有内容都是完全分片的。**

答案

我们可以调用前一个部分中的公式

\[\begin{align*} \tiny \text{理论步骤时间（通用）} = \underbrace{\frac{\text{批处理大小} \times \text{KV 缓存大小}}{\tiny \text{总内存带宽}}}_{\text{注意力（总是带宽限制的）}} + \underbrace{\max\left(\frac{2 \times \text{批处理大小} \times \text{参数数量}}{\text{总 FLOPs/s}}, \frac{\text{参数大小}}{\text{总内存带宽}}\right)}_{\tiny \text{MLP（可能是计算限制的）}} \end{align*}\]

在这里，我们的关键批处理大小将是大约 120，因为我们的参数在 int8 中，但我们的 FLOPs 在 bfloat16 中。我们也可以手动计算 RHS 的最大值，但这基本上是我们已经做过多次的计算。**因此，我们的矩阵乘法和 FLOPs 都已经进入了内存限制阶段。**

严格地看内存带宽，我们的步骤时间是 `(KV 大小 + 参数大小) / (8 * HBM 带宽) = 112e9 / (8 * 8.1e11) = 17ms`。**因此，从理论上讲，我们的步骤时间大约是 17ms。**我们的吞吐量将是 `32 / .017 = 1882 tokens / sec`，或者 `1882 / 8 = 235 tokens / sec / chip`。

这里有一个需要注意的地方，就是检查我们是否可能在矩阵乘法上受到 ICI 的限制。我们可以在这里为其分配 2 个轴，因此当 $Y > 2 * F / 2200 = 2 * 28672 / 2200 = 26$ 时，我们在理论上受到 ICI 的限制，所以没问题！

如果我们在 `4x4` 上运行，我们仍然在 ICI 方面没有问题，所以我们的延迟将降低到 `17 / 2 = 8.5ms`，但每片的吞吐量将保持不变。

### 考虑吞吐量

让我们花点时间纯粹地思考吞吐量。当我们优化吞吐量时，我们希望达到计算限制，这意味着我们接近利用所有 TPU MXU 容量。通常这意味着我们希望批量大小尽可能大，这样我们就能做尽可能多的工作。

**问题：** 在 TPU v5e 上，使用 bfloat16 权重和激活，我们的批量大小需要多大才能在 matmuls 中达到计算限制？如果我们使用 int8 权重但在 bfloat16 中执行 FLOPs 会怎样？关于 int8 权重和 int8 FLOPs 又会怎样？

回答

如第七部分所述，对于任何 $B \ll D, F$ 的 bfloat16 matmul，我们有

\[\begin{equation*} T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240 \end{equation*}\]

当我们的权重在 int8 时，分母中会损失一个因子，所以我们有 $2BDF / DF = 2B > 240$，或者同样 $B > 120$，是之前临界批量大小的一半。这对我们来说非常有帮助！当我们使用 int8 权重和 int8 FLOPs 时，我们必须使用 TPU FLOPs/s 的 int8 值，这从 bfloat16 的 1.97e14 增加到 3.94e14，几乎翻倍。这意味着我们回到了大约 $B > 240$ 的起点。

int8 权重和 bfloat16 FLOPs 的情况相当常见，因为无损量化参数通常比进行低精度算术更容易。

**问题：** 使用 bfloat16、int8 和 int4（包括 KVs 和参数）以及 8k 上下文，我们能在 TPU v5e 上为 LLaMA 3-70B 提供最小的拓扑吗？*你可以将 KV 缓存视为在这个例子中可以忽略不计的小。*

回答

这很简单！如果我们对很小的批量大小没有问题，那么唯一的限制就是将参数内存拟合到 HBM 中，即它只是 `ceil(num_params * sizeof(dtype) / HBM per TPU)`，或者 `ceil(70e9 * sizeof(dtype) / 16e9)` 四舍五入到最接近的合理拓扑（2 的某个倍数）：

| dtype | param size | KV size / token (bytes) | min TPU v5es | actual min slice | remaining HBM for KV caches | num KV caches @ 8k |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 140GB | 324kB | 8.75 | 4x4 = 16 chips | 116 | 43 |
| int8 | 70GB | 162kB | 4.38 | 4x2 = 8 chips | 58 | 43 |
| int4 | 35GB | 81kB | 2.81 | 2x2 = 4 chips | 29 | 43 |

这非常酷！它告诉我们，如果我们想的话，我们可以在 TPU v5e 2x2 上拟合 LLaMA 70B。但你会注意到 KV 缓存的数目非常小。那就是我们的批量大小！这意味着我们将得到非常差的 FLOPs 利用率。我们会非常高兴使用更大的拓扑来将我们的批量大小提高到 240。

**问题：** 假设我们使用适合这些拓扑的最大批量大小，我们能为每个生成步骤预期多少延迟？

回答

这也很简单，因为我们选择批处理大小来填满所有的 HBM！这只是一个问题，即加载一个完整的 TPU v5e 字节数需要多长时间进入 MXU。这仅仅是`v5e HBM / v5e HBM 内存带宽 = 16GB / 8.2e11 = 19ms`，所以这是**每步 19ms**。假设我们的生成文本的中位数长度为 512 个标记，那么每次解码大约需要 9 秒。注意，如果我们使用更小的批处理大小，例如如果我们只查看 int4 模型参数，我们的最小延迟大约是每步 10ms，因为 HBM 不再满。

**总结**：我们可以通过询问从 HBM 加载所有模型参数到 MXU 需要多长时间来降低解码延迟的下限。当我们的 KV 缓存较小时，你可以将每一层视为逐块加载权重然后丢弃它们。除非我们使用大批处理大小或大量的跨设备通信，这通常是一个合理的界限（在 1.5 倍以内）。当我们的批处理大小更大时，我们需要将 KV 缓存加载也建模，因为这将主导参数。

同样，在 FLOPs 限制区域（例如训练或大批推理），我们可以使用\(\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)\)的下限，这假设没有通信。

**问题**：对于这些，每芯片的吞吐量是多少（以查询/芯片计）？*你可以假设我们的中值解码长度为 512 个标记。*

答案

这是一个重要的问题，因为它与成本/标记直接相关。

根据我们对中值解码长度的假设，我们的吞吐量仅仅是\(B / (\text{每步延迟} \cdot \text{中值步数} \cdot N) \approx 43 / (0.019 * 512 * N)\)。这给我们大约\((4.42 / N)\) QPS，所以将\(N\)代入我们得到：

| 数据类型 | 每芯片 QPS |
| --- | --- |
| bfloat16 | 0.27 |
| int8 | 0.55 |
| int4 | 1.11 |

注意，这相当乐观，因为它完全忽略了前向传递的工作内存（分配给激活和注意力的内存）。这对于 Flash Attention 来说并不荒谬，但也不是现实的。实际的数字可能大约是这个数字的一半。为了获得绝对的最大吞吐量，我们可能需要将芯片数量加倍以上，并且显著增加批处理大小。

**问题**：如果我们加倍上述每个示例中的拓扑结构，我们的峰值吞吐量会如何变化？

答案

如果我们使用 4x8 切片在 bfloat16 中，我们将剩下 372GB 用于 KV 缓存，这将使我们能够将批处理大小提高到 140。然后，由于我们的步长时间保持不变，我们将有`14.39 / num_chips`的吞吐量，

| 数据类型 | 每芯片 QPS |
| --- | --- |
| bfloat16（在 4x8 上） | 0.44 |
| int8（在 4x4 上） | 0.90 |
| int4（在 2x4 上） | 1.80 |

进一步增加将带来更大的收益！重要的结论是，**在所有情况下，最小的拓扑结构并不一定是性能最佳的拓扑结构**，如果我们受 KV 缓存大小的限制。

**问题：** 现在我们来深入探讨分片的问题。假设我们想在 TPU v5e 4x8 上以 bfloat16 格式提供服务。在生成过程中，我们为 TPU v5e 4x8 上的模型使用什么分片方式？我们能避免成为通信瓶颈吗？

答案

如前节所述，我们在生成过程中的分片选项实际上只有一个：模型并行。在我们成为通信瓶颈之前我们能做多少？正如前节所述，我们的模型在大约以下情况下成为通信瓶颈：

\[Y > \frac{F \cdot M_Y}{2200}\]

对于 LLaMA 3-70B，我们有`F = 28,672`，所以如果我们进行 2 轴模型分片，这将给我们大约 \(Y = 28672 \cdot 2 / 2200 = 26\)，因此，一般来说，我们可以扩展到大约 16 个芯片而不会成为通信瓶颈，这让我们可以使用`4x4`而不是`4x8`。一般来说，由于我们并没有完美地重叠计算，即使这个估计也过于乐观。

**要点：** 实际上，我们无法在 4x8 上仅使用纯模型并行来提供服务。我们能做到的最好情况是 4x2 或者*可能*是 4x4。

然而，正如我们之前讨论的，当我们的批量大小较小时，我们通常可以进行更多的模型并行而不会显著影响吞吐量，因为我们的模型是内存带宽瓶颈而不是 FLOPs 瓶颈。我们之前说过这个值大约是 $Y=F / (8\cdot B)$，所以如果我们使用批量大小 64，理论上我们可以进行到`Y = 28,672 / (8 * 64) = 56`个模型并行，在我们成为 ICI 瓶颈之前。为了验证这一点，我们可以查看单个矩阵乘法的$T_\text{ici comms}$，$T_\text{hbm comms}$和$T_\text{math}$。我们明显有：

\[\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}\]

对于`4x8`，这将给我们 $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`，$T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.1e11) = 18us`，和 $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`，所以在理论上我们仍然是 HBM 带宽瓶颈，这是很好的！*注意，从`4x4`扩展到`4x8`可能从吞吐量角度来看并不有帮助，但它会减少我们的延迟！*

如果我们查看 int8 和 int4 配置，我们可以使用纯模型并行来完成这些。所以，我们已经达到了一个点，量化实际上给我们带来了超越更快 FLOPs 的有意义的优势：它让我们在使用通信瓶颈之前可以使用更大的批量大小。**所以这个故事的结果是，我们无法在 4x8 上实现峰值吞吐量，但对于 int8 和 int4 配置，我们可以进行纯模型并行**。

**提示**：最大有用的模型并行度取决于 \(d_{ff}\) 和你在哪个轴上对模型进行分片。最大值通常在 8 到 32 之间，具体取决于模型大小。你可以超过这个限制以在吞吐量有一定成本的情况下提高延迟。

### 预填充怎么样？

我们在这里主要忽略了预填充，因为它要简单得多。让我们将几个概念结合起来，思考端到端的情况。

**问题**：假设我们在预填充期间达到 40% 的 FLOPs 利用率。长度为 8192 的预填充在 16 个 TPU v5e 芯片上需要多长时间？

答案

在 8k 标记时，我们完全受计算限制，所以我们只需要考虑 FLOPs。我们知道我们的模型有 `70e9` 个参数，所以每次前向传递使用 `2 * 70e9 * B` FLOPs。假设 40% MFU（FLOPs 利用率），这给我们大约 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s` 的运行时间。与之前我们看到的数字相比，这实际上相当多！

**问题**：假设我们有平均预填充长度为 8192 个标记和平均解码长度为 4096 个标记。如果我们有一个生成批大小为 32。平均每步有多少序列完成解码？平均每步有多少标记从我们的 KV 缓存中移除？

答案

这相当直接。由于我们有平均解码长度为 4096 个标记，一个序列大约每 1 / 4096 个标记完成一次。给定批大小为 32，这意味着每步我们有 `32 / 4096` 个序列被移除。由于我们的 KV 缓存长度大约为 `8192 + 4096`，这意味着每步移除 `32 * (8192 + 4096) / 4096 = 96` 个标记。一般公式是 $B * (P + G) / G$，其中 $P$ 和 $G$ 是预填充和生成长度。

**问题**：假设我们使用平均预填充长度为 8192 和平均解码长度为 512 的解耦服务。假设预填充和生成延迟如上所述在 bfloat16 下计算。你需要多少预填充：生成服务器的比例才能保持两者都完全饱和？

答案

这是一个很有趣的问题。设 $P$ 为预填充服务器的数量，$G$ 为生成服务器的数量。所以一般来说，这是一个管道问题，我们以 `P / prefill_latency` 的速率输入序列，以 `B * G / (generate_latency * median_decode_length)` 的速率消费它们。我们在批大小为 43（让我们称之为 32）的情况下计算了每个预填充步骤 `910ms` 和每个解码步骤 `19ms`。因此，我们需要 `P / 0.91 = 32 * G / (0.019 * 512)` 或 `P = 3G`，也就是说，我们需要比生成服务器多大约 3 倍的预填充服务器！

## 可视化延迟吞吐量权衡

让我们继续关注 LLaMA 70B，实际查看生成过程中不同批量大小的延迟和吞吐量。正如我们在上一节中为 PaLM 模型所展示的，这为我们提供了吞吐量/延迟的帕累托前沿。让我们假设 16 方张量并行性，因为这是我们在 MLP 块中保持计算受限的合理界限。这里我们将使用 TPU v5e 4x4 体系结构。**滑块控制序列长度，你可以看到更大 KV 缓存的影响。**

+   **看看成本和延迟之间的权衡是多么的显著。** 以每-token 延迟加倍为代价，我们可以实现每-token 成本的约 100 倍减少。此外，我们的延迟可以从低批量大小的 5.5ms 到非常大的批量大小的 20ms 不等。

+   注意，在 2k 上下文中，当达到 BS 120 屋顶线（这里因为使用 int8 权重但 bf16 FLOPs，所以是 120）时，吞吐量实际上在 1 token / ms / chip 左右达到平台期。然而，随着序列长度的增加，我们无法将这个批大小放入内存中，所以我们永远不会达到完全饱和的点。

+   注意，在相同吞吐量下，大批量大小的延迟要高得多，因为 KV 加载变得占主导地位（而不是参数加载）。

我们可以通过将成本和延迟的来源分解为参数加载时间、KV 加载时间和 FLOPs 时间来更好地理解这一点。红色区域是我们预期在 MLP 块中计算受限的区域。

这讲述了一个相当有趣的故事。你可以看到，最初，参数加载代表了延迟的绝大部分，直到批量大到足以使 FLOPs 和 KV 加载变得更重要。值得注意的是，在所有大于 2048 的序列长度中，我们在 KV 缓存加载上花费的时间比在 FLOPs 上还要多！**因此，虽然我们可以通过增加批大小来提高硬件利用率，但在长上下文长度中，KV 加载总是主导总步骤时间。**

**总结：** 对于 LLaMA 3-70B，我们在几乎所有这些配置中都是强 KV 缓存内存带宽限制（以及 HBM 限制）的（并且是 HBM 限制），这突出了减少 KV 缓存大小对于生成吞吐量是多么的重要。也请注意，这里的延迟/吞吐量权衡仍然非常显著。

此代码相当简单。

这里是计算这些屋顶线的代码：

```py
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do param_size = 70e9  # int8 means 1 byte per param sequence_length = 8192  # can vary this 
hbm_bandwidth = 8.20E+11  # v5e flops = 1.97E+14  # v5e 
param_size = bytes_per_param * param_count

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80

def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(max_num_chips: int = 16):
  # for num_chips in topo_sizes:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  num_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(num_chips <= max_num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(num_chips, sequence_length, param_size)  # get the largest batch size that can fit batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_count * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense 
mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate 
latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips) 
```

注意我们如何非常明确地将延迟分为两个来源：KV 加载和参数加载，以及延迟是由 FLOPs 还是通信限制，哪个更大。

## 已解决的问题

这里有一些已解决的问题。其中一些重复了上面已经解决的问题，但可能在教学上是有用的。

**问题 1：** LLaMA 3-405B 每个前向传递每-token 使用多少 FLOPs？假设我们是 FLOPs 限制的，那么在 N 个芯片上 TPU v5e 上单个前向传递的下限是多少？如果我们是通信限制的呢？*忽略模型无法适应单个芯片的事实。*

**问题 2：** 假设我们想使用 BS240 和 int8 权重以及 int8 KV 缓存来服务 LLaMA 3-8B。以下内容分别使用了多少字节：（a）模型参数（b）KV 缓存和（c）峰值工作激活（大致数量）。我们可以在最小的拓扑结构上运行这个模型吗？

**问题 3：** 你会如何在 TPU v5e 上服务 LLaMA 3-405B？假设使用 int8 权重和 bfloat16 FLOPs。如果我们有一个严格的 15ms / token 限制，我们能够达到的最高吞吐量配置是什么？理论上的最小步时间是多少？

### 第八部分（Part 8）的内容到此结束！要深入了解 XLA 和 TPU 性能分析（profiling），请点击这里。  ### 杂项

^*在 Google DeepMind 完成的工作，现在在 MatX。

### 引用

在学术环境中进行归属时，请引用以下工作：

```py
 Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025. 
```

或者作为一个 BibTeX 条目：

```py
 @article{scaling-book,
      title = {How to Scale Your Model},
      author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad
      and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
      publisher = {Google DeepMind},
      howpublished = {Online},
      note = {Retrieved from https://jax-ml.github.io/scaling-book/},
      year = {2025}
    } 
``` 
