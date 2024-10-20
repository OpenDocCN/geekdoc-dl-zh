# 比较 Paperspace 上可用的 Ampere GPUs

> 原文：<https://blog.paperspace.com/ampere-gpus-with-paperspace/>

自从进入 GPU 市场以来，像 A100 这样的 Ampere GPUs 因其在机器和深度学习任务中令人难以置信的能力而受到了好评。由于他们这一代技术的进步，他们一直能够以低得多的成本运行其他类型的机器，并在一些类似的规格(如 GPU 内存)上匹配并超越旧的 GPU。即使与强大的特斯拉 GPU 相比，它们在性能上也不相上下，而价格却只有近一半。

为任何给定的任务选择使用哪种 GPU 总是一个挑战。这篇博客文章将涵盖安培 RTX GPU、A4000、A5000 和 A6000 使用 Paperspace 产品的优势，优于具有类似 GPU 内存值的其他 GPU。由于这是用户选择的第一个常用指标，我们还将建议一些其他统计数据，供您在为 ML/DL 任务选择最佳 GPU 时考虑。我们将从研究安培 RTX GPU 架构及其创新开始，然后深入分析这些 GPU 在实践中与具有类似功能的其他一代 GPU 的比较能力。

## Nvidia 安培 RTX 架构

英伟达安培 RTX 和夸德罗 RTX GPU 系列旨在为专业可视化带来最强大的技术，并显著增强 RTX 芯片在人工智能训练或渲染等任务中的性能。Quadro RTX 系列最初基于图灵微体系结构，具有实时光线跟踪功能。这是通过使用新的 RT 核心来加速的，这些核心旨在处理四叉树和球形层次，并加速与单个三角形的碰撞测试。Ampere GPU 内存还带有纠错码(ECC ),可在不影响计算精度和可靠性的情况下运行处理任务。由于这些升级的功能，安培标签已经成为尖端 GPU 技术的代名词；随着生产线上更先进的机器的发展，如 A100，这种技术得到了进一步的发展。

安培 RTX 图形处理器代表了第二代的夸德罗 RTX 技术。该架构建立在前代 RTX 技术的基础上，与前代产品相比，显著增强了渲染、图形、人工智能和计算工作负载的性能。这些更新的功能包括:

*   第二代光线跟踪内核:正如其[深度学习超级采样](https://developer.nvidia.com/rtx/dlss)范例所示，RT 内核可以大幅提高帧速率，并有助于使用 DLSS 神经网络在可视化任务中生成清晰的图像。这对于在 Paperspace 核心计算机上处理图形的设计人员非常有用。
*   第三代 tensor cores:“新的 Tensor Float 32 (TF32) precision 为 5X 提供了超过上一代产品的训练吞吐量，以加速人工智能和数据科学模型训练，而无需任何代码更改。” [(1)](https://www.nvidia.com/en-us/design-visualization/ampere-architecture/)
*   CUDA 内核:Ampere GPUs 中的 CUDA 内核的能效高达 2 倍。这些实现了单精度 FP 32 运算的双倍速度处理，并实现了深度学习任务的显著性能提升
*   PCI Express Gen 4.0: PCIe Gen 4.0 提供 2 倍于 PCIe Gen 3.0 的带宽，提高了 CPU 内存的数据传输速度，适用于人工智能和数据科学等数据密集型任务。 [(1)](https://www.nvidia.com/en-us/design-visualization/ampere-architecture/) 一篇性能指标评测文章证明了这一点，它显示“PCIe 第四代与 PCIe 第三代相比，读取数据快 61%，而 PCIe 第四代的写入数据快 46%。”[②](https://store.patriotmemory.com/blogs/news/what-is-the-difference-between-pcle-gen-3-gen-4)
*   第三代 NVLink: NVLink 支持 2 个 GPU 连接并共享单个任务的功能和性能。凭借高达 112 千兆字节/秒(GB/s)的双向带宽和高达 96 GB 的组合图形内存，专业人士可以处理最大的渲染、人工智能、虚拟现实和视觉计算工作负载 [(1)](https://www.nvidia.com/en-us/design-visualization/ampere-architecture/)

## 将安培 RTX GPU 与同类产品中的其他 GPU 进行比较

在 Paperspace，您可以访问 4 种不同的安培 GPU:a 4000、A5000、A6000 和 A100。其中的每一个都可以在高达 x 2 的多 GPU 实例中使用。在我们最近的计算机视觉基准测试报告中，我们得出的结论是，就速度和功率而言，A100 是深度学习任务的最佳 GPU，但我们注意到，在实践中，其他安培 RTX GPU 通常更具成本效益。为了涵盖新材料，我们将在本文中更深入地关注 A4000、A5000 和 A6000，并展示为什么这些机器通常是您深度学习任务的最佳选择。

### 安培 GPU 规格

下表比较了安培 RTX GPU 和其他性能相当的 GPU 内存。由于 A6000 没有可比的 GPU，就目前可用的 GPU 内存而言，我们选择将其与 A100 进行比较。


| 国家政治保卫局。参见 OGPU | A4000 | RTX5000 | v100 至 16GB | A5000 | P6000 | v100 至 32GB | A6000 | A100 |
| 产生 | 安培 | 图灵 | 沃尔特河 | 安培 | 帕 | 沃尔特河 | 安培 | 安培 |
|  |  |  |  |  |  |  |  |  |
| CUDA 核心 | Six thousand one hundred and forty-four | Three thousand and seventy-two | Five thousand one hundred and twenty | Eight thousand one hundred and ninety-two | Three thousand eight hundred and forty | Five thousand one hundred and twenty | Ten thousand seven hundred and fifty-two | Six thousand nine hundred and twelve |
| GPU 内存(GB) | Sixteen | Sixteen | Sixteen | Twenty-four | Twenty-four | Thirty-two | Forty-eight | Forty |
| 万亿次浮点运算中的单精度性能(SP FP32) | Nineteen point two | Eleven point two | Fourteen | Twenty-seven point eight | Twelve | Fourteen | Thirty-eight point seven | Nineteen point five |
|  |  |  |  |  |  |  |  |  |
| 内存带宽(GB/s) | Four hundred and forty-eight | Four hundred and forty-eight | Nine hundred | Seven hundred and sixty-eight | Four hundred and thirty-two | Nine hundred | Seven hundred and sixty-eight | One thousand five hundred and fifty-five |
|  |  |  |  |  |  |  |  |  |
| vCPU | eight | eight | eight | eight | eight | eight | eight | Twelve |
| 记忆 | Forty-eight | Thirty-two | Thirty-two | Forty-eight | Thirty-two | Thirty-two | Forty-eight | Ninety-seven |
| 存储包括(GB) | Fifty | Fifty | Fifty | Fifty | Fifty | Fifty | Fifty | Fifty |
| 每小时价格 | $0.76 | $0.82 | $2.30 | $1.38 | $1.10 | $2.30 | $1.89 | $3.09 |
| 每月价格(仅限使用，无订阅) | $0 | $0 | $0 | $0 | $0 | $0 | $0 | $0 |
|  |  |  |  |  |  |  |  |  |
| 此实例所需的订阅价格 | $8.00 | $8.00 | $39.00 | $39.00 | $8.00 | $39.00 | $39.00 | $39.00 |
| 每月价格使用和订阅 | $8 | $8 | $39 | $39 | $8 | $39 | $39 | $39 |
|  |  |  |  |  |  |  |  |  |
| 可供选择 | 是 | 是 | 是 | 是 | 是 | 是 | 是 | 是 |
| 每 GB/分钟的费用(吞吐量) | $0.10 | $0.11 | $0.15 | $0.11 | $0.15 | $0.15 | $0.15 | $0.12 |
| 每 100 个 CUDA 核心的费用(小时) | $0.01 | $0.03 | $0.04 | $0.02 | $0.03 | $0.04 | $0.02 | $0.04 |
| 每内存美元(GB) | $0.05 | $0.05 | $0.14 | $0.06 | $0.05 | $0.07 | $0.04 | $0.08 |
| 每万亿次浮点运算 32 美元 | $0.42 | $0.71 | $2.79 | $1.40 | $0.67 | $2.79 | $1.01 | $2.00 |

安培 RTX GPU 令人难以置信的性能的原因可以在三个地方用数字显示:每个安培 GPU 包含的 CUDA 内核比可比的其他机器更多，单精度性能值明显更高，更重要的是，更多的 CPU 内存。更多的 CUDA 内核直接转化为在任何给定时间并行处理更多数据的能力，单精度性能反映了每秒可以在万亿次浮点运算中完成多少次浮点运算，CPU 内存量有助于 GPU 以外的其他进程，如数据清理或绘图。一起，我们可以开始建立我们的情况下，建议安培 RTX 机器使用在任何可能的地方。

除了这些规格之外，以下是我们最近的基准测试报告中所选 GPU 的相应基准测试。这些都显示了完成的时间，因此我们可以更好地理解这些规格差异在实践中是如何体现的。


|  | A4000 | RTX5000 | V100 至 16GB | A5000 | P5000 | v 1100 至 32GB | A6000 | A100 |
| 黄色 | Eighteen point six five eight | Twenty point zero one six | Nineteen point three three five | Fourteen point one zero seven | Sixty point nine seven six | 17.0447 | Thirteen point eight seven eight | Twelve point four two one |
| StyleGAN_XL (s) | One hundred and six point six six seven | One hundred and seven | Eighty-five point six six seven | Eighty-three | Two hundred and ninety-five | One hundred and five point three three three | Eighty-seven point three three three | Eighty-eight point three three three |
| 高效网络 | Nine hundred and twenty-five | Eight hundred and sixty-seven | Eight hundred and one | Seven hundred and sixty-nine | OOM* | Six hundred and ninety | Six hundred and twenty-seven | Five hundred and twenty-eight |
| 成本(每小时美元) | $0.76 | $0.82 | $2.30 | $1.38 | $0.78 | $2.30 | $1.89 | $3.09 |
| 单次运行成本** | $0.004 | $0.005 | $0.012 | $0.005 | $0.013 | $0.011 | $0.007 | $0.011 |
| StyleGAN_XL 单次运行成本** | $0.02 | $0.02 | $0.05 | $0.03 | $0.06 | $0.07 | $0.05 | $0.08 |
| 单次运行净成本** | $0.20 | $0.20 | $0.51 | $0.29 | 失败！ | $0.44 | $0.33 | $0.45 |

*OOM:内存不足。这表明由于内核中缺少内存资源，训练失败。

**单次运行成本:以秒为单位的时间转换为小时，然后乘以每小时的成本。反映任务在 GPU 上运行一次的成本。

YOLOR 基准测试测量了在短视频上生成图像检测的时间，StyleGAN XL 任务用于在 Pokemon 图像数据集上训练单个时期，而 EfficientNet 基准测试是在 tiny-imagenet-200 上训练单个时期。然后，我们根据它们的每小时成本将每个时间与完成值进行比较。要查看完整的基准测试报告，[请访问此链接。](https://blog.paperspace.com/best-gpu-paperspace-2022/)

#### 何时使用 A4000:

如上所示，A4000 在规格、成本指标和基准测试时间方面都表现非常出色。每小时 0.76 美元，也是本指南中最实惠的选择。它还拥有同类产品中最高的 CUDA 核心数、SP FP32 和 CPU 内存。

虽然吞吐量相对较低，但与 V100- 16 GB 相比，A4000 使用 tiny-imagenet-200 训练 EfficientNet 基准测试模型所需的时间仅为 115.48%，训练 StyleGAN XL 模型所需的时间仅为 124.51%。对于 YOLOR 的探测任务，它实际上更快。这表明，与 V100 - 16GB 相比，A4000 实例提供了极佳的预算选择。此外，总的来说，成本/吞吐量、每 100 个 CUDA 核心的成本以及成本 SP FP32 值是表中所有选项中最好的。

虽然 RTX5000 的性能和成本几乎相同，但 A4000 仍将为您带来更高的价值。从 SP FP32 和 CUDA 内核的差异来看，随着机器的任务变得更加复杂，这一点将变得更加明显。**基于这些累积的证据，我们可以推荐 A4000 作为任何用户在 Gradient 上寻求最具成本效益的 GPU 的首选，运行成本低于每小时 1 美元。**

#### 何时使用 A5000

以每小时 1.38 美元的价格，A5000 代表了一个在纸张空间机器的价格范围中间的有趣的地方。尽管如此，A5000 的 CUDA 内核和 SP 32 在列表中排名第二。它还拥有第三高的 GPU 内存，仅次于 V100 32 GB 和 A6000。

实际上，A5000 在所有基准测试中都表现出色，尤其是与同类产品中的其他产品相比。它超越了 P5000 的检测时间，展示了旧的 Pascale 和新的 Ampere Quadro 机器在性能上的显著差异。

A5000 在所有任务上都优于 V100 32 GB，除了 EfficientNet training，其训练时间是 V100 32GB 的 111.44%。令人惊讶的是，对于 StyleGAN XL 任务，它的平均性能实际上也超过了 A100 和 A6000。由于训练任务本身的特殊性，这可能是一个侥幸，但仍然提供了一个有趣的演示，说明 Ampere 架构在执行这些高度复杂的任务时有多快。

就成本而言，A5000 比同类产品中的其他选项更高效。总体而言，运行任何一项训练或检测任务的平均成本排在第三位，成本与吞吐量之比排在第三位，成本超过 100 个 CUDA 内核。

总而言之，数据表明，A5000 代表了 Gradient 上最强大的 GPU(如 A100、A6000 和 V100，32GB)和最弱的 GPU(如 RTX4000 和 P4000)之间的一个出色的中间选择。每小时 1.38 美元，是对 A4000 的直接升级，每小时额外支付 62 美分。与 A4000 相比，A5000 的速度和内存都有所提高，因此也超过了所有可用的预算 GPU 和同类产品中的所有其他 GPU，这使得 a 5000 成为更复杂任务的绝佳选择，如在预算内培训图像生成器或物体检测器。**因此，当速度和预算必须同等考虑时，我们可以建议用户尝试 A5000，**因为速度的提高以及因此减少的总使用时间不会完全抵消 A4000 增加的成本。

#### 何时使用 A6000

从情况来看，A6000 是 Gradient 上唯一最好的 GPU，A100 是唯一的竞争对手。虽然 A100 的内存少了 8gb，但由于其升级的架构，它的吞吐量明显更高。正是这种令人难以置信的吞吐量使得 A100 对于深度学习任务如此令人惊叹，但每小时 3.09 美元，这是在 Gradient 上使用的最昂贵的单个 GPU(尽管仍然比竞争对手[的 A100 便宜)。](https://aws.amazon.com/ec2/instance-types/p4/)

为了帮助理解 A6000 最适合在哪里使用，以下是针对最大批量培训的[基准报告](https://blog.paperspace.com/best-gpu-paperspace-2022/)的一小部分:


| 国家政治保卫局。参见 OGPU | A6000 | A100 | v 1100 至 32GB |
| 成本(每小时美元) | $1.89 | $3.09 | $2.30 |
| StyleGAN_XL(最大批量) | Thirty-two | Thirty-two | Sixteen |
| StyleGAN_XL(用于批处理项) | 0.0590625 | 0.0965625 | 0.14375 |
| 效率网(最大批量) | Two hundred and fifty-six | One hundred and twenty-eight | One hundred and twenty-eight |
| 效率净值(每批项目美元) | 0.0073828125 | 0.024140625 | 0.01796875 |

48 GB 的 GPU 内存和 10，752 个 CUDA 内核使 A6000 成为处理大量输入和批量数据的最强大的 GPU。这也表明 A6000 GPU 是执行并行处理的最佳 GPU。此外，A6000 在每批次项目的加工成本方面比其竞争对手更具成本效益，这在一定程度上归功于其处理大批量的能力。

A6000 每小时 1.89 美元，比 A100 每小时便宜 1.20 美分。在考虑成本时，A6000 的成本约为 A100 的 2/3，应始终将其视为 A100 的替代产品。其原因反映在完成时间基准时间中。YOLOR 任务和 EfficientNet 任务在 A6000 上运行的时间分别是 A100 的 111.73 %和 118.75%，而 StyleGAN XL 训练在 A6000 上实际上更快。因此，很容易看出，运行一个训练时间大约是训练时间的 115%而成本是 61.16%的模型是一个很好的权衡。

因此，我们可以从这一证据中推断出，A6000 是比稍快的 A100 更具成本效益的替代产品。**我们建议用户在处理特别复杂的任务、大批量时使用 A6000，尤其是在对模型进行长时间训练时。**特别是对于输入量特别大的任务，A6000 可能比 A100 更好。

## 总结想法

在这篇博客文章中，我们研究了安培 RTX GPU 架构，然后将其中三个 GPU a 4000、A5000 和 A6000 与其他具有类似 GPU 内存值的 Paperspace GPUs 进行了比较。我们看到了安培 RTX 架构相对于上一代微架构的改进，然后将它们的规格和基准测试结果与其他上一代 GPU 进行了比较，以了解它们在实际任务中的表现。

我们发现，对于寻求每小时成本低于 1 美元的机器的用户来说，A4000 是 Paperspace 上的最佳 GPU。性能仍然非常出色，而成本只是具有类似规格的 V100 16 GB 的一小部分。接下来，我们确定 A5000 是一款出色的普通 GPU。当速度和成本必须同等考虑时，A5000 是最佳的首选。最后，显示了 A6000 可能是 Paperspace 上最强大的 GPU，其成本不到 A100 的 2/3。我们建议在梯度上处理特别复杂或冗长的任务时使用 A6000，如训练像 [YOLOR 这样的对象检测模型。](https://blog.paperspace.com/yolor/)

有关 Paperspace 机器的更多信息，请访问这里的[我们的文档。](https://docs.paperspace.com/gradient/machines/)

有关 Nvidia Ampere 和 Ampere RTX GPU 的更多信息，请访问以下链接:

*   [https://www . NVIDIA . com/en-us/design-visualization/ampere-architecture/](https://www.nvidia.com/en-us/design-visualization/ampere-architecture/)
*   [https://www . NVIDIA . com/content/PDF/NVIDIA-ampere-ga-102-GPU-architecture-white paper-v2 . PDF](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
*   [https://www.nvidia.com/en-us/data-center/a100/](https://www.nvidia.com/en-us/data-center/a100/)
*   [https://www . NVIDIA . com/content/dam/en-ZZ/Solutions/design-visualization/Quadro-product-literature/proviz-print-NVIDIA-RTX-a 6000-data sheet-us-NVIDIA-1454980-r9-web(1)。pdf](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf)
*   [https://www . NVIDIA . com/content/dam/en-ZZ/Solutions/gtcs 21/RTX-a 5000/NVIDIA-RTX-a 5000-data sheet . pdf](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs21/rtx-a5000/nvidia-rtx-a5000-datasheet.pdf)
*   [https://www . NVIDIA . com/content/dam/en-ZZ/Solutions/gtcs 21/RTX-a 4000/NVIDIA-RTX-a 4000-data sheet . pdf](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs21/rtx-a4000/nvidia-rtx-a4000-datasheet.pdf)

感谢阅读！