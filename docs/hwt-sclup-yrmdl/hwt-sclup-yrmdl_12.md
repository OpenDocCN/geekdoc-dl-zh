# 结论和进一步阅读

> 原文：[`jax-ml.github.io/scaling-book/conclusion`](https://jax-ml.github.io/scaling-book/conclusion)

《如何扩展您的模型》的第十一部分 How To Scale Your Model (第十部分：JAX | 第十二部分：GPU)

感谢您阅读！在这里，我们将包括一些更多参考资料以供进一步学习。  ### 内容

致谢进一步阅读反馈

**感谢您阅读这一系列论文，并祝贺您坚持到最后。** 在我们结束之前，有一些致谢：

## 致谢

这份文档代表了谷歌 DeepMind 许多人的重大集体投资，我们想简要地表示感谢！

+   詹姆斯·布拉德伯里、雷纳·波佩和布莱克·赫奇曼最初从这份手稿中提炼了许多想法，并且是早期理解 Transformer 系统视图的人。

+   詹姆斯·布拉德伯里负责撰写这份文档的第一版，并负责启动这个项目。他比任何人都更负责这份文档的整体叙述。

+   杰克·奥斯汀领导了将这个初版从粗糙笔记转变为更加精致和全面的文档的工作。他做了大量的编辑、格式化和发布文档的工作，并协调了其他作者的贡献。

+   大多数图表和动画都是由安塞尔姆·莱夫塞亚和查理·陈制作的。

+   查理·陈撰写了推理部分并绘制了许多推理图表。

+   罗伊·弗罗斯蒂格帮助了出版、编辑以及旅程中的许多其他步骤。

我们还感谢许多人在整个过程中提供了宝贵的反馈，特别是 Zak Stone、Nikhil Sethi、Caitlin Stanton、Alex Dimitriev、Sridhar Lakshmanamurthy、Albert Magyar、Diwakar Gupta、Jeff Dean、Corry Wang、Matt Johnson、Peter Hawkins 以及许多其他人。感谢高瑞琪在 HTML 格式化方面的帮助。

**感谢大家！**

在你离开之前，你也许会喜欢阅读关于 NVIDIA GPU 的新第十二部分！

## 进一步阅读

有很多相关的写作，包括以下内容：

+   [**TPU 深度解析**](https://henryhmko.github.io/posts/tpu/tpu.html)：这本书精神下的 TPU 架构的深入探讨。

+   [**针对 AI 推理的特定领域架构**](https://fleetwood.dev/posts/domain-specific-architectures)：这本书精神下的硬件和模型深入探讨。

+   [**用于训练深度神经网络的特定领域超级计算机**](https://dl.acm.org/doi/pdf/10.1145/3360307)：这是原始 TPU 论文之一，这里有很多关于谷歌 TPU 项目的详细信息，这里没有涵盖。

+   [**从原理出发让深度学习冷下来**](https://horace.io/brrr_intro.html)：一个更专注于 GPU 和 PyTorch 的教程，关于 LLM rooflines 和性能工程。

+   [**使用 Pallas 编写 TPU 内核**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)：越来越多的 TPU 编程涉及在 Pallas 中编写自定义内核。这个系列讨论了如何编写内核以及许多这里未提及的更低级别的 TPU 细节。

+   [**如何优化 CUDA Matmul 内核以实现 cuBLAS 类似性能：工作日志**](https://siboehm.com/articles/22/CUDA-MMM)：虽然针对 GPU 和 CUDA，但这是一篇出色的博客文章，展示了如何在 CUDA 中优化 matmul 内核。这可能是一个深入了解 TPUs 和 GPUs 如何不同的好机会。

+   [**分布式数组和自动并行化**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)：这是一份关于 JAX 并行 API 的非常棒的指南，也是学习如何实际实现我们在这里讨论的一些想法的好方法。

+   [**Rafi Witten 的 2024 年高性能 LLMs 课程**](https://github.com/rwitten/HighPerfLLMs2024)：我们前同事 Rafi 举办了一门关于 TPU 性能工程的精彩课程，幻灯片全部在 GitHub 上。这涵盖了比我们这里更深入的一些内容。

+   [**[2211.05102] 高效扩展 Transformer 推理**](https://arxiv.org/abs/2211.05102)：一篇关于 Transformer 推理数学的详细论文。这是本文档许多内容的灵感来源。

+   [**Huggingface 超大规模操作手册**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：这本书的 GPU 类似物，更深入地讨论了 PyTorch 在训练期间如何实现并行化技术和内存节省技术。

+   [**Transformer 推理算术**](https://kipp.ly/transformer-inference-arithmetic/)：一篇包含与本书相同许多想法的博客，以及一些优秀的插图。

+   [**斯坦福 CS336 课程幻灯片和视频**](https://stanford-cs336.github.io/spring2025/index.html#coursework)：这是一门精彩的斯坦福课程，涵盖了 LLM 训练和服务的许多细节，还有一些有用的练习。作业 1 和 2 特别相关。

+   [**Stas Bekman 的机器学习工程手册**](https://github.com/stas00/ml-engineering)：一本高度实用的机器学习基础设施指南，涵盖了本书未涉及的主题，如如何与云服务提供商谈判、集群管理和 GPU 吞吐量的经验测量。

在这个领域还有很多综合写作的空间，所以我们希望这份手稿能鼓励更多人进行写作！我们也相信这是一个值得研究和探索的领域。在许多情况下，即使没有很多硬件加速器，也可以完成。

## 反馈

请留下评论或问题，以便我们进一步改进。您可以通过 jacobaustin123 [at] gmail [dot] com 联系我们的通讯作者 Jacob Austin，或者通过在 GitHub 上发布问题、拉取请求或讨论来提出编辑建议 [on GitHub](https://github.com/jax-ml/scaling-book)。

^*在谷歌 DeepMind 完成的工作，现在在 MatX。

### 引用

在学术环境中进行归属时，请引用此作品如下：

```py
 Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025. 
```

或者作为 BibTeX 条目：

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
