# 用于物体检测的深度学习的最新进展(1)

> 原文：<https://www.dlology.com/blog/recent-advances-in-deep-learning-for-object-detection/>

###### 发帖人:[程维](/blog/author/Chengwei/)三年零四个月前

([评论](/blog/recent-advances-in-deep-learning-for-object-detection/#disqus_thread))

![advance](img/dfe0945e491f3739e2501380337817b8.png)

在训练自定义对象检测模型时， [TensorFlow 对象检测 API](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/) 和 [MMdetection](https://www.dlology.com/blog/how-to-train-an-object-detection-model-with-mmdetection/) (PyTorch)是两个现成的选项，我已经向您展示了如何在 Google Colab 的免费 GPU 资源上完成这项工作。

这两个框架易于使用，配置接口简单，让框架源代码来完成繁重的工作。但是你有没有想过这些年来深度学习对象检测算法是如何进化的，它们的优缺点是什么？

我发现这篇论文- [物体检测深度学习的最新进展](https://arxiv.org/pdf/1908.03673.pdf)很好地回答了这个问题。让我总结一下我所学到的，希望能以更直观的方式阐述。

*文字颜色: **pro** / **cons***

## 检测范例

### 两级检测器

![two-stage](img/c3835d3c31ab4b16b5cea7878125bdfa.png)

 

### 一级检测器

![one-stage](img/3ecb05c81e03a75033fba25ddae9a98f.png)

 

### 主干架构

##   Conclusion and further reading

这篇快速帖子从三个方面总结了深度学习对象检测的最新进展，两阶段检测器、一阶段检测器和主干架构。下次您使用第三方开源框架训练自定义对象检测时，您将更有信心通过检查它们的优缺点来为您的应用程序选择最佳选项。

在下一篇文章中，我将完成我们在论文中留下的内容，即建议生成、特征表示学习和学习策略。如果你感兴趣，强烈建议看一下[论文](https://arxiv.org/pdf/1908.03673.pdf)，它将非常值得你花时间去读。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/recent-advances-in-deep-learning-for-object-detection/&text=Recent%20Advances%20in%20Deep%20Learning%20for%20Object%20Detection%20-%20Part%201) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/recent-advances-in-deep-learning-for-object-detection/)

*   [←如何在 Nvidia Docker 容器中的 Jetson Nano 上运行 Keras 模型](/blog/how-to-run-keras-model-on-jetson-nano-in-nvidia-docker-container/)
*   [用于物体检测的深度学习的最新进展-第二部分→](/blog/recent-advances-in-deep-learning-for-object-detection-part-2/)