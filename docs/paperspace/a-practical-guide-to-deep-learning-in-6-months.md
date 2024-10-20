# 6 个月深度学习实用指南

> 原文：<https://blog.paperspace.com/a-practical-guide-to-deep-learning-in-6-months/>

这篇文章将为你提供学习深度学习的详细路线图，并将帮助你在 6 个月内获得**深度学习实习和全职工作**。这篇文章是实用的，以结果为导向的，遵循自上而下的方法。它是针对时间紧张的初学者以及中级从业者的。

如果你一次又一次地学习 MOOC，像大多数其他教程一样，钻研数学和理论，你只能在 3 个月内建立起你的第一个神经网络。你应该可以很快造出一个来。这篇文章遵循两个阶段的策略，

*   **获得深度学习的高层次想法:**你做初级-中级水平的项目，做不涉及太多数学的课程和理论。
    *   专注-在数学和理论上建立酷的东西+获得深度学习前景的高层次概述。
    *   时间- 3 个月
*   **深入深度学习:**详细阅读关于数学和机器学习的内容。你将开始做一些雄心勃勃的项目，这些项目需要相当多的理论知识，并且需要更大的代码库和更多的功能。
    *   专注的理论和更大的项目。
    *   时间- 3 个月

### 先决条件

*   你懂基本编程。
*   对微积分、线性代数和概率有基本的了解。
*   你愿意每周花 20 个小时。

## 第一阶段

### **学习 Python**

*   做 [Python 速成班](https://www.amazon.com/Python-Crash-Course-Hands-Project-Based/dp/1593276036)。对于 Python 初学者来说，这是一个很棒的资源，非常实用，并且是项目驱动的。它简明扼要。大量的乐趣与最佳实践和宝石。几乎涵盖了用深度学习构建事物所需的所有概念。
*   阅读 [pep8 规则](https://pep8.org/)。知道如何正确地编写 python 并设计其风格是很重要的。

**需要熟悉的重要包装:**

*   **数据角力**
    *   操作系统(用于文件管理)
    *   json(相当多的数据集都是 json 格式的)
    *   Argparse(用于编写简洁的脚本)
    *   Pandas(用于处理 csv 和其他表格数据)
*   **绘图**
    *   OpenCV
    *   Matplotlib
*   **科学堆栈**
    *   NumPy
    *   我的天啊

**时间:1 周**

### 机器学习

*   在深入研究深度学习之前，必须对机器学习有一个很好的了解。
*   在 Coursera 上学习吴恩达的机器学习课程，直到第八周。第 9、10、11 周没有前 8 周重要。前 8 周涵盖必要的理论，第 9、10、11 周面向应用。虽然课程表上注明需要 8 周才能完成，但是 4-6 周完成内容还是很有可能的。课程很好，但是编程作业在 Octave。作为一名机器学习工程师/研究人员，你几乎不会使用 Octave，你的大部分工作肯定会用 Python 来完成。
*   要练习 Python 编程，就去做 Jake Vanderplas 的[机器学习笔记本](https://github.com/jakevdp/sklearn_tutorial/tree/master/notebooks)。它们包含对机器学习的很好的高层次概述和足够的 Python 练习，并向您介绍 scikit-learn，这是一个非常受欢迎的机器学习库。为此你需要安装 Jupyter Lab / Notebook，你可以在这里找到安装和使用说明。
*   至此，你应该对机器学习有了很好的理论和实践理解。考验你技术的时候到了。在 Kaggle 上做[泰坦尼克分类](https://www.kaggle.com/c/titanic)挑战，摆弄数据，即插即用不同的机器学习模型。这是一个学以致用的好平台。

**时间:4-6 周**

### 深度学习

*   能够访问 GPU 来运行任何深度学习实验是很重要的。[谷歌合作实验室](https://colab.research.google.com/notebooks/gpu.ipynb)有免费的 GPU 访问。然而，Colab 可能不是最好的 GPU 解决方案，并且众所周知经常断开连接，可能会滞后。有几个构建自己的 GPU 平台的指南，但最终这是一种干扰，会减慢你的速度。像 AWS 这样的云提供商提供 GPU 实例，但它们的设置和管理非常复杂，这也成为了一个干扰因素。像 [Gradient](https://www.paperspace.com/gradient) (也包括负担得起的 GPU)这样的完全托管服务消除了这种头痛，因此您可以将所有精力集中在成为深度学习开发者上。
*   Do [fast.ai V1](https://course.fast.ai/) ，程序员实用深度学习。这是一门涵盖基础知识的非常好的课程。注重实施而非理论。
*   ****开始阅读研究论文。**** 这是一份很好的清单，列出了[几篇早期和深度学习](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)中的重要论文。它们涵盖了基本原理。

*   **选择 Pytorch / TensorFlow 中的任意一个，开始建造东西**。**对你选择的框架感到非常舒服**。积累丰富的经验，这样你就会变得非常多才多艺，并且知道这个框架的来龙去脉。
    *   PyTorch: 易于实验，不会花很长时间。有大量的教程和大量的社区支持(我的 goto 库),你几乎可以控制管道的每个方面，非常灵活。Fast.ai V1 会给你足够的 PyTorch 经验。
    *   **TensorFlow:** 学习曲线适中，调试困难。拥有比 PyTorch 更多的特性、教程和一个非常强大的社区。
    *   Keras 的很多功能都可以被打倒，而且很容易学习，但是，我总是发现它有太多的黑箱，有时很难定制。但是，如果你是一个寻求建立快速简单的神经网络的初学者，Keras 是聪明的。

*   ****开始在你感兴趣的领域做项目**。**建立良好的形象。领域包括-对象检测，分割，VQA，甘斯，自然语言处理等。构建应用程序并对其进行开源。如果你在学校，找到教授，开始在他们的指导下做研究。以我的经验来看，公司似乎对研究论文和流行的开源库同等重视。

**时间:4-6 周**

现在，你应该，

*   对深度学习有很好的理解。
*   有 2-3 个深度学习的项目。
*   知道如何在一个流行的框架中舒适地构建深度学习模型。

你现在就可以开始申请实习和工作，这就足够了。大多数创业公司关心的是你能建立和优化一个模型有多好，以及你是否具备基本的理论知识。但是要想进入大公司，你需要深入研究，对数学和理论有很好的理解。

## 第二阶段

这就是事情变得有趣的地方。你更深入地钻研理论，从事更大、更有雄心的项目。

### 数学

数学是机器学习的基础，在面试中非常重要。确保你很好地理解了基础知识。

*   ****线性代数:**** 做 [Ch。深度学习书之二](https://www.deeplearningbook.org/contents/linear_algebra.html)。可以用[吉尔伯特·斯特朗的麻省理工开放式课程](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)作为参考。
*   ****微积分:**** [深度学习需要的矩阵微积分](https://arxiv.org/pdf/1802.01528.pdf)是非常好的相关资源。
*   ****概率:**** 阅读更多概率论与统计-[hos sein Pishro-Nik 著《概率、统计与随机过程导论》。](https://www.probabilitycourse.com/)才华横溢。比起任何 MOOC 或教科书，我强烈推荐这个。坚实的理论，重点是简洁，足够的例子和解决方案的问题。接着用 [Ch。深度学习书的 3](https://www.deeplearningbook.org/contents/prob.html)。
*   ****优化:**** 这些来自 [NYU](https://cims.nyu.edu/~cfgranda/pages/DSGA1002_fall15/material/optimization.pdf) 的课程笔记非常值得一读。Coursera 上的第 5 周[机器学习数学也是一个非常好的资源。做](https://www.coursera.org/learn/multivariate-calculus-machine-learning) [Ch。深度学习书籍](https://www.deeplearningbook.org/contents/numerical.html)之 4 固化你的理解。

### 机器学习

*   做某事。深度学习书的 5。这是一本浓缩读物。ML/DL 面试通常有 40-50%是关于机器学习的。
*   **参考:**[Bishop——模式识别与机器学习](https://www.amazon.in/Pattern-Recognition-Learning-Information-Statistics/dp/1493938436?tag=googinhydr18418-21&tag=googinkenshoo-21&ascsubtag=_k_Cj0KCQiA8_PfBRC3ARIsAOzJ2uodznM9nNbfR6WY9jSCQK4FNc3pHsR3xgp6J4Hc8i8WjhYPaliv3rUaAkBfEALw_wcB_k_&gclid=Cj0KCQiA8_PfBRC3ARIsAOzJ2uodznM9nNbfR6WY9jSCQK4FNc3pHsR3xgp6J4Hc8i8WjhYPaliv3rUaAkBfEALw_wcB)(注意，这是一篇比较难的文字！)

### 深度学习

### 

*   在 Coursera 上做深度学习专业化。有 5 道菜
    *   **神经网络和深度学习:**深入主题，将是 fast.ai V1 的良好延续。
    *   **改进深度神经网络:超参数调整、正则化和优化:**这可能是最重要的课程，涵盖了面试中常见的重要话题(批处理、退出、正则化等)
    *   构建机器学习项目:这将教你建立一个 ML 模型，并给你一些实用的技巧。(如果时间紧迫，可以跳过并在以后完成)
    *   卷积神经网络:本课程深入探讨了 CNN 的理论和实际应用。
    *   **序列模型:**探索自然语言模型(LSTMs，GRUs 等)和 NLP，NLU 和 NMT。
*   继续致力于深度学习领域更大、更雄心勃勃的项目。将您的项目推送到 GitHub，并拥有一个活跃的 GitHub 个人资料。
*   了解更多关于深度学习的一个好方法是重新实现一篇论文。重新实现一篇受欢迎的论文(来自 FAIR、DeepMind、Google AI 等大型实验室)会给你很好的体验。

**时间:3 个月**

在这个阶段，你应该有很好的理论理解和足够的深度学习经验。你可以开始申请更好的角色和机会。

### 下一步做什么？

*   如果你有冒险精神，读一读 [Bishop 的《模式识别和机器学习](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)，对机器学习有一个非常好的理解。
*   阅读深度学习书的其余部分(Ch。六通道。12 覆盖相关位)

### Protips

*   浏览 PyTorch 或 TensorFlow **源代码**，看看他们是如何实现基本功能的。此外，Keras 的源代码和结构非常简单，所以您可以以此为起点。
*   **[Cs231n 的作业](http://cs231n.github.io)** 都不错。理解 Dropout、Batchnorm 和 Backprop 的最好方法是用 NumPy 对它们进行编码！
*   以我的经验，面试=数据结构与算法+数学+机器学习+深度学习。粗略的划分会是——数学= 40%，经典机器学习= 30%，深度学习= 30%。
*   真实世界的经历会教会你很多东西。做远程演出(AngelList 是一个很棒的资源)或者部署一个机器学习模型，就像这样:[https://platerecognizer.com/](https://platerecognizer.com/)
*   Jupyter Lab/notebook 非常适合实验和调试，但也有其缺点。在 Jupyter Notebook 上使用标准文本编辑器/IDE (Sublime Text、atom、PyCharm)。它更快，有助于编写良好的、可重复的代码。
*   跟上研究的步伐。为了提高模型的准确性，你需要跟上研究的步伐。深度学习的研究进展非常快。受欢迎的会议包括:
    *   计算机视觉: CVPR，ICCV，ECCV，BMVC。
    *   **机器学习和强化学习(理论上):** NeurIPS，ICML，ICLR
    *   **NLP**ACL、主题图、针

### 其他资源

*   这篇[中型文章](https://blog.usejournal.com/what-i-learned-from-interviewing-at-multiple-ai-companies-and-start-ups-a9620415e4cc)有一个很好的公司申请名单。
*   Shervine Amidi 的[深度学习小抄](https://stanford.edu/~shervine/teaching/cs-230.html)。面试前快速复习的好资源。
*   查看[distilt . pub](http://distill.pub)的酷炫互动文章。