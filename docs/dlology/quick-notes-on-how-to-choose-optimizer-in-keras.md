# 如何在 Keras 中选择优化器的快速说明

> 原文：<https://www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 11 个月前

([评论](/blog/quick-notes-on-how-to-choose-optimizer-in-keras/#disqus_thread))

![zig](img/18a8fb7ab59ff34cf63349c3aa93ba3f.png)

TL；博士 ***亚当*** *在实践中效果很好，胜过其他自适应技术。*

对浅层网络使用 SGD+内斯特罗夫，对深网使用 Adam 或 RMSprop。

我参加了 Coursera 的[课程 2:改善深度神经网络](https://www.coursera.org/learn/deep-neural-network/home/welcome)。

这门课的第二周是关于最优化算法的。我发现这有助于发展对不同优化算法如何工作的更好的直觉，即使我们只对将深度学习应用于现实生活中的问题感兴趣。

以下是我从一些研究中学到的一些东西。

### **亚当**

**亚当:自适应力矩估计**

*亚当= RMSprop +动量*

Adam 的一些优势包括:

*   相对较低的内存需求(尽管高于梯度下降和带动量的梯度下降)
*   即使对超参数进行小的调整，通常也能很好地工作。

在 Keras，我们可以这样定义。

 **#### **什么是气势？**

动量将过去的梯度考虑在内，以平滑梯度下降的步骤。它可以应用于批量梯度下降、小批量梯度下降或随机梯度下降。

### **随机梯度下降(SGD)**

在 Keras 中，我们可以这样做来启用 SGD +内斯特罗夫，它适用于浅层网络。

#### **什么是内斯特罗夫的气势？**

涅斯捷罗夫加速梯度(nag)

直觉如何加速梯度下降。

我们希望有一个更智能的球，一个知道自己要去哪里的球，这样它就知道在山坡再次向上倾斜之前减速。

这是一个动画梯度下降与多个优化。

![optimzers](img/c4d4839acdcfecd963bba8e5d1d94b94.png)

*图片来源:[cs 231n](https://cs231n.github.io/neural-networks-3/)*

注意这两个基于动量的优化器(**-绿色动量** ，**-紫色动量 )** 有过冲行为，类似于一个**球**滚下山坡。

与标准动量相比，内斯特罗夫动量的超调较小，因为它采用了下面所示的“ **gamble- > correction** ”方法。

![Nesterov](img/d576e8a71542d33cc5a97d2d75f017bc.png)

### 阿达格勒

它对不频繁的参数进行大的更新，对频繁的参数进行小的更新。因此，**非常适合处理稀疏数据**。

Adagrad 的主要好处是**我们不需要手动调整学习速率**。大多数实现使用默认值 0.01，并保持不变。

**缺点** —

它的主要弱点是学习率总是在降低和衰减。

### **阿达德尔塔**

它是 AdaGrad 的扩展，旨在消除其学习率下降的问题。

AdaDelta 的另一个特点是**我们甚至不需要设置默认的学习速率**。

## 进一步阅读

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)[Keras doc Usage of optimizers](https://keras.io/optimizers/)[Course 2 Improving Deep Neural Networks](https://www.coursera.org/learn/deep-neural-network/home/welcome) from Coursera[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/&text=Quick%20Notes%20on%20How%20to%20choose%20Optimizer%20In%20Keras) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/)

*   [←过拟合模型的两个简单配方](/blog/two-simple-recipes-for-over-fitted-model/)
*   [如何在 Keras 中缺失标签的情况下进行多任务学习→](/blog/how-to-multi-task-learning-with-missing-labels-in-keras/)**