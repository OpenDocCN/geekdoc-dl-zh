# Keras 中标签缺失的多任务学习

> 原文：<https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/>

###### 发帖人:[程维](/blog/author/Chengwei/) 4 年 10 个月前

([评论](/blog/how-to-multi-task-learning-with-missing-labels-in-keras/#disqus_thread))

![multi-pencils](img/aac592beb5bc4aa5f11494d5f46d2dbd.png)

多任务学习使我们能够训练一个模型同时完成几项任务。

例如，假设一张照片是由自动驾驶汽车拍摄的，我们希望检测图像中不同的东西。**停车标志、交通灯、汽车**等。

在没有多任务学习的情况下，我们必须为我们想要检测的每个对象训练模型，并且利用一个输出来检测或不检测目标对象。

但是使用多任务学习，我们可以只训练一次模型，通过 3 个输出标签来检测是否有任何目标对象被检测到。

![self driving image](img/125abf2b58310b55fd3ff35a5d3c2cba.png)

模型输入是图像，输出有 3 个标签，1 表示检测到特定对象。

![labels](img/75cda17bce0d8c41783744cc0c4b493e.png)

对于在像图像这样的数据集上训练的模型，训练一个模型来执行多个任务比单独训练来单独检测对象的模型执行得更好，因为在训练期间学习的较低级图像特征可以在所有对象类型之间共享。

多任务学习的另一个好处是它允许训练数据输出被部分标记。比方说，我们不标记前面的 3 个物体，而是希望人类标记器在所有给定的图像中标记另外 3 个不同的物体，行人、骑自行车的人、路障。他/她可能最终会感到厌倦，也懒得去标记是否有停车标志或是否有路障。

因此，带标签的训练输出可能看起来像这样，其中我们将未加标签的表示为**-1 "**。

![missing labels](img/d7f243f4bc33589d075e25948af8050d.png)

那么，我们如何用这样的数据集训练我们的模型呢？

关键是损失函数我们要用**【掩码】**标注数据。意味着对于未标记的输出，我们在计算的时不考虑损失函数。

## 多任务学习演示

让我们通过一个具体的例子来训练一个可以执行多任务的 Keras 模型。出于演示的目的，我们建立我们的玩具数据集，因为它更容易训练和可视化的结果。

这里，我们在 2D 空间中随机生成 100，000 个数据点。每个轴都在 0 到 1 的范围内。

对于输出 Y，我们在以下逻辑中有 3 个标签

![demo logics](img/0abfa056a6a8bb65c021a1ff4ed860d3.png)

我们将建立一个模型来发现 X 和 Y 之间的这种关系，

为了让问题更复杂，我们将模拟贴标机掉落一些输出标签。

让我们建立一个 4 层的简单模型，

这里是重要的部分，在这里我们定义了我们的自定义损失函数来**“屏蔽”**仅仅标记数据。

**掩码**将是一个张量，为每个训练样本存储 3 个值，不管标签是否等于我们的**掩码 _ 值** (-1)，

然后，在计算二进制交叉熵损失的过程中，我们只计算那些被屏蔽的损失。

训练很简单，我们先把最后生成的 3000 个数据留作最后的评测测试。

然后在训练期间将剩余的数据分成 90%用于训练，10%用于开发。

经过 2000 个纪元的训练，让我们用预留的评估测试数据来检验模型性能。

损失/精确度为

```py
0.908/0.893
```

为了帮助形象化模型的想法，让我们为我们的 3 个标签中的每一个画出它的决策边界。

![boundaries](img/cb20cc37324fa090dece391ff8fa8fe0.png)

看起来我们的模型搞清楚了 X 和 Y 之间的逻辑；)



如果你不相信我们的自定义损失函数的有效性，让我们并排比较一下。

要禁用我们的自定义损失函数，只需将损失函数改回默认的`'binary_crossentropy'`就像这样。

然后再次运行模型训练和评估。

最终评估的准确度只有 0.527 左右，这比我们之前使用自定义损失函数的模型差得多。

```py
0.909/0.527
```

在我的 GitHub [repo](https://github.com/Tony607/Keras_Multi_task) 上查看源代码。

## 总结和进一步思考

通过多任务学习，我们可以在一组任务上训练模型，这些任务可以受益于共享的低级特征。

通常情况下，你每个任务的数据量都是差不多的。

训练数据中某种程度的标签缺失不是问题，可以用自定义损失函数来处理，只屏蔽有标签的数据。

我们可以做得更多，我想到的一个可能的数据集是 [(MBTI)迈尔斯-布里格斯人格类型数据集](https://www.kaggle.com/datasnaek/mbti-type)。

输入是给定人员发布的文本。

有 4 个输出标签。

*   内向(I)-外向(E)
*   直觉(N)–感觉(S)
*   思考(T)-感受(F)
*   判断(J)-感知(P)

我们可以把每一个都当作一个二元标签。

我们可以允许标注者为给定的人的帖子留下任何未标注的人格类型。

模型应该还是能够用我们自定义的损失函数算出输入和输出之间的关系。

如果你尝试过这个，请在下面留下评论，让我们知道它是否有效。

*   标签:
*   [深度学习](/blog/tag/deep-learning/)，
*   [keras](/blog/tag/keras/) ,
*   [教程](/blog/tag/tutorial/)

[Share on Twitter](https://twitter.com/intent/tweet?url=https%3A//www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/&text=How%20to%20Multi-task%20learning%20with%20missing%20labels%20in%20Keras) [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https://www.dlology.com/blog/how-to-multi-task-learning-with-missing-labels-in-keras/)

*   [←如何在 Keras 中选择优化器的快速说明](/blog/quick-notes-on-how-to-choose-optimizer-in-keras/)
*   [如何使用 Keras 生成真实的 yelp 餐厅评论→](/blog/how-to-generate-realistic-yelp-restaurant-reviews-with-keras/)